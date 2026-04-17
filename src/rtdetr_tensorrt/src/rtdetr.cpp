//
// RT-DETR TensorRT inference engine.
//
// Loads an RT-DETR .engine, runs inference, and decodes the three-output format:
//   labels (int64), boxes (float32 xyxy), scores (float32)
//
// Created by niklasrah on 14-4-2026.
//

#include "rtdetr.hpp"
#include <algorithm>
#include <cstdint>
#include <cuda_runtime_api.h>
#include <cuda.h>
#include <string>
#include <vector>

//----------------------------------------------------------------------------------------
/**
 * @brief Load engine file, init TRT runtime/context, discover binding metadata.
 *
 * Unlike YOLO26 (which has a single output tuple tensor), RT-DETR has three named
 * outputs — we locate them by name so we don't rely on engine ordering.
 */
RTDETR::RTDETR(const std::string& engine_file_path)
{
    this->class_names = {"aphid"};

    // Read engine file. Release builds strip assert(), so check explicitly —
    // otherwise a missing file flows into std::vector(size_t(-1)) → std::length_error.
    std::ifstream file(engine_file_path, std::ios::binary);
    if (!file.good()) {
        throw std::runtime_error("RTDETR: cannot open engine file: " + engine_file_path);
    }
    file.seekg(0, std::ios::end);
    std::streamsize size = file.tellg();
    if (size <= 0) {
        throw std::runtime_error("RTDETR: engine file is empty or unreadable: " + engine_file_path);
    }
    file.seekg(0, std::ios::beg);
    std::vector<char> trtModelStream(static_cast<size_t>(size));
    file.read(trtModelStream.data(), size);
    file.close();

    // Init plugins & runtime
    initLibNvInferPlugins(&this->gLogger, "");
    this->runtime = nvinfer1::createInferRuntime(this->gLogger);
    assert(this->runtime != nullptr);

    this->engine = this->runtime->deserializeCudaEngine(trtModelStream.data(), static_cast<size_t>(size));
    assert(this->engine != nullptr);

    this->context = this->engine->createExecutionContext();
    assert(this->context != nullptr);

    CHECK(cudaStreamCreate(&this->stream));

    // Discover I/O tensors
    this->num_iotensors = this->engine->getNbIOTensors();
    for (int i = 0; i < this->num_iotensors; ++i) {
        const char* tname = this->engine->getIOTensorName(i);
        assert(tname != nullptr);
        nvinfer1::TensorIOMode mode  = this->engine->getTensorIOMode(tname);
        nvinfer1::DataType     dtype = this->engine->getTensorDataType(tname);
        nvinfer1::Dims         dims  = this->engine->getTensorShape(tname);

        det::Binding b;
        b.name  = std::string(tname);
        b.dsize = static_cast<size_t>(type_to_size(dtype));
        b.dtype = dtype;
        b.dims  = dims;
        b.size  = get_size_by_dims_local(dims);

        if (mode == nvinfer1::TensorIOMode::kINPUT) {
            input_bindings.push_back(b);
            ++this->num_inputs;
        } else {
            output_bindings.push_back(b);
            ++this->num_outputs;
        }
    }

    // Resolve named outputs (labels/boxes/scores). These are the standard names used
    // by the RT-DETR ONNX exports in this project; adjust if your export uses different names.
    for (int i = 0; i < this->num_outputs; ++i) {
        const std::string& n = this->output_bindings[i].name;
        if (n == "labels") this->labels_index = i;
        else if (n == "boxes")  this->boxes_index  = i;
        else if (n == "scores") this->scores_index = i;
    }

    if (this->scores_index >= 0) {
        const auto& d = this->output_bindings[this->scores_index].dims;
        if (d.nbDims >= 2) {
            this->num_queries = static_cast<size_t>(d.d[d.nbDims - 1]);
        }
    }
}
//----------------------------------------------------------------------------------------
RTDETR::~RTDETR()
{
    delete this->context;
    delete this->engine;
    delete this->runtime;
    cudaStreamDestroy(this->stream);
    for (auto& ptr : this->device_ptrs) {
        CHECK(cudaFree(ptr));
    }
    for (auto& ptr : this->host_ptrs) {
        CHECK(cudaFreeHost(ptr));
    }
}
//----------------------------------------------------------------------------------------
void RTDETR::SetClassNames(const std::vector<std::string>& names)
{
    this->class_names = names;
}
//----------------------------------------------------------------------------------------
/**
 * @brief Allocate device/host buffers, wire up tensor addresses, optional warmup.
 */
void RTDETR::MakePipe(bool warmup)
{
#ifndef CUDART_VERSION
#error CUDART_VERSION Undefined!
#endif

    // RT-DETR ONNX exports typically have a dynamic batch dim (['N', 3, 640, 640]).
    // Resolve batch -> 1 and set the input shape on the context BEFORE allocating, so
    // the output tensor shapes become concrete (otherwise they contain -1 and we'd
    // compute a garbage allocation size that corrupts the heap).
    for (auto& inb : this->input_bindings) {
        nvinfer1::Dims shape = inb.dims;
        size_t in_size = 0;
        for (int i = 0; i < shape.nbDims; ++i) {
            if (shape.d[i] <= 0) {
                // Dynamic dim — batch is the only dynamic dim we expect; force it to 1.
                shape.d[i] = 1;
            }
            in_size = (in_size == 0) ? static_cast<size_t>(shape.d[i])
                                     : (in_size * static_cast<size_t>(shape.d[i]));
        }
        inb.dims = shape;
        inb.size = in_size;

        this->context->setInputShape(inb.name.c_str(), shape);

        void* d_ptr = nullptr;
        size_t bytes = inb.size * inb.dsize;
#if (CUDART_VERSION < 11000)
        CHECK(cudaMalloc(&d_ptr, bytes));
#else
        CHECK(cudaMallocAsync(&d_ptr, bytes, this->stream));
#endif
        this->device_ptrs.push_back(d_ptr);
    }

    for (auto& outb : this->output_bindings) {
        void* d_ptr = nullptr;
        void* h_ptr = nullptr;
        nvinfer1::Dims runtime_dims = this->context->getTensorShape(outb.name.c_str());
        size_t runtime_size = 0;
        for (int i = 0; i < runtime_dims.nbDims; ++i) {
            const int d = static_cast<int>(runtime_dims.d[i]);
            if (d <= 0) {
                throw std::runtime_error(
                    "RTDETR: output tensor '" + outb.name +
                    "' has unresolved dim after setInputShape — engine may be malformed");
            }
            runtime_size = (runtime_size == 0) ? static_cast<size_t>(d)
                                               : (runtime_size * static_cast<size_t>(d));
        }
        outb.dims = runtime_dims;
        outb.size = runtime_size;

        size_t bytes = outb.size * outb.dsize;
#if (CUDART_VERSION < 11000)
        CHECK(cudaMalloc(&d_ptr, bytes));
#else
        CHECK(cudaMallocAsync(&d_ptr, bytes, this->stream));
#endif
        CHECK(cudaHostAlloc(&h_ptr, bytes, 0));
        this->device_ptrs.push_back(d_ptr);
        this->host_ptrs.push_back(h_ptr);
    }

    // Refresh num_queries from the resized scores binding
    if (this->scores_index >= 0) {
        const auto& d = this->output_bindings[this->scores_index].dims;
        if (d.nbDims >= 2) {
            this->num_queries = static_cast<size_t>(d.d[d.nbDims - 1]);
        }
    }

    // Bind buffers by tensor name (TRT10 requirement)
    int devIndex = 0;
    for (size_t i = 0; i < input_bindings.size(); ++i) {
        const char* name = input_bindings[i].name.c_str();
        void* d_ptr = device_ptrs[devIndex++];
        context->setTensorAddress(name, d_ptr);
    }
    for (size_t i = 0; i < output_bindings.size(); ++i) {
        const char* name = output_bindings[i].name.c_str();
        void* d_ptr = device_ptrs[devIndex++];
        context->setTensorAddress(name, d_ptr);
    }

    if (warmup) {
        for (int k = 0; k < 5; ++k) {
            for (size_t i = 0; i < this->input_bindings.size(); ++i) {
                size_t bytes = this->input_bindings[i].size * this->input_bindings[i].dsize;
                std::vector<char> zeros(bytes, 0);
                CHECK(cudaMemcpyAsync(this->device_ptrs[i], zeros.data(), bytes, cudaMemcpyHostToDevice, this->stream));
            }
            this->Infer();
        }
    }
}
//----------------------------------------------------------------------------------------
/**
 * @brief RT-DETR preprocessing: resize → RGB → /255 → (x - mean) / stddev → NCHW float.
 *
 * No letterboxing. The isaac_ros_rtdetr pipeline uses keep_aspect_ratio=False, which
 * matches how the model was trained — so we do a direct cv::resize to (target_w, target_h).
 * This gives non-uniform x/y scaling, which we record in pparam.ratio_x / ratio_y to rescale
 * boxes back to the original image afterwards.
 */
void RTDETR::PreprocessResize(const cv::Mat& image, cv::Mat& out, int target_w, int target_h)
{
    const float height = static_cast<float>(image.rows);
    const float width  = static_cast<float>(image.cols);

    this->pparam.ratio_x = width  / static_cast<float>(target_w);
    this->pparam.ratio_y = height / static_cast<float>(target_h);
    this->pparam.dw      = 0.0f;
    this->pparam.dh      = 0.0f;
    this->pparam.height  = height;
    this->pparam.width   = width;

    // Resize (no aspect preservation) → RGB → float [0, 1]
    cv::Mat resized;
    cv::resize(image, resized, cv::Size(target_w, target_h));
    cv::Mat rgb;
    cv::cvtColor(resized, rgb, cv::COLOR_BGR2RGB);
    cv::Mat rgb_f;
    rgb.convertTo(rgb_f, CV_32FC3, 1.0 / 255.0);

    // Per-channel normalisation: (x - mean) / stddev
    // Split into channels, apply, merge back.
    std::vector<cv::Mat> ch(3);
    cv::split(rgb_f, ch);
    for (int c = 0; c < 3; ++c) {
        ch[c] = (ch[c] - this->mean_rgb[c]) / this->std_rgb[c];
    }
    cv::Mat normalized;
    cv::merge(ch, normalized);

    // HWC → NCHW via blobFromImage (no further scaling or BGR swap — already done).
    cv::dnn::blobFromImage(normalized, out, 1.0, cv::Size(), cv::Scalar(0, 0, 0), false, false, CV_32F);
}
//----------------------------------------------------------------------------------------
void RTDETR::CopyFromMat(const cv::Mat& image)
{
    assert(this->input_bindings.size() > 0);
    auto& in_binding = this->input_bindings[0];

    const int height = (in_binding.dims.nbDims >= 4) ? in_binding.dims.d[2] : 640;
    const int width  = (in_binding.dims.nbDims >= 4) ? in_binding.dims.d[3] : 640;

    cv::Mat nchw;
    this->PreprocessResize(image, nchw, width, height);

    nvinfer1::Dims4 shape{1, 3, height, width};
    this->context->setInputShape(in_binding.name.c_str(), shape);

    size_t bytes = nchw.total() * nchw.elemSize();
    CHECK(cudaMemcpyAsync(this->device_ptrs[0], nchw.ptr<float>(), bytes, cudaMemcpyHostToDevice, this->stream));
}
//----------------------------------------------------------------------------------------
void RTDETR::Infer()
{
    this->context->enqueueV3(this->stream);

    // Copy outputs back to host (outputs start at index num_inputs in device_ptrs)
    for (int i = 0; i < this->num_outputs; ++i) {
        size_t osize = this->output_bindings[i].size * this->output_bindings[i].dsize;
        void* devPtr  = this->device_ptrs[i + this->num_inputs];
        void* hostPtr = this->host_ptrs[i];
        CHECK(cudaMemcpyAsync(hostPtr, devPtr, osize, cudaMemcpyDeviceToHost, this->stream));
    }
    CHECK(cudaStreamSynchronize(this->stream));
}
//----------------------------------------------------------------------------------------
const float* RTDETR::GetScoresPtr() const
{
    if (this->scores_index < 0) return nullptr;
    return static_cast<const float*>(this->host_ptrs[this->scores_index]);
}
//----------------------------------------------------------------------------------------
const float* RTDETR::GetBoxesPtr() const
{
    if (this->boxes_index < 0) return nullptr;
    return static_cast<const float*>(this->host_ptrs[this->boxes_index]);
}
//----------------------------------------------------------------------------------------
/**
 * @brief Decode RT-DETR outputs (labels/boxes/scores) into filtered detections.
 *
 * Labels tensor dtype is int64 (from ONNX). Boxes are xyxy in model input pixel space;
 * we rescale them back to the original image using pparam.ratio_x / ratio_y.
 */
void RTDETR::PostProcess(std::vector<Object>& objs, float score_thres, int topk)
{
    objs.clear();

    if (this->scores_index < 0 || this->boxes_index < 0 || this->labels_index < 0) {
        return;  // malformed engine — shouldn't happen if load succeeded
    }

    const float*   scores = static_cast<const float*>(this->host_ptrs[this->scores_index]);
    const float*   boxes  = static_cast<const float*>(this->host_ptrs[this->boxes_index]);
    const int64_t* labels = static_cast<const int64_t*>(this->host_ptrs[this->labels_index]);

    if (!scores || !boxes || !labels || this->num_queries == 0) return;

    const float ratio_x = this->pparam.ratio_x;
    const float ratio_y = this->pparam.ratio_y;
    const float orig_w  = this->pparam.width;
    const float orig_h  = this->pparam.height;

    // Collect all above-threshold detections, sorted by score descending, then topk.
    struct Tmp {
        float x0, y0, x1, y1, score;
        int   label;
    };
    std::vector<Tmp> kept;
    kept.reserve(this->num_queries);

    for (size_t q = 0; q < this->num_queries; ++q) {
        const float s = scores[q];
        if (s < score_thres) continue;

        const float* b = boxes + q * 4;
        Tmp t;
        t.x0    = clamp(b[0] * ratio_x, 0.f, orig_w);
        t.y0    = clamp(b[1] * ratio_y, 0.f, orig_h);
        t.x1    = clamp(b[2] * ratio_x, 0.f, orig_w);
        t.y1    = clamp(b[3] * ratio_y, 0.f, orig_h);
        t.score = s;
        t.label = static_cast<int>(labels[q]);
        kept.push_back(t);
    }

    std::sort(kept.begin(), kept.end(), [](const Tmp& a, const Tmp& b) { return a.score > b.score; });
    if (static_cast<int>(kept.size()) > topk) kept.resize(topk);

    for (const auto& t : kept) {
        Object obj;
        obj.rect.x      = t.x0;
        obj.rect.y      = t.y0;
        obj.rect.width  = std::max(0.f, t.x1 - t.x0);
        obj.rect.height = std::max(0.f, t.y1 - t.y0);
        obj.prob        = t.score;
        obj.label       = t.label;
        objs.push_back(obj);
    }
}
//----------------------------------------------------------------------------------------
void RTDETR::DrawObjects(cv::Mat& bgr, const std::vector<Object>& objs)
{
    char text[256];

    for (const auto& obj : objs) {
        cv::rectangle(bgr, obj.rect, cv::Scalar(0, 255, 0));

        std::string label_name = (obj.label >= 0 && obj.label < static_cast<int>(this->class_names.size()))
                                 ? this->class_names[obj.label]
                                 : ("class_" + std::to_string(obj.label));
        sprintf(text, "%s %.1f%%", label_name.c_str(), obj.prob * 100);

        int      baseLine   = 0;
        cv::Size label_size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);

        int x = (int)obj.rect.x;
        int y = (int)obj.rect.y - label_size.height - baseLine;
        if (y < 0) y = 0;
        if (y > bgr.rows) y = bgr.rows;
        if (x + label_size.width > bgr.cols) x = bgr.cols - label_size.width;

        cv::rectangle(bgr, cv::Rect(cv::Point(x, y), cv::Size(label_size.width, label_size.height + baseLine)),
                      cv::Scalar(255, 255, 255), -1);
        cv::putText(bgr, text, cv::Point(x, y + label_size.height), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0));
    }
}
//----------------------------------------------------------------------------------------
