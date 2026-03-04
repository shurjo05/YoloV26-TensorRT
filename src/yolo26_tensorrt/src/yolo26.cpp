//
// Created by  triple-Mu     on 24-1-2023.
// Modified by Q-engineering on  6-3-2024,
// Modified by niklasrah     on 9-12-2025,
//

#include "yolo26.hpp"
#include <cuda_runtime_api.h>
#include <cuda.h>
#include <string>
#include <vector>

//----------------------------------------------------------------------------------------
//using namespace det;
//----------------------------------------------------------------------------------------
/**
 * @brief Construct detector state from a serialized TensorRT engine file.
 *
 * Loads the engine, creates TensorRT runtime/context objects, creates a CUDA stream,
 * and discovers input/output tensor metadata.
 *
 * @param engine_file_path Path to the TensorRT .engine file.
 * @return No explicit return value (constructor).
 */
YOLO26::YOLO26(const std::string& engine_file_path)
{
    // Default class names — override via SetClassNames() for custom models
    this->class_names = {"aphid"};

    // Read engine file
    std::ifstream file(engine_file_path, std::ios::binary);
    assert(file.good());
    file.seekg(0, std::ios::end);
    std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);
    std::vector<char> trtModelStream(static_cast<size_t>(size));
    file.read(trtModelStream.data(), size);
    file.close();

    // init plugins & runtime
    initLibNvInferPlugins(&this->gLogger, "");
    this->runtime = nvinfer1::createInferRuntime(this->gLogger);
    assert(this->runtime != nullptr);

    // TRT10: deserializeCudaEngine has a third parameter (IHostMemoryProvider* or nullptr)
    this->engine = this->runtime->deserializeCudaEngine(trtModelStream.data(), static_cast<size_t>(size));
    assert(this->engine != nullptr);

    // create context
    this->context = this->engine->createExecutionContext();
    assert(this->context != nullptr);

    // create stream
    CHECK(cudaStreamCreate(&this->stream));

    // discover I/O tensors (name-based)
    this->num_iotensors = this->engine->getNbIOTensors();
    for (int i = 0; i < this->num_iotensors; ++i) {
        const char* tname = this->engine->getIOTensorName(i);
        assert(tname != nullptr);
        nvinfer1::TensorIOMode mode = this->engine->getTensorIOMode(tname);
        nvinfer1::DataType dtype = this->engine->getTensorDataType(tname);
        nvinfer1::Dims dims = this->engine->getTensorShape(tname);  // static shape in engine or profile shape

        det::Binding b;
        b.name = std::string(tname);
        b.dsize = static_cast<size_t>(type_to_size(dtype));
        b.dtype = dtype;
        b.dims = dims;
        b.size = get_size_by_dims_local(dims);

        if (mode == nvinfer1::TensorIOMode::kINPUT) {
            input_bindings.push_back(b);
            ++this->num_inputs;
        } else {
            if (this->primary_output_binding_index < 0) {
                this->primary_output_binding_index = this->num_outputs;
                this->primary_output_dims = dims;
                this->primary_output_dtype = dtype;
                this->primary_output_elements = b.size;
            }
            output_bindings.push_back(b);
            ++this->num_outputs;
        }
    }

}
//----------------------------------------------------------------------------------------
/**
 * @brief Release TensorRT resources, CUDA stream, and allocated host/device buffers.
 *
 * @return No explicit return value (destructor).
 */
YOLO26::~YOLO26()
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
/**
 * @brief Override the default class label names.
 *
 * @param names Vector of class name strings, indexed by class ID.
 */
void YOLO26::SetClassNames(const std::vector<std::string>& names)
{
    this->class_names = names;
}
//----------------------------------------------------------------------------------------
/**
 * @brief Allocate inference buffers, bind tensor addresses, and optionally warm up the pipeline.
 *
 * @param warmup If true, runs several warmup inferences after setup.
 * @return void
 */
void YOLO26::MakePipe(bool warmup)
{
#ifndef CUDART_VERSION
#error CUDART_VERSION Undefined!
#endif

    for (auto& inb : this->input_bindings) {
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
        bool runtime_dims_valid = true;
        for (int i = 0; i < runtime_dims.nbDims; ++i) {
            const int d = static_cast<int>(runtime_dims.d[i]);
            if (d <= 0) {
                runtime_dims_valid = false;
                break;
            }
            runtime_size = (runtime_size == 0) ? static_cast<size_t>(d) : (runtime_size * static_cast<size_t>(d));
        }
        if (runtime_dims_valid && runtime_size > 0) {
            outb.dims = runtime_dims;
            outb.size = runtime_size;
        }
        size_t bytes = outb.size * outb.dsize;
#if (CUDART_VERSION < 11000)
        CHECK(cudaMalloc(&d_ptr, bytes));
#else
        CHECK(cudaMallocAsync(&d_ptr, bytes, this->stream));
#endif
        CHECK(cudaHostAlloc(&h_ptr, bytes, 0));
        this->device_ptrs.push_back(d_ptr);
        this->host_ptrs.push_back(h_ptr);

        if (this->primary_output_binding_index >= 0 && (&outb - &this->output_bindings[0]) == this->primary_output_binding_index) {
            this->primary_output_dims = outb.dims;
            this->primary_output_dtype = outb.dtype;
            this->primary_output_elements = outb.size;
        }
    }

    // Map buffers to tensor names (required for TRT10)
    int devIndex = 0;

    // Assign input tensor addresses
    for (size_t i = 0; i < input_bindings.size(); ++i) {
        const char* name = input_bindings[i].name.c_str();
        void* d_ptr = device_ptrs[devIndex++];
        context->setTensorAddress(name, d_ptr);
    }

    // Assign output tensor addresses
    for (size_t i = 0; i < output_bindings.size(); ++i) {
        const char* name = output_bindings[i].name.c_str();
        void* d_ptr = device_ptrs[devIndex++];
        context->setTensorAddress(name, d_ptr);
    }

    // optional warmup
    if (warmup) {
        for (int k = 0; k < 5; ++k) {
            // zero input buffers and copy
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
 * @brief Resize/pad input image with aspect-ratio preservation and convert to network blob.
 *
 * Updates preprocessing metadata (`pparam`) used later to map detections back to the
 * original image coordinate space.
 *
 * @param image Original input image.
 * @param out Output blob in NCHW float format.
 * @param size Target model input size (width/height).
 * @return void
 */
void YOLO26::Letterbox(const cv::Mat& image, cv::Mat& out, cv::Size& size)
{
    const float inp_h  = size.height;
    const float inp_w  = size.width;
    float       height = static_cast<float>(image.rows);
    float       width  = static_cast<float>(image.cols);

    float r    = std::min(inp_h / height, inp_w / width);
    int   padw = std::round(width * r);
    int   padh = std::round(height * r);

    cv::Mat tmp;
    if ((int)width != padw || (int)height != padh) {
        cv::resize(image, tmp, cv::Size(padw, padh));
    }
    else {
        tmp = image.clone();
    }

    float dw = inp_w - padw;
    float dh = inp_h - padh;

    dw /= 2.0f;
    dh /= 2.0f;
    int top    = int(std::round(dh - 0.1f));
    int bottom = int(std::round(dh + 0.1f));
    int left   = int(std::round(dw - 0.1f));
    int right  = int(std::round(dw + 0.1f));

    cv::copyMakeBorder(tmp, tmp, top, bottom, left, right, cv::BORDER_CONSTANT, {114, 114, 114});

    cv::dnn::blobFromImage(tmp, out, 1 / 255.f, cv::Size(), cv::Scalar(0, 0, 0), true, false, CV_32F);
    this->pparam.ratio  = 1 / r;
    this->pparam.dw     = dw;
    this->pparam.dh     = dh;
    this->pparam.height = height;
    this->pparam.width  = width;
}
//----------------------------------------------------------------------------------------
/**
 * @brief Preprocess image using model-derived input size and upload to GPU input buffer.
 *
 * @param image Input image to preprocess and upload.
 * @return void
 */
void YOLO26::CopyFromMat(const cv::Mat& image)
{
    cv::Mat nchw;
    // assume single input
    assert(this->input_bindings.size() > 0);
    auto& in_binding = this->input_bindings[0];
    // Get target width/height from input binding dims (expect dims format: {N,C,H,W} or nbDims==4)
    int width = (in_binding.dims.nbDims >= 4) ? in_binding.dims.d[3] : 640;
    int height = (in_binding.dims.nbDims >= 4) ? in_binding.dims.d[2] : 640;
    cv::Size size{width, height};
    this->Letterbox(image, nchw, size);

    // set input shape by name (TensorRT 10 name-based API)
    nvinfer1::Dims4 shape{1, 3, height, width};
    this->context->setInputShape(in_binding.name.c_str(), shape);

    size_t bytes = nchw.total() * nchw.elemSize();
    CHECK(cudaMemcpyAsync(this->device_ptrs[0], nchw.ptr<float>(), bytes, cudaMemcpyHostToDevice, this->stream));
}
//----------------------------------------------------------------------------------------
/**
 * @brief Preprocess image using explicit input size and upload to GPU input buffer.
 *
 * @param image Input image to preprocess and upload.
 * @param size Desired input width/height.
 * @return void
 */
void YOLO26::CopyFromMat(const cv::Mat& image, cv::Size& size)
{
    cv::Mat nchw;
    this->Letterbox(image, nchw, size);

    assert(this->input_bindings.size() > 0);
    auto& in_binding = this->input_bindings[0];
    nvinfer1::Dims4 shape{1, 3, size.height, size.width};
    this->context->setInputShape(in_binding.name.c_str(), shape);

    size_t bytes = nchw.total() * nchw.elemSize();
    CHECK(cudaMemcpyAsync(this->device_ptrs[0], nchw.ptr<float>(), bytes, cudaMemcpyHostToDevice, this->stream));
}
//----------------------------------------------------------------------------------------
/**
 * @brief Run one TensorRT inference and copy output tensors from device to host buffers.
 *
 * @return void
 */
void YOLO26::Infer()
{
    // On TRT10 use enqueueV3 (name-based tensors still require raw device pointer array)
    this->context->enqueueV3(this->stream);
    // copy outputs back to host; outputs start at index num_inputs
    for (int i = 0; i < this->num_outputs; ++i) {
        size_t osize = this->output_bindings[i].size * this->output_bindings[i].dsize;
        void* devPtr = this->device_ptrs[i + this->num_inputs];
        void* hostPtr = this->host_ptrs[i];
        CHECK(cudaMemcpyAsync(hostPtr, devPtr, osize, cudaMemcpyDeviceToHost, this->stream));
    }
    CHECK(cudaStreamSynchronize(this->stream));
}
//----------------------------------------------------------------------------------------
/**
 * @brief Infer class count for YOLO26 end-to-end tuple outputs.
 *
 * @return Detected class count, or 0 when class count cannot be inferred.
 */
int YOLO26::GetNumClasses() const
{
    if (this->primary_output_binding_index < 0 ||
        this->primary_output_binding_index >= static_cast<int>(this->output_bindings.size())) {
        return 0;
    }

    const auto& dims = this->primary_output_dims;
    if (dims.nbDims < 2) return 0;

    const int tuple_size = static_cast<int>(dims.d[dims.nbDims - 1]);
    if (tuple_size < 6) return 0;

    return static_cast<int>(this->class_names.size());
}
//----------------------------------------------------------------------------------------
/**
 * @brief Convert raw model outputs into filtered detections.
 *
 * Expects tuple-style YOLO26 end-to-end outputs:
 * (x1, y1, x2, y2, confidence, class_id[, extra...]).
 *
 * @param objs Output detection list to populate.
 * @param score_thres Minimum confidence threshold.
 * @param iou_thres Unused for end-to-end YOLO26 output (NMS handled by model/export).
 * @param topk Maximum number of detections to keep.
 * @param num_labels Unused for end-to-end YOLO26 output.
 * @return void
 */
void YOLO26::PostProcess(std::vector<Object>& objs, float score_thres, float iou_thres, int topk, int num_labels)
{
    (void)iou_thres;
    (void)num_labels;
    objs.clear();
    assert(this->output_bindings.size() > 0);

    auto& dw     = this->pparam.dw;
    auto& dh     = this->pparam.dh;
    auto& width  = this->pparam.width;
    auto& height = this->pparam.height;
    auto& ratio  = this->pparam.ratio;

    if (this->primary_output_binding_index < 0 ||
        this->primary_output_binding_index >= static_cast<int>(this->output_bindings.size()) ||
        this->primary_output_binding_index >= static_cast<int>(this->host_ptrs.size())) {
        return;
    }

    const auto& dims = this->primary_output_dims;
    if (dims.nbDims < 2) {
        return;
    }
    const int tuple_size = static_cast<int>(dims.d[dims.nbDims - 1]);
    if (tuple_size < 6) {
        return;
    }
    if (this->primary_output_dtype != nvinfer1::DataType::kFLOAT) {
        return;
    }
    if (this->primary_output_elements == 0) {
        return;
    }

    const size_t det_count = this->primary_output_elements / static_cast<size_t>(tuple_size);
    const float* dets_ptr = static_cast<float*>(this->host_ptrs[this->primary_output_binding_index]);
    for (size_t i = 0; i < det_count; ++i) {
        const float* row = dets_ptr + i * static_cast<size_t>(tuple_size);
        const float score = row[4];
        if (score < score_thres) continue;

        Object obj;
        const float x0 = clamp((row[0] - dw) * ratio, 0.f, width);
        const float y0 = clamp((row[1] - dh) * ratio, 0.f, height);
        const float x1 = clamp((row[2] - dw) * ratio, 0.f, width);
        const float y1 = clamp((row[3] - dh) * ratio, 0.f, height);
        obj.rect.x = x0;
        obj.rect.y = y0;
        obj.rect.width = std::max(0.f, x1 - x0);
        obj.rect.height = std::max(0.f, y1 - y0);
        obj.prob = score;
        obj.label = static_cast<int>(row[5]);
        objs.push_back(obj);
        if ((int)objs.size() >= topk) break;
    }
}
//----------------------------------------------------------------------------------------
/**
 * @brief Draw detection boxes and class/confidence labels onto a BGR image.
 *
 * @param bgr Image to annotate in place.
 * @param objs Detection objects to render.
 * @return void
 */
void YOLO26::DrawObjects(cv::Mat& bgr, const std::vector<Object>& objs)
{
    char text[256];

    for (auto& obj : objs) {
        cv::rectangle(bgr, obj.rect, cv::Scalar(255, 0, 0));

        std::string label_name = (obj.label >= 0 && obj.label < (int)this->class_names.size())
                                 ? this->class_names[obj.label]
                                 : ("class_" + std::to_string(obj.label));
        sprintf(text, "%s %.1f%%", label_name.c_str(), obj.prob * 100);

        int      baseLine   = 0;
        cv::Size label_size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);

        int x = (int)obj.rect.x;
        int y = (int)obj.rect.y - label_size.height - baseLine;
        if (y < 0)        y = 0;
        if (y > bgr.rows) y = bgr.rows;
        if (x + label_size.width > bgr.cols) x = bgr.cols - label_size.width;

        cv::rectangle(bgr, cv::Rect(cv::Point(x, y), cv::Size(label_size.width, label_size.height + baseLine)), cv::Scalar(255, 255, 255), -1);
        cv::putText(bgr, text, cv::Point(x, y + label_size.height), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0));
    }
}
//----------------------------------------------------------------------------------------
