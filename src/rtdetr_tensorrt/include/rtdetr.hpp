//
// Created by niklasrah on 14-4-2026.
// Based on YoloV26-TensorRT/yolo26.hpp
//
#ifndef DETECT_NORMAL_RTDETR_HPP
#define DETECT_NORMAL_RTDETR_HPP

#include <array>
#include <string>
#include <vector>

#include "NvInferPlugin.h"
#include "common.hpp"
#include "fstream"

using namespace det;

//----------------------------------------------------------------------------------------
/**
 * @brief TensorRT inference engine for RT-DETR object detection.
 *
 * Loads a serialized TensorRT engine, allocates GPU/host buffers, runs inference,
 * and post-processes RT-DETR's three-output format:
 *   - labels [N, Q]     int64   — class IDs per query
 *   - boxes  [N, Q, 4]  float32 — xyxy in model input pixel space
 *   - scores [N, Q]     float32 — confidence per query
 *
 * Preprocessing:
 *   - Resize without aspect-ratio preservation (keep_aspect_ratio=False in training).
 *   - BGR → RGB, /255, per-channel mean/std normalisation.
 *
 * Usage:
 *   RTDETR model("rtdetrsp3.engine");
 *   model.MakePipe(true);
 *   model.SetClassNames({"aphid"});
 *   model.CopyFromMat(frame);
 *   model.Infer();
 *   model.PostProcess(objs, 0.25f, 100);
 *   model.DrawObjects(frame, objs);
 */
class RTDETR {
public:
    //------------------------------------------------------------------------------------
    /**
     * @brief Construct detector state from a serialized TensorRT engine file.
     *
     * @param engine_file_path Path to the TensorRT .engine file.
     */
    explicit RTDETR(const std::string& engine_file_path);

    /**
     * @brief Release TensorRT resources, CUDA stream, and allocated buffers.
     */
    ~RTDETR();

    //------------------------------------------------------------------------------------
    /**
     * @brief Allocate inference buffers, bind tensor addresses, and optionally warm up.
     *
     * @param warmup If true, runs several zero-input warmup inferences after setup.
     */
    void MakePipe(bool warmup = true);

    /**
     * @brief Preprocess image using model-derived input size and upload to GPU.
     *
     * @param image Input BGR image.
     */
    void CopyFromMat(const cv::Mat& image);

    /**
     * @brief Run one TensorRT inference pass and copy outputs to host.
     */
    void Infer();

    /**
     * @brief Decode RT-DETR outputs into filtered detections.
     *
     * Outputs are expected to be named "labels", "boxes", "scores".
     * Boxes are in model input pixel coordinates (e.g. [0, 640]) and are rescaled
     * to the original image coordinate space using the stored preprocess ratios.
     *
     * @param objs        Output detection list to populate.
     * @param score_thres Minimum confidence threshold.
     * @param topk        Maximum detections to keep.
     */
    void PostProcess(std::vector<Object>& objs, float score_thres, int topk);

    /**
     * @brief Draw detection boxes and labels onto a BGR image.
     *
     * @param bgr  Image to annotate in place.
     * @param objs Detection objects to render.
     */
    void DrawObjects(cv::Mat& bgr, const std::vector<Object>& objs);

    //------------------------------------------------------------------------------------
    /**
     * @brief Override the default class label names.
     *
     * @param names Vector of class name strings, indexed by class ID.
     */
    void SetClassNames(const std::vector<std::string>& names);

    /**
     * @brief Override the per-channel mean used during normalisation.
     *
     * @param mean RGB mean values (typically in [0, 1]).
     */
    void SetMean(const std::array<float, 3>& mean) { this->mean_rgb = mean; }

    /**
     * @brief Override the per-channel standard deviation used during normalisation.
     *
     * @param stddev RGB stddev values (typically in [0, 1]).
     */
    void SetStddev(const std::array<float, 3>& stddev) { this->std_rgb = stddev; }

    /// Class label names, indexed by class ID. Default: {"aphid"}.
    std::vector<std::string> class_names;

    // Debug accessors: expose score tensor for raw inspection.
    const float*  GetScoresPtr() const;
    const float*  GetBoxesPtr() const;
    size_t        GetNumQueries() const { return num_queries; }

private:
    //--------------------------------------------------------------------------------
    // TensorRT objects
    nvinfer1::ICudaEngine*       engine  = nullptr;
    nvinfer1::IRuntime*          runtime = nullptr;
    nvinfer1::IExecutionContext* context = nullptr;

    // CUDA stream
    cudaStream_t stream = nullptr;

    // Bindings
    std::vector<Binding> input_bindings;
    std::vector<Binding> output_bindings;

    // Buffers (inputs first, then outputs)
    std::vector<void*> device_ptrs;
    std::vector<void*> host_ptrs;  // host pinned outputs only

    // Logger
    Logger gLogger{nvinfer1::ILogger::Severity::kERROR};

    // Binding metadata
    int num_iotensors = 0;
    int num_inputs    = 0;
    int num_outputs   = 0;

    // Indices into output_bindings / host_ptrs for named RT-DETR outputs.
    // -1 means the output wasn't found (detected once at engine-load time).
    int    labels_index = -1;
    int    boxes_index  = -1;
    int    scores_index = -1;
    size_t num_queries  = 0;  // e.g. 300 for rtdetrsp3

    PreParam pparam;

    // Normalisation parameters (training-time values from sly_camera_bringup/rtdetr_bag.launch.py)
    std::array<float, 3> mean_rgb = {0.2377f, 0.3481f, 0.3058f};
    std::array<float, 3> std_rgb  = {0.2299f, 0.2375f, 0.2250f};

    // Helper used during engine-load to discover binding sizes.
    size_t get_size_by_dims_local(const nvinfer1::Dims& d) { return static_cast<size_t>(get_size_by_dims(d)); }

    // Preprocess: resize (no letterbox) + RGB + /255 + mean/std → NCHW float blob.
    void PreprocessResize(const cv::Mat& image, cv::Mat& out, int target_w, int target_h);
};
#endif  // DETECT_NORMAL_RTDETR_HPP
