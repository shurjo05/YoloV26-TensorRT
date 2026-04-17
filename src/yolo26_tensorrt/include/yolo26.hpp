//
// Created by  triple-Mu     on 24-1-2023.
// Modified by Q-engineering on  6-3-2024,
// Modified by niklasrah     on 9-12-2025,
//
#ifndef DETECT_NORMAL_YOLO26_HPP
#define DETECT_NORMAL_YOLO26_HPP

#include "NvInferPlugin.h"
#include "common.hpp"
#include "fstream"

using namespace det;

//----------------------------------------------------------------------------------------
/**
 * @brief TensorRT inference engine for YOLO26 object detection.
 *
 * Loads a serialized TensorRT engine, allocates GPU/host buffers, runs inference,
 * and post-processes NMS-free end-to-end detection outputs.
 *
 * Usage:
 *   YOLO26 model("yolo26.engine");
 *   model.MakePipe(true);             // allocate + warmup
 *   model.SetClassNames({"aphid"});   // configure labels
 *   model.CopyFromMat(frame, size);   // preprocess + upload
 *   model.Infer();                    // run inference
 *   model.PostProcess(objs, ...);     // decode outputs
 *   model.DrawObjects(frame, objs);   // annotate image
 */
class YOLO26 {
    public:
        //--------------------------------------------------------------------------------
        /**
         * @brief Construct detector state from a serialized TensorRT engine file.
         *
         * @param engine_file_path Path to the TensorRT .engine file.
         */
        explicit YOLO26(const std::string& engine_file_path);

        /**
         * @brief Release TensorRT resources, CUDA stream, and allocated buffers.
         */
        ~YOLO26();

        //--------------------------------------------------------------------------------
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
         * @brief Preprocess image using explicit input size and upload to GPU.
         *
         * @param image Input BGR image.
         * @param size  Desired model input width/height.
         */
        void CopyFromMat(const cv::Mat& image, cv::Size& size);

        /**
         * @brief Resize/pad image with aspect-ratio preservation into a network blob.
         *
         * @param image Original input image.
         * @param out   Output blob in NCHW float format.
         * @param size  Target model input size (width/height).
         */
        void Letterbox(const cv::Mat& image, cv::Mat& out, cv::Size& size);

        /**
         * @brief Run one TensorRT inference pass and copy outputs to host.
         */
        void Infer();

        /**
         * @brief Infer class count from primary output tensor shape.
         *
         * @return Detected class count, or 0 if it cannot be inferred.
         */
        int GetNumClasses() const;

        /**
         * @brief Decode raw model outputs into filtered detections.
         *
         * Expects NMS-free tuple outputs: (x1, y1, x2, y2, confidence, class_id).
         *
         * @param objs        Output detection list to populate.
         * @param score_thres Minimum confidence threshold.
         * @param iou_thres   Unused (NMS handled by model export).
         * @param topk        Maximum detections to keep.
         * @param num_labels  Unused (class count derived from class_names).
         */
        void PostProcess(std::vector<Object>& objs, float score_thres, float iou_thres, int topk, int num_labels  = 80);

        /**
         * @brief Draw detection boxes and labels onto a BGR image.
         *
         * @param bgr  Image to annotate in place.
         * @param objs Detection objects to render.
         */
        void DrawObjects(cv::Mat& bgr, const std::vector<Object>& objs);

        //--------------------------------------------------------------------------------
        /**
         * @brief Override the default class label names.
         *
         * Call this after construction to configure labels for your model
         * (e.g. {"aphid"} for single-class, or multiple classes for future models).
         *
         * @param names Vector of class name strings, indexed by class ID.
         */
        void SetClassNames(const std::vector<std::string>& names);

        /// Class label names, indexed by class ID. Default: {"aphid"}.
        std::vector<std::string> class_names;

        // DEBUG accessors: expose primary output tensor for raw inspection.
        // Returns nullptr / 0 if no primary output is available.
        const float* GetPrimaryOutputPtr() const;
        int GetPrimaryTupleSize() const;
        size_t GetPrimaryDetCount() const;

    private:
        //--------------------------------------------------------------------------------
        // TensorRT objects
        nvinfer1::ICudaEngine* engine  = nullptr;
        nvinfer1::IRuntime* runtime = nullptr;
        nvinfer1::IExecutionContext* context = nullptr;

        // CUDA stream
        cudaStream_t stream  = nullptr;

        // Bindings (name-based for TRT10)
        std::vector<Binding> input_bindings;
        std::vector<Binding> output_bindings;

        // Buffers
        std::vector<void*> device_ptrs;  // inputs first, then outputs
        std::vector<void*> host_ptrs;    // host pinned outputs only

        // logger
        Logger gLogger{nvinfer1::ILogger::Severity::kERROR};

        // TRT10 name-based binding metadata
        int num_iotensors = 0;
        int num_inputs = 0;
        int num_outputs = 0;
        int primary_output_binding_index = -1;
        nvinfer1::Dims primary_output_dims{};
        nvinfer1::DataType primary_output_dtype = nvinfer1::DataType::kFLOAT;
        size_t primary_output_elements = 0;

        PreParam pparam;

        // helpers
        size_t get_size_by_dims_local(const nvinfer1::Dims& d) { return static_cast<size_t>(get_size_by_dims(d)); }

};
#endif  // DETECT_NORMAL_YOLO26_HPP
