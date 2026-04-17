//
// Shared TensorRT/CV structs and helpers for the RT-DETR inference pipeline.
//
// Mirrors YoloV26-TensorRT/src/yolo26_tensorrt/include/common.hpp, with the addition
// of INT64 support in type_to_size() — RT-DETR outputs a labels tensor as int64.
//

#ifndef RTDETR_COMMON_HPP
#define RTDETR_COMMON_HPP
#include "NvInfer.h"
#include "opencv2/opencv.hpp"

#define CHECK(call)                                                                                                    \
    do {                                                                                                               \
        const cudaError_t error_code = call;                                                                           \
        if (error_code != cudaSuccess) {                                                                               \
            printf("CUDA Error:\n");                                                                                   \
            printf("    File:       %s\n", __FILE__);                                                                  \
            printf("    Line:       %d\n", __LINE__);                                                                  \
            printf("    Error code: %d\n", error_code);                                                                \
            printf("    Error text: %s\n", cudaGetErrorString(error_code));                                            \
            exit(1);                                                                                                   \
        }                                                                                                              \
    } while (0)

class Logger: public nvinfer1::ILogger {
public:
    nvinfer1::ILogger::Severity reportableSeverity;

    explicit Logger(nvinfer1::ILogger::Severity severity = nvinfer1::ILogger::Severity::kINFO):
        reportableSeverity(severity)
    {
    }

    void log(nvinfer1::ILogger::Severity severity, const char* msg) noexcept override
    {
        if (severity > reportableSeverity) {
            return;
        }
        switch (severity) {
            case nvinfer1::ILogger::Severity::kINTERNAL_ERROR:
                std::cerr << "INTERNAL_ERROR: ";
                break;
            case nvinfer1::ILogger::Severity::kERROR:
                std::cerr << "ERROR: ";
                break;
            case nvinfer1::ILogger::Severity::kWARNING:
                std::cerr << "WARNING: ";
                break;
            case nvinfer1::ILogger::Severity::kINFO:
                std::cerr << "INFO: ";
                break;
            default:
                std::cerr << "VERBOSE: ";
                break;
        }
        std::cerr << msg << std::endl;
    }
};

inline int get_size_by_dims(const nvinfer1::Dims& dims)
{
    int size = 1;
    for (int i = 0; i < dims.nbDims; i++) {
        size *= dims.d[i];
    }
    return size;
}

inline int type_to_size(const nvinfer1::DataType& dataType)
{
    switch (dataType) {
        case nvinfer1::DataType::kFLOAT:
            return 4;
        case nvinfer1::DataType::kHALF:
            return 2;
        case nvinfer1::DataType::kINT32:
            return 4;
        case nvinfer1::DataType::kINT64:  // RT-DETR labels tensor is int64
            return 8;
        case nvinfer1::DataType::kINT8:
            return 1;
        case nvinfer1::DataType::kBOOL:
            return 1;
        default:
            return 4;
    }
}

inline static float clamp(float val, float min, float max)
{
    return val > min ? (val < max ? val : max) : min;
}

namespace det {
struct Binding {
    size_t             size  = 1;
    size_t             dsize = 1;
    nvinfer1::DataType dtype = nvinfer1::DataType::kFLOAT;
    nvinfer1::Dims     dims;
    std::string        name;
};

struct Object {
    cv::Rect_<float> rect;
    int              label = 0;
    float            prob  = 0.0;
};

// Preprocessing metadata used to map detections back to the original image
// coordinate space. For RT-DETR we do plain resize (no letterbox), so dw/dh are
// unused but kept for parity with the yolo26 pipeline.
struct PreParam {
    float ratio_x = 1.0f;  // orig_width  / model_input_width
    float ratio_y = 1.0f;  // orig_height / model_input_height
    float dw      = 0.0f;
    float dh      = 0.0f;
    float height  = 0;
    float width   = 0;
};
}  // namespace det
#endif  // RTDETR_COMMON_HPP
