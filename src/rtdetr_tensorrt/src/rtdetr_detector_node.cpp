//
// ROS2 node for RT-DETR TensorRT inference.
//
// Subscribes to an image topic, runs detection, and publishes:
//   - Structured detections (vision_msgs/Detection2DArray)
//   - Annotated debug image (sensor_msgs/Image + image_transport compressed variant)
//
// Mirrors the interface of yolo_detector_node so the two are drop-in interchangeable.
//

#include <memory>
#include <atomic>
#include <set>
#include <string>
#include <vector>

#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/image.hpp"
#include "vision_msgs/msg/detection2_d_array.hpp"
#include "image_transport/image_transport.hpp"

#include "rtdetr.hpp"
#include "ros_detection_bridge.hpp"

//----------------------------------------------------------------------------------------
/**
 * @brief ROS2 node wrapping the RT-DETR TensorRT inference engine.
 */
class RTDETRDetectorNode : public rclcpp::Node
{
public:
    RTDETRDetectorNode()
        : Node("rtdetr_detector_node"), is_processing_(false)
    {
        // --- Declare parameters ---
        this->declare_parameter<std::string>("engine_file_path", "");
        this->declare_parameter<double>("confidence_threshold", 0.25);
        this->declare_parameter<std::vector<std::string>>("class_names", {"aphid"});
        this->declare_parameter<int>("topk", 100);
        this->declare_parameter<std::vector<std::string>>("filter_classes", std::vector<std::string>{});
        this->declare_parameter<std::vector<double>>("image_mean",   std::vector<double>{0.2377, 0.3481, 0.3058});
        this->declare_parameter<std::vector<double>>("image_stddev", std::vector<double>{0.2299, 0.2375, 0.2250});
        this->declare_parameter<bool>("rotate_180", true);

        // --- Read parameters ---
        std::string engine_path = this->get_parameter("engine_file_path").as_string();
        conf_threshold_ = this->get_parameter("confidence_threshold").as_double();
        auto names      = this->get_parameter("class_names").as_string_array();
        topk_           = this->get_parameter("topk").as_int();
        auto filter     = this->get_parameter("filter_classes").as_string_array();
        auto mean_v     = this->get_parameter("image_mean").as_double_array();
        auto std_v      = this->get_parameter("image_stddev").as_double_array();
        rotate_180_     = this->get_parameter("rotate_180").as_bool();

        // Filter class IDs (empty = allow all)
        if (!filter.empty()) {
            for (size_t i = 0; i < names.size(); ++i) {
                for (const auto& f : filter) {
                    if (names[i] == f) {
                        filter_class_ids_.insert(static_cast<int>(i));
                    }
                }
            }
        }

        if (engine_path.empty()) {
            RCLCPP_FATAL(this->get_logger(), "engine_file_path parameter is required");
            rclcpp::shutdown();
            return;
        }

        // --- Init engine ---
        cudaSetDevice(0);

        RCLCPP_INFO(this->get_logger(), "Loading RT-DETR TensorRT engine: %s", engine_path.c_str());
        rtdetr_ = std::make_unique<RTDETR>(engine_path);
        rtdetr_->MakePipe(true);
        rtdetr_->SetClassNames(names);

        // Override normalisation from params if they look well-formed.
        if (mean_v.size() == 3 && std_v.size() == 3) {
            rtdetr_->SetMean({static_cast<float>(mean_v[0]),
                              static_cast<float>(mean_v[1]),
                              static_cast<float>(mean_v[2])});
            rtdetr_->SetStddev({static_cast<float>(std_v[0]),
                                static_cast<float>(std_v[1]),
                                static_cast<float>(std_v[2])});
            RCLCPP_INFO(this->get_logger(),
                        "Normalization: mean=[%.4f %.4f %.4f] std=[%.4f %.4f %.4f]",
                        mean_v[0], mean_v[1], mean_v[2], std_v[0], std_v[1], std_v[2]);
        }

        RCLCPP_INFO(this->get_logger(), "Engine loaded. %zu queries, %zu label(s) configured.",
                    rtdetr_->GetNumQueries(), names.size());

        // --- Publishers ---
        det_pub_ = this->create_publisher<vision_msgs::msg::Detection2DArray>(
            "~/detections", 10);
        img_pub_ = image_transport::create_publisher(this, "~/detections/image");

        // --- Subscriber ---
        // Mirrors yolo_detector_node: compressed image_transport at /image_raw,
        // SensorDataQoS for best-effort frame dropping.
        image_sub_ = image_transport::create_subscription(
            this,
            "/image_raw",
            std::bind(&RTDETRDetectorNode::on_image, this, std::placeholders::_1),
            "compressed",
            rclcpp::SensorDataQoS().get_rmw_qos_profile());

        RCLCPP_INFO(this->get_logger(), "RT-DETR detector node ready. Listening on /image_raw (compressed transport)");
    }

private:
    //------------------------------------------------------------------------------------
    void on_image(const sensor_msgs::msg::Image::ConstSharedPtr & msg)
    {
        // Drop frame if inference is still running
        if (is_processing_.exchange(true)) {
            return;
        }

        try {
            cv::Mat frame = RosDetectionBridge::imageMsgToMat(msg);

            if (rotate_180_) {
                cv::rotate(frame, frame, cv::ROTATE_180);
            }

            rtdetr_->CopyFromMat(frame);
            rtdetr_->Infer();

            std::vector<det::Object> objs;
            rtdetr_->PostProcess(objs, conf_threshold_, topk_);

            // Throttled debug log: max score + first few queries
            {
                static int dbg_count = 0;
                if ((dbg_count++ % 30) == 0) {
                    const float* scores = rtdetr_->GetScoresPtr();
                    const float* bxs    = rtdetr_->GetBoxesPtr();
                    const size_t Q      = rtdetr_->GetNumQueries();
                    if (scores && bxs && Q > 0) {
                        float  max_s   = 0.0f;
                        size_t max_idx = 0;
                        for (size_t q = 0; q < Q; ++q) {
                            if (scores[q] > max_s) { max_s = scores[q]; max_idx = q; }
                        }
                        const float* b = bxs + max_idx * 4;
                        RCLCPP_INFO(this->get_logger(),
                            "RT-DETR: %zu queries, max_score=%.4f at q=%zu (box=%.1f,%.1f,%.1f,%.1f). Kept %zu > %.3f",
                            Q, max_s, max_idx, b[0], b[1], b[2], b[3], objs.size(), conf_threshold_);
                    }
                }
            }

            if (!filter_class_ids_.empty()) {
                objs.erase(std::remove_if(objs.begin(), objs.end(), [this](const det::Object& o) {
                    return filter_class_ids_.find(o.label) == filter_class_ids_.end();
                }), objs.end());
            }

            auto det_array = RosDetectionBridge::objectsToDetections(
                objs, rtdetr_->class_names, msg->header);
            det_pub_->publish(det_array);

            rtdetr_->DrawObjects(frame, objs);
            auto img_msg = RosDetectionBridge::matToImageMsg(frame, msg->header);
            img_pub_.publish(*img_msg);

        } catch (const cv_bridge::Exception& e) {
            RCLCPP_ERROR(this->get_logger(), "cv_bridge exception: %s", e.what());
        } catch (const std::exception& e) {
            RCLCPP_ERROR(this->get_logger(), "Inference error: %s", e.what());
        }

        is_processing_.store(false);
    }

    //------------------------------------------------------------------------------------
    std::unique_ptr<RTDETR> rtdetr_;

    image_transport::Subscriber image_sub_;
    rclcpp::Publisher<vision_msgs::msg::Detection2DArray>::SharedPtr det_pub_;
    image_transport::Publisher img_pub_;

    float conf_threshold_;
    int   topk_;
    bool  rotate_180_ = true;
    std::set<int> filter_class_ids_;

    std::atomic<bool> is_processing_;
};

//----------------------------------------------------------------------------------------
int main(int argc, char** argv)
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<RTDETRDetectorNode>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}
