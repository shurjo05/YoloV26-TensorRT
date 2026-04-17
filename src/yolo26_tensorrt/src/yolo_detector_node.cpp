//
// ROS2 node for YOLO26 TensorRT inference.
//
// Subscribes to an image topic, runs detection, and publishes:
//   - Structured detections (vision_msgs/Detection2DArray)
//   - Annotated debug image (sensor_msgs/Image)
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

#include "yolo26.hpp"
#include "ros_detection_bridge.hpp"

//----------------------------------------------------------------------------------------
/**
 * @brief ROS2 node that wraps the YOLO26 TensorRT inference engine.
 *
 * Thin orchestration layer: parameters, subscriptions, publishing, and frame dropping.
 * All type conversion is delegated to RosDetectionBridge.
 * All inference is delegated to YOLO26.
 */
class YoloDetectorNode : public rclcpp::Node
{
public:
    //------------------------------------------------------------------------------------
    /**
     * @brief Construct the detector node.
     *
     * Reads ROS2 parameters, initialises the YOLO26 TensorRT engine, creates
     * publishers and a subscriber with SensorDataQoS for frame dropping.
     *
     * Expected parameters (set via params.yaml or launch overrides):
     *   - engine_file_path   (string)   Path to the TensorRT .engine file.
     *   - confidence_threshold (double) Minimum detection confidence (default 0.25).
     *   - iou_threshold       (double)  IoU threshold (default 0.65, unused by NMS-free model).
     *   - class_names         (string[]) Label list indexed by class ID (default ["aphid"]).
     *   - topk                (int)     Max detections to keep (default 100).
     */
    YoloDetectorNode()
        : Node("yolo_detector_node"), is_processing_(false)
    {
        // --- Declare parameters ---
        this->declare_parameter<std::string>("engine_file_path", "");
        this->declare_parameter<double>("confidence_threshold", 0.25);
        this->declare_parameter<double>("iou_threshold", 0.65);
        this->declare_parameter<std::vector<std::string>>("class_names", {"aphid"});
        this->declare_parameter<int>("topk", 100);
        this->declare_parameter<std::vector<std::string>>("filter_classes", std::vector<std::string>{});

        // --- Read parameters ---
        std::string engine_path = this->get_parameter("engine_file_path").as_string();
        conf_threshold_ = this->get_parameter("confidence_threshold").as_double();
        iou_threshold_  = this->get_parameter("iou_threshold").as_double();
        auto names      = this->get_parameter("class_names").as_string_array();
        topk_           = this->get_parameter("topk").as_int();
        auto filter     = this->get_parameter("filter_classes").as_string_array();

        // Build set of allowed class IDs from filter_classes names (empty = allow all)
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

        // --- Initialize inference engine ---
        cudaSetDevice(0);

        RCLCPP_INFO(this->get_logger(), "Loading TensorRT engine: %s", engine_path.c_str());
        yolo26_ = std::make_unique<YOLO26>(engine_path);
        yolo26_->MakePipe(true);
        yolo26_->SetClassNames(names);

        int num_classes = yolo26_->GetNumClasses();
        if (num_classes <= 0) {
            RCLCPP_FATAL(this->get_logger(), "Could not infer class count from engine output");
            rclcpp::shutdown();
            return;
        }
        RCLCPP_INFO(this->get_logger(), "Engine loaded. %d class(es), %d label(s) configured.",
                     num_classes, static_cast<int>(names.size()));

        // --- Publishers ---
        det_pub_ = this->create_publisher<vision_msgs::msg::Detection2DArray>(
            "~/detections", 10);
        // image_transport publisher: automatically provides both raw and compressed
        // sub-topics (e.g. ~/detections/image/compressed) so Foxglove can subscribe
        // to the compressed transport without the detector node needing to encode manually.
        img_pub_ = image_transport::create_publisher(this, "~/detections/image");

        // --- Subscriber ---
        // Compressed transport — arducam_publisher.py publishes JPEG to keep fps viable
        // from Python (Issue 25). image_transport decompresses before the callback runs.
        // SensorDataQoS = best-effort, depth 1 — drops frames if inference can't keep up.
        image_sub_ = image_transport::create_subscription(
            this,
            "/image_raw",
            std::bind(&YoloDetectorNode::on_image, this, std::placeholders::_1),
            "compressed",
            rclcpp::SensorDataQoS().get_rmw_qos_profile());

        RCLCPP_INFO(this->get_logger(), "YOLO26 detector node ready. Listening on /image_raw (compressed transport)");
    }

private:
    //------------------------------------------------------------------------------------
    /**
     * @brief Image callback: run inference and publish results.
     *
     * Drops the frame immediately if the previous inference is still running
     * (guarded by atomic flag). On success, publishes both a Detection2DArray
     * and an annotated debug image.
     *
     * @param msg Incoming ROS image message from the subscribed topic.
     */
    void on_image(const sensor_msgs::msg::Image::ConstSharedPtr & msg)
    {
        // Frame dropping: skip if already processing
        if (is_processing_.exchange(true)) {
            return;
        }

        try {
            // ROS Image → cv::Mat (via bridge)
            cv::Mat frame = RosDetectionBridge::imageMsgToMat(msg);

            // Rotate to correct for upside-down camera mounting
            cv::rotate(frame, frame, cv::ROTATE_180);

            // Inference
            yolo26_->CopyFromMat(frame);
            yolo26_->Infer();

            std::vector<det::Object> objs;
            yolo26_->PostProcess(objs, conf_threshold_, iou_threshold_, topk_);

            // TEMP DEBUG: log raw output tensor values + count above threshold.
            // Throttled to once per ~30 frames so logs stay readable.
            {
                static int dbg_count = 0;
                if ((dbg_count++ % 30) == 0) {
                    const float* out = yolo26_->GetPrimaryOutputPtr();
                    const int tup = yolo26_->GetPrimaryTupleSize();
                    const size_t det_count = yolo26_->GetPrimaryDetCount();
                    if (out && tup >= 6) {
                        // Find max confidence across all detection rows to see what the model is producing.
                        float max_conf = 0.0f;
                        size_t max_idx = 0;
                        for (size_t i = 0; i < det_count; ++i) {
                            float c = out[i * tup + 4];
                            if (c > max_conf) { max_conf = c; max_idx = i; }
                        }
                        RCLCPP_INFO(this->get_logger(),
                            "RAW OUTPUT: %zu rows x %d. Max conf=%.4f at row %zu (cls=%.0f, x1=%.1f y1=%.1f x2=%.1f y2=%.1f)",
                            det_count, tup, max_conf, max_idx,
                            out[max_idx*tup+5], out[max_idx*tup+0], out[max_idx*tup+1],
                            out[max_idx*tup+2], out[max_idx*tup+3]);
                        // Dump first 3 rows verbatim
                        for (int d = 0; d < 3 && d < (int)det_count; ++d) {
                            RCLCPP_INFO(this->get_logger(),
                                "  det[%d]: x1=%.2f y1=%.2f x2=%.2f y2=%.2f conf=%.4f cls=%.0f",
                                d, out[d*tup+0], out[d*tup+1], out[d*tup+2], out[d*tup+3],
                                out[d*tup+4], out[d*tup+5]);
                        }
                        RCLCPP_INFO(this->get_logger(),
                            "  PostProcess kept %zu objects above threshold %.3f",
                            objs.size(), conf_threshold_);
                    } else {
                        RCLCPP_WARN(this->get_logger(),
                            "RAW OUTPUT unavailable: ptr=%p tuple_size=%d", (const void*)out, tup);
                    }
                }
            }

            // Filter to allowed classes if filter_classes is set
            if (!filter_class_ids_.empty()) {
                objs.erase(std::remove_if(objs.begin(), objs.end(), [this](const det::Object& o) {
                    return filter_class_ids_.find(o.label) == filter_class_ids_.end();
                }), objs.end());
            }

            // Objects → Detection2DArray (via bridge) → publish
            auto det_array = RosDetectionBridge::objectsToDetections(
                objs, yolo26_->class_names, msg->header);
            det_pub_->publish(det_array);

            // Annotated image → publish
            yolo26_->DrawObjects(frame, objs);
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
    // Inference engine
    std::unique_ptr<YOLO26> yolo26_;

    // ROS2 pub/sub
    image_transport::Subscriber image_sub_;
    rclcpp::Publisher<vision_msgs::msg::Detection2DArray>::SharedPtr det_pub_;
    image_transport::Publisher img_pub_;

    // Parameters (cached from ROS2 param server)
    float conf_threshold_;
    float iou_threshold_;
    int   topk_;
    std::set<int> filter_class_ids_;  // empty = allow all classes

    // Frame dropping flag (atomic for thread safety)
    std::atomic<bool> is_processing_;
};

//----------------------------------------------------------------------------------------
/**
 * @brief Program entry point: initialise ROS2 and spin the YOLO26 detector node.
 *
 * @param argc Argument count (forwarded to rclcpp::init).
 * @param argv Argument values (forwarded to rclcpp::init).
 * @return 0 on normal shutdown.
 */
int main(int argc, char** argv)
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<YoloDetectorNode>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}
