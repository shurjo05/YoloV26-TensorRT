//
// ROS2 node for YOLO26 TensorRT inference.
//
// Subscribes to an image topic, runs detection, and publishes:
//   - Structured detections (vision_msgs/Detection2DArray)
//   - Annotated debug image (sensor_msgs/Image)
//

#include <memory>
#include <atomic>
#include <string>
#include <vector>

#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/image.hpp"
#include "vision_msgs/msg/detection2_d_array.hpp"

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
     *   - input_width         (int)     Network input width  (default 640).
     *   - input_height        (int)     Network input height (default 640).
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
        this->declare_parameter<int>("input_width", 640);
        this->declare_parameter<int>("input_height", 640);
        this->declare_parameter<std::vector<std::string>>("class_names", {"aphid"});
        this->declare_parameter<int>("topk", 100);

        // --- Read parameters ---
        std::string engine_path = this->get_parameter("engine_file_path").as_string();
        conf_threshold_ = this->get_parameter("confidence_threshold").as_double();
        iou_threshold_  = this->get_parameter("iou_threshold").as_double();
        int input_w     = this->get_parameter("input_width").as_int();
        int input_h     = this->get_parameter("input_height").as_int();
        auto names      = this->get_parameter("class_names").as_string_array();
        topk_           = this->get_parameter("topk").as_int();

        input_size_ = cv::Size(input_w, input_h);

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
        img_pub_ = this->create_publisher<sensor_msgs::msg::Image>(
            "~/detections/image", 10);

        // --- Subscriber ---
        // SensorDataQoS = best-effort, depth 1 — drops frames if inference can't keep up
        auto qos = rclcpp::SensorDataQoS();
        image_sub_ = this->create_subscription<sensor_msgs::msg::Image>(
            "/image_raw", qos,
            std::bind(&YoloDetectorNode::on_image, this, std::placeholders::_1));

        RCLCPP_INFO(this->get_logger(), "YOLO26 detector node ready. Listening on /image_raw");
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
    void on_image(const sensor_msgs::msg::Image::SharedPtr msg)
    {
        // Frame dropping: skip if already processing
        if (is_processing_.exchange(true)) {
            return;
        }

        try {
            // ROS Image → cv::Mat (via bridge)
            cv::Mat frame = RosDetectionBridge::imageMsgToMat(msg);

            // Inference
            yolo26_->CopyFromMat(frame, input_size_);
            yolo26_->Infer();

            std::vector<det::Object> objs;
            yolo26_->PostProcess(objs, conf_threshold_, iou_threshold_, topk_);

            // Objects → Detection2DArray (via bridge) → publish
            auto det_array = RosDetectionBridge::objectsToDetections(
                objs, yolo26_->class_names, msg->header);
            det_pub_->publish(det_array);

            // Annotated image → publish
            yolo26_->DrawObjects(frame, objs);
            auto img_msg = RosDetectionBridge::matToImageMsg(frame, msg->header);
            img_pub_->publish(*img_msg);

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
    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr image_sub_;
    rclcpp::Publisher<vision_msgs::msg::Detection2DArray>::SharedPtr det_pub_;
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr img_pub_;

    // Parameters (cached from ROS2 param server)
    float conf_threshold_;
    float iou_threshold_;
    int   topk_;
    cv::Size input_size_;

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
