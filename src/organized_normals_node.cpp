// src/organized_normals_node.cpp


#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/features/integral_image_normal.h>
#include <pcl_conversions/pcl_conversions.h>

// ros2 run <your_pkg> organized_normals_node \
//   --ros-args -p input:=/camera/points -p output:=/camera/points_normals \
//              -p method:=average_3d_gradient -p smoothing_size:=10.0 -p depth_dependent:=true

class OrganizedNormalsNode : public rclcpp::Node {
public:
  OrganizedNormalsNode()
  : Node("organized_normals_node") {
    in_topic_  = declare_parameter<std::string>("input",  "/camera/points");
    out_topic_ = declare_parameter<std::string>("output", "/camera/points_normals");
    method_    = declare_parameter<std::string>("method", "average_3d_gradient"); // or "average_depth_change"
    smooth_    = declare_parameter<double>("smoothing_size", 10.0);
    depth_dep_ = declare_parameter<bool>("depth_dependent", true);

    sub_ = create_subscription<sensor_msgs::msg::PointCloud2>(
      in_topic_, rclcpp::SensorDataQoS(),
      std::bind(&OrganizedNormalsNode::cb, this, std::placeholders::_1));
    pub_ = create_publisher<sensor_msgs::msg::PointCloud2>(out_topic_, rclcpp::SensorDataQoS());
  }

private:
  void cb(const sensor_msgs::msg::PointCloud2::ConstSharedPtr &msg) {
    if (msg->height <= 1) {
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000, "Input not organized, skipping.");
      return;
    }
    pcl::PointCloud<pcl::PointXYZ>::Ptr in(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::fromROSMsg(*msg, *in);

    pcl::IntegralImageNormalEstimation<pcl::PointXYZ, pcl::Normal> ne;
    if (method_ == "average_depth_change")
      ne.setNormalEstimationMethod(ne.AVERAGE_DEPTH_CHANGE);
    else
      ne.setNormalEstimationMethod(ne.AVERAGE_3D_GRADIENT);
    ne.setMaxDepthChangeFactor(0.02f);
    ne.setNormalSmoothingSize(static_cast<float>(smooth_));
    ne.setDepthDependentSmoothing(depth_dep_);
    ne.setInputCloud(in);

    pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal>);
    ne.compute(*normals);

    // concatenate XYZ + Normal
    pcl::PointCloud<pcl::PointNormal> out;
    pcl::concatenateFields(*in, *normals, out);

    sensor_msgs::msg::PointCloud2 out_msg;
    pcl::toROSMsg(out, out_msg);
    out_msg.header = msg->header;
    pub_->publish(out_msg);
  }

  // params & comms
  std::string in_topic_, out_topic_, method_;
  double smooth_; bool depth_dep_;
  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr sub_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pub_;
};

int main(int argc, char** argv){
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<OrganizedNormalsNode>());
  rclcpp::shutdown();
  return 0;
}
