#include <iostream>
#include <memory>
#include <string>

// PCL includes
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl_conversions/pcl_conversions.h>

// ROS 2 includes
#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/point_cloud2.hpp"

// Define the name of the filter node
static const std::string NODE_NAME = "voxel_grid_filter_node";

class VoxelGridFilterNode : public rclcpp::Node
{
public:
  // Constructor
  explicit VoxelGridFilterNode()
    : Node(NODE_NAME)
  {
    RCLCPP_INFO(this->get_logger(), "Starting Voxel Grid Filter Node...");

    // Declare parameters with default values
    this->declare_parameter("leaf_size", 0.1);
    this->declare_parameter("input_topic", "input_point_cloud");
    this->declare_parameter("output_topic", "downsampled_point_cloud");

    // Get parameter values
    this->get_parameter("leaf_size", leaf_size_);
    std::string input_topic;
    this->get_parameter("input_topic", input_topic);
    std::string output_topic;
    this->get_parameter("output_topic", output_topic);

    RCLCPP_INFO(this->get_logger(), "Leaf size: %f", leaf_size_);
    RCLCPP_INFO(this->get_logger(), "Subscribing to: %s", input_topic.c_str());
    RCLCPP_INFO(this->get_logger(), "Publishing to: %s", output_topic.c_str());

    // Create a subscriber to the input point cloud topic
    subscription_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
        input_topic, 10, std::bind(&VoxelGridFilterNode::cloud_callback, this, std::placeholders::_1));

    // Create a publisher for the downsampled point cloud
    publisher_ = this->create_publisher<sensor_msgs::msg::PointCloud2>(output_topic, 10);
  }

private:
  // Callback function for the point cloud subscription
  void cloud_callback(const sensor_msgs::msg::PointCloud2::SharedPtr msg) const
  {
    // Create PCL point cloud containers
    pcl::PCLPointCloud2::Ptr input_pcl(new pcl::PCLPointCloud2);
    pcl::PCLPointCloud2::Ptr downsampled_pcl(new pcl::PCLPointCloud2);

    // Convert ROS 2 PointCloud2 message to PCL format
    pcl_conversions::toPCL(*msg, *input_pcl);

    // Create a VoxelGrid filter object
    pcl::VoxelGrid<pcl::PCLPointCloud2> vg;
    vg.setInputCloud(input_pcl);
    vg.setLeafSize(leaf_size_, leaf_size_, leaf_size_);
    vg.filter(*downsampled_pcl);

    // Convert the filtered PCL point cloud back to a ROS 2 message
    sensor_msgs::msg::PointCloud2 output_msg;
    pcl_conversions::fromPCL(*downsampled_pcl, output_msg);
    output_msg.header = msg->header; // Preserve the original header

    // Publish the downsampled point cloud
    publisher_->publish(output_msg);

    //RCLCPP_INFO(this->get_logger(), "Published downsampled cloud with %u points.", downsampled_pcl->width * downsampled_pcl->height);
  }

  // Member variables
  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr subscription_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr publisher_;
  double leaf_size_;
};

int main(int argc, char *argv[])
{
  rclcpp::init(argc, argv);
  // Create and spin the node
  rclcpp::spin(std::make_shared<VoxelGridFilterNode>());
  rclcpp::shutdown();
  return 0;
}
