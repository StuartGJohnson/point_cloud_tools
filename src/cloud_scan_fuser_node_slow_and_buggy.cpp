#include <memory>
#include <string>

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <sensor_msgs/msg/laser_scan.hpp>
#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <message_filters/sync_policies/exact_time.h>

#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>
#include <tf2_sensor_msgs/tf2_sensor_msgs.hpp>

#include <laser_geometry/laser_geometry.hpp>

#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl_conversions/pcl_conversions.h>

class CloudScanFuserNode : public rclcpp::Node {
public:
  CloudScanFuserNode()
  : Node("cloud_scan_fuser_node"),
    tf_buffer_(this->get_clock()),
    tf_listener_(tf_buffer_) {

    cloud_topic_  = declare_parameter<std::string>("cloud_topic",  "/camera/points");
    scan_topic_   = declare_parameter<std::string>("scan_topic",   "/scan");
    output_topic_ = declare_parameter<std::string>("output_topic", "/fusion/points");
    target_frame_ = declare_parameter<std::string>("target_frame", "base_link");
    exact_time_   = declare_parameter<bool>("exact_time", false);
    queue_size_   = declare_parameter<int>("queue_size", 10);
    approx_slop_  = declare_parameter<double>("approx_slop", 0.05); // s
    voxel_leaf_   = declare_parameter<double>("voxel_leaf", 0.01);   // 0 disables

    pub_ = create_publisher<sensor_msgs::msg::PointCloud2>(output_topic_, rclcpp::SensorDataQoS());

    auto qos = rclcpp::SensorDataQoS().get_rmw_qos_profile();

    sub_cloud_ = std::make_shared<message_filters::Subscriber<sensor_msgs::msg::PointCloud2>>(
      this, cloud_topic_, qos);
    sub_scan_ = std::make_shared<message_filters::Subscriber<sensor_msgs::msg::LaserScan>>(
      this, scan_topic_, qos);

    if (exact_time_) {
      using Exact = message_filters::sync_policies::ExactTime<
        sensor_msgs::msg::PointCloud2, sensor_msgs::msg::LaserScan>;
      sync_exact_ = std::make_shared<message_filters::Synchronizer<Exact>>(Exact(queue_size_), *sub_cloud_, *sub_scan_);
      sync_exact_->registerCallback(std::bind(&CloudScanFuserNode::callback, this, std::placeholders::_1, std::placeholders::_2));
      RCLCPP_INFO(get_logger(), "ExactTime sync (queue=%d)", queue_size_);
    } else {
      using Approx = message_filters::sync_policies::ApproximateTime<
        sensor_msgs::msg::PointCloud2, sensor_msgs::msg::LaserScan>;
      sync_approx_ = std::make_shared<message_filters::Synchronizer<Approx>>(Approx(queue_size_), *sub_cloud_, *sub_scan_);
#if MESSAGE_FILTERS_HAS_SETMAXINTERVALDURATION
      sync_approx_->setMaxIntervalDuration(rclcpp::Duration::from_seconds(approx_slop_));
#endif
      sync_approx_->registerCallback(std::bind(&CloudScanFuserNode::callback, this, std::placeholders::_1, std::placeholders::_2));
      RCLCPP_INFO(get_logger(), "ApproxTime sync (queue=%d, slop=%.3fs)", queue_size_, approx_slop_);
    }

    RCLCPP_INFO(get_logger(), "Fusing %s (cloud) + %s (scan) -> %s  [target=%s, voxel=%.3f m]",
      cloud_topic_.c_str(), scan_topic_.c_str(), output_topic_.c_str(), target_frame_.c_str(), voxel_leaf_);
  }

private:
  bool transformCloud(const sensor_msgs::msg::PointCloud2 &in,
                      sensor_msgs::msg::PointCloud2 &out,
                      const rclcpp::Time &stamp) {
    try {
      auto tf = tf_buffer_.lookupTransform(target_frame_, in.header.frame_id, stamp,
                                           rclcpp::Duration::from_seconds(0.05));
      tf2::doTransform(in, out, tf);
      return true;
    } catch (const std::exception &e) {
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000,
                           "TF @stamp failed (%s). Trying latest...", e.what());
      try {
        auto tf = tf_buffer_.lookupTransform(target_frame_, in.header.frame_id, tf2::TimePointZero);
        tf2::doTransform(in, out, tf);
        return true;
      } catch (const std::exception &e2) {
        RCLCPP_ERROR(get_logger(), "TF transform failed: %s", e2.what());
        return false;
      }
    }
  }

  template<typename PointT>
  void publishConcat(const sensor_msgs::msg::PointCloud2 &c1_tf,
                     const sensor_msgs::msg::PointCloud2 &c2_tf,
                     const rclcpp::Time &stamp) {
    typename pcl::PointCloud<PointT>::Ptr p1(new pcl::PointCloud<PointT>);
    typename pcl::PointCloud<PointT>::Ptr p2(new pcl::PointCloud<PointT>);
    pcl::fromROSMsg(c1_tf, *p1);
    pcl::fromROSMsg(c2_tf, *p2);

    typename pcl::PointCloud<PointT>::Ptr fused(new pcl::PointCloud<PointT>);
    *fused = *p1 + *p2;

    if (voxel_leaf_ > 1e-6) {
      pcl::VoxelGrid<PointT> vg;
      vg.setLeafSize(voxel_leaf_, voxel_leaf_, voxel_leaf_);
      vg.setInputCloud(fused);
      typename pcl::PointCloud<PointT>::Ptr filtered(new pcl::PointCloud<PointT>);
      vg.filter(*filtered);
      fused.swap(filtered);
    }

    sensor_msgs::msg::PointCloud2 out;
    pcl::toROSMsg(*fused, out);
    out.header.frame_id = target_frame_;
    out.header.stamp = stamp;
    pub_->publish(out);
  }

  void callback(const sensor_msgs::msg::PointCloud2::ConstSharedPtr &cloud_msg,
                const sensor_msgs::msg::LaserScan::ConstSharedPtr &scan_msg) {
    // 1) Transform camera cloud to target
    sensor_msgs::msg::PointCloud2 cam_tf;
    if (!transformCloud(*cloud_msg, cam_tf, rclcpp::Time(cloud_msg->header.stamp))) return;

    // 2) Project scan -> cloud (in scan frame)
    sensor_msgs::msg::PointCloud2 scan_cloud;
    try {
      projector_.projectLaser(*scan_msg, scan_cloud, -1.0); // no range cutoff
      scan_cloud.header = scan_msg->header; // ensure frame/time
    } catch (const std::exception &e) {
      RCLCPP_ERROR(get_logger(), "laser_geometry projection failed: %s", e.what());
      return;
    }

    // 3) Transform scan cloud to target
    sensor_msgs::msg::PointCloud2 scan_tf;
    if (!transformCloud(scan_cloud, scan_tf, rclcpp::Time(scan_cloud.header.stamp))) return;

    // 4) Publish concatenation as XYZ (drop extra fields)
    rclcpp::Time stamp_pub = rclcpp::Time(cloud_msg->header.stamp) > rclcpp::Time(scan_msg->header.stamp)
                           ? rclcpp::Time(cloud_msg->header.stamp) : rclcpp::Time(scan_msg->header.stamp);
    publishConcat<pcl::PointXYZ>(cam_tf, scan_tf, stamp_pub);
  }

  // Params
  std::string cloud_topic_, scan_topic_, output_topic_, target_frame_;
  bool exact_time_; int queue_size_; double approx_slop_; double voxel_leaf_;

  // Pub
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pub_;

  // Subs + sync
  std::shared_ptr<message_filters::Subscriber<sensor_msgs::msg::PointCloud2>> sub_cloud_;
  std::shared_ptr<message_filters::Subscriber<sensor_msgs::msg::LaserScan>> sub_scan_;
  std::shared_ptr<message_filters::Synchronizer<
    message_filters::sync_policies::ApproximateTime<
      sensor_msgs::msg::PointCloud2, sensor_msgs::msg::LaserScan>>> sync_approx_;
  std::shared_ptr<message_filters::Synchronizer<
    message_filters::sync_policies::ExactTime<
      sensor_msgs::msg::PointCloud2, sensor_msgs::msg::LaserScan>>> sync_exact_;

  // TF + laser projection
  tf2_ros::Buffer tf_buffer_;
  tf2_ros::TransformListener tf_listener_;
  laser_geometry::LaserProjection projector_;
};

int main(int argc, char *argv[]) {
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<CloudScanFuserNode>());
  rclcpp::shutdown();
  return 0;
}
