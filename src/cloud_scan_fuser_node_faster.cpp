// src/cloud_scan_fuser_fast.cpp
#include <deque>
#include <mutex>
#include <algorithm>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <sensor_msgs/msg/laser_scan.hpp>
#include <sensor_msgs/point_cloud2_iterator.hpp>

#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>
#include <tf2_sensor_msgs/tf2_sensor_msgs.hpp>
#include <laser_geometry/laser_geometry.hpp>

rclcpp::Duration abs_dur(const rclcpp::Duration& duration) {
    if (duration < rclcpp::Duration(0, 0)) { // Check if the duration is negative
        return duration * -1; // Negate the duration
    }
    return duration; // Return the duration as is if not negative
}

class CloudScanFuserFast : public rclcpp::Node {
public:
  CloudScanFuserFast()
  : Node("cloud_scan_fuser_fast"),
    tf_buffer_(this->get_clock()),
    tf_listener_(tf_buffer_) {

    cloud_topic_  = declare_parameter<std::string>("cloud_topic",  "/camera/points");
    scan_topic_   = declare_parameter<std::string>("scan_topic",   "/scan");
    output_topic_ = declare_parameter<std::string>("output_topic", "/fusion/points");
    target_frame_ = declare_parameter<std::string>("target_frame", "base_link");
    voxel_leaf_   = declare_parameter<double>("voxel_leaf", 0.0);  // (unused here)
    slop_sec_     = declare_parameter<double>("slop", 0.05);       // pairing slop

    auto be5 = rclcpp::SensorDataQoS().keep_last(5).best_effort();
    pub_ = create_publisher<sensor_msgs::msg::PointCloud2>(output_topic_, be5);

    sub_scan_ = create_subscription<sensor_msgs::msg::LaserScan>(
      scan_topic_, be5,
      [this](sensor_msgs::msg::LaserScan::ConstSharedPtr msg){
        std::lock_guard<std::mutex> lk(mtx_);
        scans_.push_back(std::move(msg));
        while (scans_.size() > 20) scans_.pop_front();
      });

    sub_cloud_ = create_subscription<sensor_msgs::msg::PointCloud2>(
      cloud_topic_, be5,
      [this](sensor_msgs::msg::PointCloud2::ConstSharedPtr cloud){
        sensor_msgs::msg::LaserScan::ConstSharedPtr scan;
        {
          std::lock_guard<std::mutex> lk(mtx_);
          if (scans_.empty()) return;
          const auto t = rclcpp::Time(cloud->header.stamp);
          auto it = std::min_element(scans_.begin(), scans_.end(),
            [&](auto &a, auto &b){
              return abs_dur(t - rclcpp::Time(a->header.stamp))
                   < abs_dur(t - rclcpp::Time(b->header.stamp));
            });
          if (it == scans_.end()) return;
          if (abs_dur(t - rclcpp::Time((*it)->header.stamp)) >
               rclcpp::Duration::from_seconds(slop_sec_)) return;
          scan = *it; // shared_ptr copy keeps it alive
        }
        process_pair(std::move(cloud), std::move(scan));
      });

    RCLCPP_INFO(get_logger(), "Fusing %s + %s -> %s  (target=%s, slop=%.0f ms)",
                cloud_topic_.c_str(), scan_topic_.c_str(),
                output_topic_.c_str(), target_frame_.c_str(), slop_sec_*1000.0);
  }

private:
  // Concatenate two PointCloud2s (XYZ only) without PCL.
  static void concat_xyz_clouds(const sensor_msgs::msg::PointCloud2 &a,
                                const sensor_msgs::msg::PointCloud2 &b,
                                sensor_msgs::msg::PointCloud2 &out) {
    const auto Na = static_cast<size_t>(a.width) * a.height;
    const auto Nb = static_cast<size_t>(b.width) * b.height;
    out.header = a.header;           // caller sets frame/time
    out.height = 1;                  // unorganized
    out.width  = Na + Nb;
    out.is_bigendian = false;
    out.is_dense = false;

    sensor_msgs::PointCloud2Modifier mod(out);
    mod.setPointCloud2FieldsByString(1, "xyz");
    mod.resize(out.width);

    sensor_msgs::PointCloud2ConstIterator<float> ax(a, "x"), ay(a, "y"), az(a, "z");
    sensor_msgs::PointCloud2ConstIterator<float> bx(b, "x"), by(b, "y"), bz(b, "z");
    sensor_msgs::PointCloud2Iterator<float> ox(out, "x"), oy(out, "y"), oz(out, "z");

    for (size_t i=0; i<Na; ++i, ++ax, ++ay, ++az, ++ox, ++oy, ++oz) {
      *ox = *ax; *oy = *ay; *oz = *az;
    }
    for (size_t i=0; i<Nb; ++i, ++bx, ++by, ++bz, ++ox, ++oy, ++oz) {
      *ox = *bx; *oy = *by; *oz = *bz;
    }
  }

  void process_pair(sensor_msgs::msg::PointCloud2::ConstSharedPtr cloud,
                    sensor_msgs::msg::LaserScan::ConstSharedPtr scan)
  {
    // 1) project scan directly into target_frame (non-blocking TF)
    sensor_msgs::msg::PointCloud2 scan_tf;
    if (!tf_buffer_.canTransform(
          target_frame_, scan->header.frame_id, rclcpp::Time(scan->header.stamp))) {
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000,
                           "No TF for scan @ stamp, drop");
      return;
    }
    try {
      projector_.transformLaserScanToPointCloud(target_frame_, *scan, scan_tf, tf_buffer_);
    } catch (const std::exception &e) {
      RCLCPP_ERROR(get_logger(), "laser_geometry failed: %s", e.what());
      return;
    }

    // 2) transform camera cloud to target_frame (non-blocking)
    sensor_msgs::msg::PointCloud2 cam_tf;
    if (!tf_buffer_.canTransform(
          target_frame_, cloud->header.frame_id, rclcpp::Time(cloud->header.stamp))) {
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000,
                           "No TF for cloud @ stamp, drop");
      return;
    }
    auto tf = tf_buffer_.lookupTransform(
        target_frame_, cloud->header.frame_id, rclcpp::Time(cloud->header.stamp));
    tf2::doTransform(*cloud, cam_tf, tf);

    // 3) concat (xyz only) and publish
    sensor_msgs::msg::PointCloud2 out;
    out.header.frame_id = target_frame_;
    out.header.stamp = (rclcpp::Time(cloud->header.stamp) > rclcpp::Time(scan->header.stamp))
                     ? cloud->header.stamp : scan->header.stamp;
    concat_xyz_clouds(cam_tf, scan_tf, out);
    pub_->publish(out);
  }

  // params
  std::string cloud_topic_, scan_topic_, output_topic_, target_frame_;
  double voxel_leaf_, slop_sec_;

  // comms
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pub_;
  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr sub_cloud_;
  rclcpp::Subscription<sensor_msgs::msg::LaserScan>::SharedPtr sub_scan_;

  // buffers
  std::mutex mtx_;
  std::deque<sensor_msgs::msg::LaserScan::ConstSharedPtr> scans_;

  // TF + projection
  tf2_ros::Buffer tf_buffer_;
  tf2_ros::TransformListener tf_listener_;
  laser_geometry::LaserProjection projector_;
};

int main(int argc, char *argv[]) {
  rclcpp::init(argc, argv);
  // Start single-threaded (robust). If you need more, swap to MultiThreaded later.
  rclcpp::spin(std::make_shared<CloudScanFuserFast>());
  rclcpp::shutdown();
  return 0;
}
