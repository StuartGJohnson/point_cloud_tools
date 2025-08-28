// src/cloud_scan_fuser_1to1.cpp
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

class CloudScanFuser1to1 : public rclcpp::Node {
public:
  CloudScanFuser1to1()
  : Node("cloud_scan_fuser_1to1"), tf_buffer_(this->get_clock()), tf_listener_(tf_buffer_) {
    cloud_topic_  = declare_parameter<std::string>("cloud_topic",  "/camera/points");
    scan_topic_   = declare_parameter<std::string>("scan_topic",   "/scan");
    output_topic_ = declare_parameter<std::string>("output_topic", "/fusion/points");
    target_frame_ = declare_parameter<std::string>("target_frame", "base_link");
    slop_         = rclcpp::Duration::from_seconds(declare_parameter<double>("slop", 0.02)); // 20 ms
    max_queue_    = declare_parameter<int>("max_queue", 30);
    cloud_stride_ = declare_parameter<int>("cloud_stride", 1);
    z_min_        = declare_parameter<double>("z_min", -1.0);
    z_max_        = declare_parameter<double>("z_max", 1.0);

    auto be5 = rclcpp::SensorDataQoS().keep_last(5).best_effort();
    pub_ = create_publisher<sensor_msgs::msg::PointCloud2>(output_topic_, be5);

    sub_cloud_ = create_subscription<sensor_msgs::msg::PointCloud2>(
      cloud_topic_, be5, [this](sensor_msgs::msg::PointCloud2::ConstSharedPtr msg){
        {
          std::lock_guard<std::mutex> lk(mtx_);
          cloud_buf_.push_back(std::move(msg));
          trim(cloud_buf_);
        }
        try_pair_and_process();
      });

    sub_scan_ = create_subscription<sensor_msgs::msg::LaserScan>(
      scan_topic_, be5, [this](sensor_msgs::msg::LaserScan::ConstSharedPtr msg){
        {
          std::lock_guard<std::mutex> lk(mtx_);
          scan_buf_.push_back(std::move(msg));
          trim(scan_buf_);
        }
        try_pair_and_process();
      });

    RCLCPP_INFO(get_logger(), "1:1 fuse %s + %s -> %s (slop=%.0f ms)",
      cloud_topic_.c_str(), scan_topic_.c_str(), output_topic_.c_str(), slop_.seconds()*1000.0);
  }

private:
  template<class T>
  void trim(std::deque<T> &q) { while ((int)q.size() > max_queue_) q.pop_front(); }

  static void concat_xyz_filtered(const sensor_msgs::msg::PointCloud2 &a,
                                  const sensor_msgs::msg::PointCloud2 &b,
                                  sensor_msgs::msg::PointCloud2 &out,
                                  int stride, float zmin, float zmax) {
    sensor_msgs::PointCloud2ConstIterator<float> ax(a,"x"), ay(a,"y"), az(a,"z");
    sensor_msgs::PointCloud2ConstIterator<float> bx(b,"x"), by(b,"y"), bz(b,"z");
    size_t Na=0,Nb=0,i=0;
    for (; ax!=ax.end(); ++ax,++ay,++az,++i)
      if ((stride<=1 || i%stride==0) && std::isfinite(*az) && *az>=zmin && *az<=zmax) ++Na;
    i=0;
    for (; bx!=bx.end(); ++bx,++by,++bz,++i)
      if ((stride<=1 || i%stride==0) && std::isfinite(*bz) && *bz>=zmin && *bz<=zmax) ++Nb;

    out.height=1; out.width=Na+Nb; out.is_bigendian=false; out.is_dense=false;
    sensor_msgs::PointCloud2Modifier mod(out);
    mod.setPointCloud2FieldsByString(1,"xyz"); mod.resize(out.width);

    ax = sensor_msgs::PointCloud2ConstIterator<float>(a,"x");
    ay = sensor_msgs::PointCloud2ConstIterator<float>(a,"y");
    az = sensor_msgs::PointCloud2ConstIterator<float>(a,"z");
    bx = sensor_msgs::PointCloud2ConstIterator<float>(b,"x");
    by = sensor_msgs::PointCloud2ConstIterator<float>(b,"y");
    bz = sensor_msgs::PointCloud2ConstIterator<float>(b,"z");
    sensor_msgs::PointCloud2Iterator<float> ox(out,"x"), oy(out,"y"), oz(out,"z");

    i=0;
    for (; ax!=ax.end(); ++ax,++ay,++az,++i)
      if ((stride<=1 || i%stride==0) && std::isfinite(*az) && *az>=zmin && *az<=zmax)
        { *ox++=*ax; *oy++=*ay; *oz++=*az; }
    i=0;
    for (; bx!=bx.end(); ++bx,++by,++bz,++i)
      if ((stride<=1 || i%stride==0) && std::isfinite(*bz) && *bz>=zmin && *bz<=zmax)
        { *ox++=*bx; *oy++=*by; *oz++=*bz; }
  }

  void try_pair_and_process() {
    sensor_msgs::msg::PointCloud2::ConstSharedPtr cloud;
    sensor_msgs::msg::LaserScan::ConstSharedPtr scan;

    while (true) {
      {
        std::lock_guard<std::mutex> lk(mtx_);
        if (cloud_buf_.empty() || scan_buf_.empty()) return;

        auto &c = cloud_buf_.front();
        auto &s = scan_buf_.front();
        rclcpp::Time tc(c->header.stamp), ts(s->header.stamp);
        auto dt = abs_dur(tc - ts);

        if (dt <= slop_) {
          cloud = c; scan = s;
          cloud_buf_.pop_front();
          scan_buf_.pop_front();
        } else {
          // drop the older one to maintain 1:1
          if (tc < ts) cloud_buf_.pop_front();
          else         scan_buf_.pop_front();
          continue; // try again
        }
      }
      // process outside lock
      process_pair(cloud, scan);
      // loop to see if more pairs are ready
    }
  }

  void process_pair(const sensor_msgs::msg::PointCloud2::ConstSharedPtr &cloud,
                    const sensor_msgs::msg::LaserScan::ConstSharedPtr &scan)
  {
    // project scan directly into target_frame
    sensor_msgs::msg::PointCloud2 scan_tf;
    if (!tf_buffer_.canTransform(target_frame_, scan->header.frame_id, rclcpp::Time(scan->header.stamp)))
      return;
    try { projector_.transformLaserScanToPointCloud(target_frame_, *scan, scan_tf, tf_buffer_); }
    catch (...) { return; }

    // transform camera cloud to target_frame
    if (!tf_buffer_.canTransform(target_frame_, cloud->header.frame_id, rclcpp::Time(cloud->header.stamp)))
      return;
    sensor_msgs::msg::PointCloud2 cam_tf;
    auto tf = tf_buffer_.lookupTransform(target_frame_, cloud->header.frame_id, rclcpp::Time(cloud->header.stamp));
    tf2::doTransform(*cloud, cam_tf, tf);

    // concat & publish
    sensor_msgs::msg::PointCloud2 out;
    out.header.frame_id = target_frame_;
    out.header.stamp = (rclcpp::Time(cloud->header.stamp) > rclcpp::Time(scan->header.stamp))
                     ? cloud->header.stamp : scan->header.stamp;
    concat_xyz_filtered(cam_tf, scan_tf, out, cloud_stride_, (float)z_min_, (float)z_max_);
    pub_->publish(out);
  }

  // params
  std::string cloud_topic_, scan_topic_, output_topic_, target_frame_;
  rclcpp::Duration slop_;
  int max_queue_, cloud_stride_; double z_min_, z_max_;

  // comms
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pub_;
  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr sub_cloud_;
  rclcpp::Subscription<sensor_msgs::msg::LaserScan>::SharedPtr sub_scan_;

  // buffers
  std::mutex mtx_;
  std::deque<sensor_msgs::msg::PointCloud2::ConstSharedPtr> cloud_buf_;
  std::deque<sensor_msgs::msg::LaserScan::ConstSharedPtr>   scan_buf_;

  // TF + projection
  tf2_ros::Buffer tf_buffer_;
  tf2_ros::TransformListener tf_listener_;
  laser_geometry::LaserProjection projector_;
};

int main(int argc,char**argv){
  rclcpp::init(argc,argv);
  rclcpp::spin(std::make_shared<CloudScanFuser1to1>());
  rclcpp::shutdown();
  return 0;
}
