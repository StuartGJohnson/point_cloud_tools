// src/cloud_scan_fuser_node.cpp
#include <deque>
#include <mutex>
#include <string>
#include <algorithm>
#include <cmath>

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <sensor_msgs/msg/laser_scan.hpp>
#include <sensor_msgs/point_cloud2_iterator.hpp>

#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>
#include <tf2_sensor_msgs/tf2_sensor_msgs.hpp>
#include <laser_geometry/laser_geometry.hpp>

#include <limits>
inline bool has_field(const sensor_msgs::msg::PointCloud2 &pc, const std::string &name){
  for (auto &f : pc.fields) if (f.name == name) return true;
  return false;
}

class CloudScanFuser1to1 : public rclcpp::Node {
public:
  explicit CloudScanFuser1to1(const rclcpp::NodeOptions &opts = rclcpp::NodeOptions())
  : rclcpp::Node("cloud_scan_fuser_1to1", opts),
    tf_buffer_(this->get_clock()),
    tf_listener_(tf_buffer_)
  {
    // ---- Parameters
    cloud_topic_   = declare_parameter<std::string>("cloud_topic",   "/camera/points");
    scan_topic_    = declare_parameter<std::string>("scan_topic",    "/scan");
    output_topic_  = declare_parameter<std::string>("output_topic",  "/fusion/points");
    target_frame_  = declare_parameter<std::string>("target_frame",  "base_link");
    slop_s_        = declare_parameter<double>("slop",               0.02); // 20 ms pairing window
    max_queue_     = declare_parameter<int>("max_queue",             30);

    // Decimate only the camera cloud (2-D pixel grid); keep scan by default
    cloud_stride_u_= declare_parameter<int>("cloud_stride_u",        2);
    cloud_stride_v_= declare_parameter<int>("cloud_stride_v",        2);
    scan_stride_   = declare_parameter<int>("scan_stride",           1);

    // Optional ROI on camera cloud rows (fractions of height to drop from top/bottom)
    roi_top_frac_    = declare_parameter<double>("cloud_roi_top",    0.0);
    roi_bottom_frac_ = declare_parameter<double>("cloud_roi_bottom", 0.0);

    // Simple Z-range gate in target_frame (meters)
    z_min_ = declare_parameter<double>("z_min", -1.0);
    z_max_ = declare_parameter<double>("z_max", 1.0);

    // QoS: best-effort, small queues to avoid backpressure
    auto be5 = rclcpp::SensorDataQoS().keep_last(5).best_effort();

    // normals
    output_normals_ = declare_parameter<bool>("output_normals", false);

    pub_ = create_publisher<sensor_msgs::msg::PointCloud2>(output_topic_, be5);

    sub_cloud_ = create_subscription<sensor_msgs::msg::PointCloud2>(
      cloud_topic_, be5,
      [this](sensor_msgs::msg::PointCloud2::ConstSharedPtr msg){
        {
          std::lock_guard<std::mutex> lk(mtx_);
          cloud_buf_.push_back(std::move(msg));
          trim_(cloud_buf_);
        }
        try_pair_and_process_();
      });

    sub_scan_ = create_subscription<sensor_msgs::msg::LaserScan>(
      scan_topic_, be5,
      [this](sensor_msgs::msg::LaserScan::ConstSharedPtr msg){
        {
          std::lock_guard<std::mutex> lk(mtx_);
          scan_buf_.push_back(std::move(msg));
          trim_(scan_buf_);
        }
        try_pair_and_process_();
      });

    RCLCPP_INFO(get_logger(),
      "1:1 fuse %s + %s -> %s  [target=%s, slop=%.0f ms, stride_u=%d, stride_v=%d, z=[%.2f,%.2f]]",
      cloud_topic_.c_str(), scan_topic_.c_str(), output_topic_.c_str(),
      target_frame_.c_str(), slop_s_*1000.0, cloud_stride_u_, cloud_stride_v_, z_min_, z_max_);
  }

private:
  // ---- helpers -------------------------------------------------------------

  template<class T>
  void trim_(std::deque<T> &q) {
    while ((int)q.size() > max_queue_) q.pop_front();
  }

  // Decimate camera cloud (organized) by (u,v) stride + ROI, keep scan stride separate. XYZ only.
  static void concat_xyz_filtered_perinput_(
      const sensor_msgs::msg::PointCloud2 &cam,   // possibly organized (W×H)
      const sensor_msgs::msg::PointCloud2 &scan,  // usually unorganized (1×N)
      sensor_msgs::msg::PointCloud2 &out,
      int cam_stride_u, int cam_stride_v, int scan_stride,
      float zmin, float zmax,
      double roi_top_frac, double roi_bottom_frac)
  {
    const uint32_t Wc = cam.width, Hc = cam.height;
    const bool cam_organized = (Hc > 1 && Wc > 0);
    const int ustep = std::max(1, cam_stride_u);
    const int vstep = std::max(1, cam_stride_v);
    const int sstep = std::max(1, scan_stride);

    int v_top = 0, v_bottom = static_cast<int>(Hc);
    if (cam_organized) {
      v_top    = static_cast<int>(std::floor(roi_top_frac * Hc));
      v_bottom = static_cast<int>(std::ceil((1.0 - roi_bottom_frac) * Hc));
      v_top    = std::clamp(v_top, 0, static_cast<int>(Hc));
      v_bottom = std::clamp(v_bottom, 0, static_cast<int>(Hc));
    }

    // ---- First pass: count kept points
    size_t keep_cam = 0, keep_scan = 0;

    {
      sensor_msgs::PointCloud2ConstIterator<float> cx(cam,"x"), cy(cam,"y"), cz(cam,"z");
      size_t i = 0;
      for (; cx != cx.end(); ++cx, ++cy, ++cz, ++i) {
        const float Z = *cz;
        if (!std::isfinite(Z) || Z < zmin || Z > zmax) continue;

        if (cam_organized) {
          const uint32_t u = static_cast<uint32_t>(i % Wc);
          const uint32_t v = static_cast<uint32_t>(i / Wc);
          if (static_cast<int>(v) < v_top || static_cast<int>(v) >= v_bottom) continue;
          if ((u % static_cast<uint32_t>(ustep)) || (v % static_cast<uint32_t>(vstep))) continue;
        } else {
          if ((i % static_cast<size_t>(ustep)) != 0) continue;
        }
        ++keep_cam;
      }
    }
    {
      sensor_msgs::PointCloud2ConstIterator<float> sx(scan,"x"), sy(scan,"y"), sz(scan,"z");
      size_t i = 0;
      for (; sx != sx.end(); ++sx, ++sy, ++sz, ++i) {
        if ((i % static_cast<size_t>(sstep)) != 0) continue;
        const float Z = *sz;
        if (!std::isfinite(Z) || Z < zmin || Z > zmax) continue;
        ++keep_scan;
      }
    }

    // ---- Allocate output (XYZ only, unorganized)
    out.header = cam.header; // caller will override stamp/frame
    out.height = 1;
    out.width  = keep_cam + keep_scan;
    out.is_bigendian = false;
    out.is_dense     = false;
    sensor_msgs::PointCloud2Modifier mod(out);
    mod.setPointCloud2FieldsByString(1, "xyz");
    mod.resize(out.width);

    // ---- Second pass: copy with prefix ++
    sensor_msgs::PointCloud2ConstIterator<float> cx2(cam,"x"), cy2(cam,"y"), cz2(cam,"z");
    sensor_msgs::PointCloud2ConstIterator<float> sx2(scan,"x"), sy2(scan,"y"), sz2(scan,"z");
    sensor_msgs::PointCloud2Iterator<float>       ox(out,"x"), oy(out,"y"), oz(out,"z");

    size_t i = 0;
    for (; cx2 != cx2.end(); ++cx2, ++cy2, ++cz2, ++i) {
      const float X = *cx2, Y = *cy2, Z = *cz2;
      if (!std::isfinite(Z) || Z < zmin || Z > zmax) continue;

      if (cam_organized) {
        const uint32_t u = static_cast<uint32_t>(i % Wc);
        const uint32_t v = static_cast<uint32_t>(i / Wc);
        if (static_cast<int>(v) < v_top || static_cast<int>(v) >= v_bottom) continue;
        if ((u % static_cast<uint32_t>(ustep)) || (v % static_cast<uint32_t>(vstep))) continue;
      } else {
        if ((i % static_cast<size_t>(ustep)) != 0) continue;
      }

      *ox = X; *oy = Y; *oz = Z;
      ++ox; ++oy; ++oz;
    }

    i = 0;
    for (; sx2 != sx2.end(); ++sx2, ++sy2, ++sz2, ++i) {
      if ((i % static_cast<size_t>(sstep)) != 0) continue;
      const float X = *sx2, Y = *sy2, Z = *sz2;
      if (!std::isfinite(Z) || Z < zmin || Z > zmax) continue;

      *ox = X; *oy = Y; *oz = Z;
      ++ox; ++oy; ++oz;
    }
  }

  static void concat_xyz_normals_perinput(
    const sensor_msgs::msg::PointCloud2 &cam,   // may have normals
    const sensor_msgs::msg::PointCloud2 &scan,  // no normals
    sensor_msgs::msg::PointCloud2 &out,
    int cam_stride_u, int cam_stride_v, int scan_stride,
    float zmin, float zmax,
    double roi_top_frac, double roi_bottom_frac,
    bool output_normals)
  {
    const bool cam_has_normals =
        has_field(cam,"normal_x") && has_field(cam,"normal_y") && has_field(cam,"normal_z");

    const uint32_t Wc = cam.width, Hc = cam.height;
    const bool cam_organized = (Hc > 1 && Wc > 0);
    const int ustep = std::max(1, cam_stride_u);
    const int vstep = std::max(1, cam_stride_v);
    const int sstep = std::max(1, scan_stride);

    int v_top = 0, v_bottom = static_cast<int>(Hc);
    if (cam_organized) {
        v_top    = static_cast<int>(std::floor(roi_top_frac * Hc));
        v_bottom = static_cast<int>(std::ceil((1.0 - roi_bottom_frac) * Hc));
        v_top    = std::clamp(v_top, 0, static_cast<int>(Hc));
        v_bottom = std::clamp(v_bottom, 0, static_cast<int>(Hc));
    }

    // count kept points
    size_t keep_cam = 0, keep_scan = 0;
    {
        sensor_msgs::PointCloud2ConstIterator<float> cx(cam,"x"), cy(cam,"y"), cz(cam,"z");
        size_t i = 0;
        for (; cx != cx.end(); ++cx, ++cy, ++cz, ++i) {
        const float Z = *cz;
        if (!std::isfinite(Z) || Z < zmin || Z > zmax) continue;
        if (cam_organized) {
            const uint32_t u = (uint32_t)(i % Wc);
            const uint32_t v = (uint32_t)(i / Wc);
            if ((int)v < v_top || (int)v >= v_bottom) continue;
            if ((u % (uint32_t)ustep) || (v % (uint32_t)vstep)) continue;
        } else if ((i % (size_t)ustep) != 0) continue;
        ++keep_cam;
        }
    }
    {
        sensor_msgs::PointCloud2ConstIterator<float> sx(scan,"x"), sy(scan,"y"), sz(scan,"z");
        size_t i = 0;
        for (; sx != sx.end(); ++sx, ++sy, ++sz, ++i) {
        const float Z = *sz;
        if (!std::isfinite(Z) || Z < zmin || Z > zmax) continue;
        if ((i % (size_t)sstep) != 0) continue;
        ++keep_scan;
        }
    }

    // allocate output fields
    out.header = cam.header; // will overwrite stamp/frame later
    out.height = 1;
    out.width  = keep_cam + keep_scan;
    out.is_bigendian = false; out.is_dense = false;
    sensor_msgs::PointCloud2Modifier mod(out);

    if (output_normals && cam_has_normals) {
        mod.setPointCloud2Fields(
        6,
        "x", 1, sensor_msgs::msg::PointField::FLOAT32,
        "y", 1, sensor_msgs::msg::PointField::FLOAT32,
        "z", 1, sensor_msgs::msg::PointField::FLOAT32,
        "normal_x", 1, sensor_msgs::msg::PointField::FLOAT32,
        "normal_y", 1, sensor_msgs::msg::PointField::FLOAT32,
        "normal_z", 1, sensor_msgs::msg::PointField::FLOAT32
        );
    } else {
        mod.setPointCloud2FieldsByString(1, "xyz");
    }
    mod.resize(out.width);

    // iterators
    sensor_msgs::PointCloud2ConstIterator<float> cx2(cam,"x"), cy2(cam,"y"), cz2(cam,"z");
    sensor_msgs::PointCloud2Iterator<float>       ox(out,"x"), oy(out,"y"), oz(out,"z");

    // optional normal iterators from camera
    std::unique_ptr<sensor_msgs::PointCloud2ConstIterator<float>> cnx, cny, cnz;
    std::unique_ptr<sensor_msgs::PointCloud2Iterator<float>> onx, ony, onz;
    if (output_normals && cam_has_normals) {
        cnx.reset(new sensor_msgs::PointCloud2ConstIterator<float>(cam,"normal_x"));
        cny.reset(new sensor_msgs::PointCloud2ConstIterator<float>(cam,"normal_y"));
        cnz.reset(new sensor_msgs::PointCloud2ConstIterator<float>(cam,"normal_z"));
        onx.reset(new sensor_msgs::PointCloud2Iterator<float>(out,"normal_x"));
        ony.reset(new sensor_msgs::PointCloud2Iterator<float>(out,"normal_y"));
        onz.reset(new sensor_msgs::PointCloud2Iterator<float>(out,"normal_z"));
    }

    // copy camera points (+ normals if present)
    size_t i = 0;
    for (; cx2 != cx2.end(); ++cx2, ++cy2, ++cz2, ++i) {
        const float X = *cx2, Y = *cy2, Z = *cz2;
        if (!std::isfinite(Z) || Z < zmin || Z > zmax) {
        if (cnx) { ++(*cnx); ++(*cny); ++(*cnz); } // keep normal iterators in lockstep
        continue;
        }
        if (cam_organized) {
        const uint32_t u = (uint32_t)(i % Wc);
        const uint32_t v = (uint32_t)(i / Wc);
        if ((int)v < v_top || (int)v >= v_bottom) { if (cnx){ ++(*cnx); ++(*cny); ++(*cnz);} continue; }
        if ((u % (uint32_t)ustep) || (v % (uint32_t)vstep)) { if (cnx){ ++(*cnx); ++(*cny); ++(*cnz);} continue; }
        } else if ((i % (size_t)ustep) != 0) { if (cnx){ ++(*cnx); ++(*cny); ++(*cnz);} continue; }

        *ox = X; *oy = Y; *oz = Z; ++ox; ++oy; ++oz;

        if (cnx) {
        const float NX = **cnx, NY = **cny, NZ = **cnz;
        *(*onx) = NX; *(*ony) = NY; *(*onz) = NZ;
        ++(*onx); ++(*ony); ++(*onz);
        ++(*cnx); ++(*cny); ++(*cnz);
        }
    }

    // copy scan points; normals → NaN if output_normals=true
    sensor_msgs::PointCloud2ConstIterator<float> sx2(scan,"x"), sy2(scan,"y"), sz2(scan,"z");
    const float NaN = std::numeric_limits<float>::quiet_NaN();
    i = 0;
    for (; sx2 != sx2.end(); ++sx2, ++sy2, ++sz2, ++i) {
        if ((i % (size_t)sstep) != 0) continue;
        const float X = *sx2, Y = *sy2, Z = *sz2;
        if (!std::isfinite(Z) || Z < zmin || Z > zmax) continue;

        *ox = X; *oy = Y; *oz = Z; ++ox; ++oy; ++oz;

        if (onx) { *(*onx) = NaN; *(*ony) = NaN; *(*onz) = NaN; ++(*onx); ++(*ony); ++(*onz); }
    }
  }

  // Try to form pairs from the heads of both queues and process them (1:1)
  void try_pair_and_process_() {
    while (true) {
      sensor_msgs::msg::PointCloud2::ConstSharedPtr cloud;
      sensor_msgs::msg::LaserScan::ConstSharedPtr scan;

      {
        std::lock_guard<std::mutex> lk(mtx_);
        if (cloud_buf_.empty() || scan_buf_.empty()) return;

        auto &c = cloud_buf_.front();
        auto &s = scan_buf_.front();
        rclcpp::Time tc(c->header.stamp), ts(s->header.stamp);

        // absolute time difference
        rclcpp::Duration dt = (tc > ts) ? (tc - ts) : (ts - tc);
        if (dt > rclcpp::Duration::from_seconds(slop_s_)) {
          // drop the older one to maintain 1:1 without reuse
          if (tc < ts) cloud_buf_.pop_front(); else scan_buf_.pop_front();
          continue; // try again
        }

        // pair is good: consume both
        cloud = c; scan = s;
        cloud_buf_.pop_front();
        scan_buf_.pop_front();
      }

      // process outside the lock
      process_pair_(cloud, scan);
      // loop to see if more pairs are ready
    }
  }

  void process_pair_(const sensor_msgs::msg::PointCloud2::ConstSharedPtr &cloud,
                     const sensor_msgs::msg::LaserScan::ConstSharedPtr &scan)
  {
    // 1) Project scan directly into target_frame (non-blocking TF)
    if (!tf_buffer_.canTransform(target_frame_, scan->header.frame_id,
                                 rclcpp::Time(scan->header.stamp))) {
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000,
                           "No TF for scan at stamp; dropping.");
      return;
    }
    sensor_msgs::msg::PointCloud2 scan_tf;
    try {
      projector_.transformLaserScanToPointCloud(target_frame_, *scan, scan_tf, tf_buffer_);
    } catch (const std::exception &e) {
      RCLCPP_ERROR(get_logger(), "laser_geometry projection failed: %s", e.what());
      return;
    }

    // 2) Transform camera cloud to target_frame (non-blocking)
    if (!tf_buffer_.canTransform(target_frame_, cloud->header.frame_id,
                                 rclcpp::Time(cloud->header.stamp))) {
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000,
                           "No TF for cloud at stamp; dropping.");
      return;
    }
    sensor_msgs::msg::PointCloud2 cam_tf;
    auto tf = tf_buffer_.lookupTransform(
        target_frame_, cloud->header.frame_id, rclcpp::Time(cloud->header.stamp));
    tf2::doTransform(*cloud, cam_tf, tf);

    // 3) Concatenate (XYZ only) with per-input decimation + ROI
    sensor_msgs::msg::PointCloud2 out;
    out.header.frame_id = target_frame_;
    out.header.stamp = (rclcpp::Time(cloud->header.stamp) > rclcpp::Time(scan->header.stamp))
                     ? cloud->header.stamp : scan->header.stamp;

    concat_xyz_filtered_perinput_(
      cam_tf, scan_tf, out,
      cloud_stride_u_, cloud_stride_v_, /*scan_stride=*/scan_stride_,
      static_cast<float>(z_min_), static_cast<float>(z_max_),
      roi_top_frac_, roi_bottom_frac_);

    // todo: switch to this
    // concat_xyz_normals_perinput(
    //   cam_tf, scan_tf, out,
    //   cloud_stride_u_, cloud_stride_v_, scan_stride_,
    //   (float)z_min_, (float)z_max_,
    //   roi_top_frac_, roi_bottom_frac_,
    //   output_normals_);

    pub_->publish(out);
  }

  // ---- members -------------------------------------------------------------
  // Params
  std::string cloud_topic_, scan_topic_, output_topic_, target_frame_;
  bool output_normals_;
  double slop_s_{0.02};
  int max_queue_{30};
  int cloud_stride_u_{2}, cloud_stride_v_{2}, scan_stride_{1};
  double roi_top_frac_{0.0}, roi_bottom_frac_{0.0};
  double z_min_{0.0}, z_max_{6.0};

  // Comms
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pub_;
  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr sub_cloud_;
  rclcpp::Subscription<sensor_msgs::msg::LaserScan>::SharedPtr   sub_scan_;

  // Buffers
  std::mutex mtx_;
  std::deque<sensor_msgs::msg::PointCloud2::ConstSharedPtr> cloud_buf_;
  std::deque<sensor_msgs::msg::LaserScan::ConstSharedPtr>   scan_buf_;

  // TF + projector (order matters: buffer before listener)
  tf2_ros::Buffer tf_buffer_;
  tf2_ros::TransformListener tf_listener_;
  laser_geometry::LaserProjection projector_;
};

int main(int argc, char **argv) {
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<CloudScanFuser1to1>());
  rclcpp::shutdown();
  return 0;
}
