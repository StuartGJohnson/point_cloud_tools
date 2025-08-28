#!/usr/bin/env python3
# file: snap_depth_cloud_and_calib.py
import os
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, PointCloud2
from cv_bridge import CvBridge
from tf2_ros import Buffer, TransformListener
from tf2_sensor_msgs.tf2_sensor_msgs import do_transform_cloud
import message_filters as mf
from sensor_msgs_py import point_cloud2 as pc2


def save_ply_xyz(path, pts_xyz):
    # pts_xyz: (N,3) float32 in meters
    with open(path, 'w') as f:
        f.write('ply\nformat ascii 1.0\n')
        f.write(f'element vertex {pts_xyz.shape[0]}\n')
        f.write('property float x\nproperty float y\nproperty float z\nend_header\n')
        for x, y, z in pts_xyz:
            f.write(f'{x} {y} {z}\n')


class Snap(Node):
    def __init__(self):
        super().__init__('snap_depth_cloud_and_calib')
        # --- params ---
        self.declare_parameter('depth_topic', '/camera/depth_image')
        self.declare_parameter('cloud_topic', '/camera/points')
        self.declare_parameter('target_frame', 'camera_optical_link')
        self.declare_parameter('out_dir', '/tmp')
        self.declare_parameter('sample_stride', 4)  # for K fit speed

        self.depth_topic = self.get_parameter('depth_topic').get_parameter_value().string_value
        self.cloud_topic = self.get_parameter('cloud_topic').get_parameter_value().string_value
        self.target_frame = self.get_parameter('target_frame').get_parameter_value().string_value
        self.out_dir = self.get_parameter('out_dir').get_parameter_value().string_value
        self.stride = int(self.get_parameter('sample_stride').get_parameter_value().integer_value or 4)

        os.makedirs(self.out_dir, exist_ok=True)

        self.bridge = CvBridge()
        self.tfbuf = Buffer()
        self.tfl = TransformListener(self.tfbuf, self)

        # QoS: sensor data
        depth_sub = mf.Subscriber(self, Image, self.depth_topic)
        cloud_sub = mf.Subscriber(self, PointCloud2, self.cloud_topic)

        # approx sync is fine; you said drops are OK
        sync = mf.ApproximateTimeSynchronizer([depth_sub, cloud_sub], queue_size=10, slop=0.05)
        sync.registerCallback(self.cb)

        self.got = False
        self.get_logger().info(f'Waiting for one pair: {self.depth_topic} + {self.cloud_topic}')

    def cb(self, depth_msg: Image, cloud_msg: PointCloud2):
        if self.got:
            return
        self.got = True
        try:
            # 1) Convert depth to float meters
            enc = depth_msg.encoding.lower()
            img = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding='passthrough')
            if enc in ('16uc1', 'mono16'):
                depth_m = img.astype(np.float32) * 1e-3
            else:
                depth_m = img.astype(np.float32)  # expect 32FC1 meters
            H, W = depth_m.shape
            depth_path_npy = os.path.join(self.out_dir, 'depth.npy')
            np.save(depth_path_npy, depth_m)
            self.get_logger().info(f'Saved depth: {depth_path_npy}  ({W}x{H}, {enc})')

            # 2) Transform cloud to target_frame if needed
            if cloud_msg.header.frame_id != self.target_frame:
                try:
                    tf = self.tfbuf.lookup_transform(self.target_frame,
                                                     cloud_msg.header.frame_id,
                                                     rclpy.time.Time())
                    cloud_t = do_transform_cloud(cloud_msg, tf)
                except Exception as e:
                    self.get_logger().warn(f'TF lookup/transform failed, using original frame: {e}')
                    cloud_t = cloud_msg
            else:
                cloud_t = cloud_msg

            self.get_logger().info("hello")

            # 3) Extract XYZ (keep NaNs)
            # 3) Extract XYZ (keep NaNs), detect organized
            try:
                # Preferred: fast structured→plain conversion
                arr = pc2.read_points_numpy(cloud_t, field_names=('x', 'y', 'z'), skip_nans=False)
                if hasattr(arr, 'dtype') and arr.dtype.names:  # structured
                    if getattr(arr, 'ndim', 1) == 2:  # (H,W)
                        xyz = np.dstack((arr['x'], arr['y'], arr['z'])).astype(np.float32)
                        organized = True
                    else:  # (N,)
                        xyz = np.stack((arr['x'], arr['y'], arr['z']), axis=-1).astype(np.float32)
                        organized = (cloud_t.height > 1 and cloud_t.width > 1 and
                                     xyz.shape[0] == cloud_t.height * cloud_t.width)
                        if organized:
                            xyz = xyz.reshape((cloud_t.height, cloud_t.width, 3))
                else:
                    # Some versions can return plain ndarray already
                    xyz = arr.astype(np.float32)
                    organized = (cloud_t.height > 1 and cloud_t.width > 1 and
                                 xyz.shape[0] == cloud_t.height * cloud_t.width)
                    if organized and xyz.ndim == 2 and xyz.shape[1] == 3:
                        xyz = xyz.reshape((cloud_t.height, cloud_t.width, 3))
            except Exception:
                # Fallback: generator path
                pts_list = list(pc2.read_points(cloud_t, field_names=('x', 'y', 'z'), skip_nans=False))
                xyz = np.asarray(pts_list, dtype=np.float32)
                organized = (cloud_t.height > 1 and cloud_t.width > 1 and
                             xyz.shape[0] == cloud_t.height * cloud_t.width)
                if organized:
                    xyz = xyz.reshape((cloud_t.height, cloud_t.width, 3))
            # Save cloud (ASCII PLY)
            ply_path = os.path.join(self.out_dir, 'cloud_target_frame.ply')
            save_ply_xyz(ply_path, xyz.reshape(-1, 3))
            self.get_logger().info(
                f'Saved cloud: {ply_path}  ({cloud_t.width}x{cloud_t.height}) in {self.target_frame}')

            # 4) If organized and same size as depth, estimate K
            if organized and cloud_t.width == W and cloud_t.height == H:
                fx, fy, cx, cy = self.estimate_K_from_cloud_only(xyz)
                if fx is not None:
                    K = [fx, 0., cx, 0., fy, cy, 0., 0., 1.]
                    P = [fx, 0., cx, 0., 0., fy, cy, 0., 0., 0., 1., 0.]
                    np.save(os.path.join(self.out_dir, 'K.npy'), np.array(K, dtype=np.float64))
                    np.save(os.path.join(self.out_dir, 'P.npy'), np.array(P, dtype=np.float64))
                    self.get_logger().info(f'Estimated K: fx={fx:.3f}, fy={fy:.3f}, cx={cx:.3f}, cy={cy:.3f}')
                else:
                    self.get_logger().warn('Could not estimate K (insufficient valid data).')
            else:
                self.get_logger().warn('Cloud not organized or size mismatch with depth; skipping K estimation.')

            # Done after one pair
            self.get_logger().info('Done. Shutting down.')
            rclpy.shutdown()
        except Exception as e:
            # print full stack, keep the exception
            self.get_logger().error('Unhandled exception in callback:\n' + traceback.format_exc())
            raise

    def estimate_K(self, depth_m: np.ndarray, xyz_img: np.ndarray):
        H, W, _ = xyz_img.shape
        # Sample pixels to keep it quick
        us = range(0, W, self.stride)
        vs = range(0, H, self.stride)
        Au, bu, Av, bv = [], [], [], []
        valid = 0
        for v in vs:
            for u in us:
                Z = xyz_img[v, u, 2]
                X = xyz_img[v, u, 0]
                Y = xyz_img[v, u, 1]
                D = depth_m[v, u]
                if not np.isfinite(Z) or not np.isfinite(X) or not np.isfinite(Y):
                    continue
                if D <= 0 or Z <= 0:
                    continue
                # Build linear systems: u = fx*(X/Z) + cx, v = fy*(Y/Z) + cy
                Au.append([X / Z, 1.0]);
                bu.append(float(u))
                Av.append([Y / Z, 1.0]);
                bv.append(float(v))
                valid += 1
        if valid < 50:
            return None, None, None, None
        Au = np.asarray(Au);
        bu = np.asarray(bu)
        Av = np.asarray(Av);
        bv = np.asarray(bv)
        fx, cx = np.linalg.lstsq(Au, bu, rcond=None)[0]
        fy, cy = np.linalg.lstsq(Av, bv, rcond=None)[0]
        return float(fx), float(fy), float(cx), float(cy)

    def estimate_K_from_pairs(self, depth_m: np.ndarray, xyz_img: np.ndarray):
        """
        depth_m: (H,W) float32 meters
        xyz_img: (H,W,3) float32 in *camera optical frame* (Z forward, X right, Y down)
        Returns fx, fy, cx, cy or (None,)*4 if not enough data.
        """
        H, W, _ = xyz_img.shape

        us = range(0, W, self.stride)
        vs = range(0, H, self.stride)

        rows_u, rows_v = [], []
        bu, bv = [], []
        count = 0

        for v in vs:
            for u in us:
                X, Y, Z = xyz_img[v, u]
                D = depth_m[v, u]
                if not np.isfinite(X) or not np.isfinite(Y) or not np.isfinite(Z):
                    continue
                if Z <= 0 or D <= 0:
                    continue
                # Optional consistency check: depth vs Z
                if abs(Z - D) > max(0.01, 0.05 * Z):  # >1 cm or 5% error → skip pair
                    continue

                rows_u.append([X / Z, 1.0])
                bu.append(float(u))
                rows_v.append([Y / Z, 1.0])
                bv.append(float(v))
                count += 1

        if count < 100:
            return (None, None, None, None)

        Au = np.asarray(rows_u)
        bu = np.asarray(bu)
        Av = np.asarray(rows_v)
        bv = np.asarray(bv)

        # Detect Y sign (Y down vs Y up) by correlation with v
        corr = np.corrcoef(Av[:, 0], bv)[0, 1]
        if not np.isfinite(corr):
            return (None, None, None, None)
        if corr < 0:
            # v increases downward in images; if correlation is negative,
            # flip Y to match optical convention
            Av[:, 0] *= -1.0

        fx, cx = np.linalg.lstsq(Au, bu, rcond=None)[0]
        fy, cy = np.linalg.lstsq(Av, bv, rcond=None)[0]

        # Sanity bounds (optional)
        if not (0 < fx < 5000 and 0 < fy < 5000):
            return (None, None, None, None)

        return float(fx), float(fy), float(cx), float(cy)

    def estimate_K_from_cloud_only(self, xyz_img: np.ndarray):
        H, W, _ = xyz_img.shape
        us = range(0, W, self.stride)
        vs = range(0, H, self.stride)
        Au, bu, Av, bv = [], [], [], []
        for v in vs:
            for u in us:
                X, Y, Z = xyz_img[v, u]
                if not np.isfinite(X) or not np.isfinite(Y) or not np.isfinite(Z) or Z <= 0:
                    continue
                Au.append([X / Z, 1.0])
                bu.append(float(u))
                Av.append([Y / Z, 1.0])
                bv.append(float(v))

        Au = np.asarray(Au)
        bu = np.asarray(bu)
        Av = np.asarray(Av)
        bv = np.asarray(bv)
        if Au.shape[0] < 10 or Av.shape[0] < 10:
            return (None, None, None, None)

        # Detect Y sign (Y-down vs Y-up) and flip if needed
        corr = np.corrcoef(Av[:, 0], bv)[0, 1]
        if np.isfinite(corr) and corr < 0:
            Av[:, 0] *= -1.0

        fx, cx = np.linalg.lstsq(Au, bu, rcond=None)[0]
        fy, cy = np.linalg.lstsq(Av, bv, rcond=None)[0]

        # Optional sanity
        # if not (0 < fx < 5000 and 0 < fy < 5000):
        #     return (None, None, None, None)
        return float(fx), float(fy), float(cx), float(cy)


def main():
    rclpy.init()
    rclpy.spin(Snap())


if __name__ == '__main__':
    main()
