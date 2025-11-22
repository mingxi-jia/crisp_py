"""A ROS2-based point cloud processing example."""

# %%
import matplotlib.pyplot as plt
import numpy as np
import time
import yaml
from pathlib import Path
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from message_filters import ApproximateTimeSynchronizer, Subscriber
from scipy.spatial.transform import Rotation
import open3d as o3d

class PointCloudManager(Node):
    """Synchronizes and merges point clouds from three RGBD cameras."""

    def __init__(self, config_path: str, downsample: int = 2):
        super().__init__("point_cloud_manager")
        self.bridge = CvBridge()
        self.rgb_images = []
        self.depth_images = []
        self.latest_pcd = None
        self.downsample = downsample  # Downsample factor for faster processing
        self.callback_count = 0

        # Load camera parameters
        with open(config_path, "r") as f:
            self.cam_params = yaml.safe_load(f)

        # Precompute rotation matrices for faster processing
        self.transforms = {}
        for cam_name in ["cam1", "cam2", "cam3"]:
            R = Rotation.from_quat(self.cam_params[cam_name]["q"]).as_matrix()
            t = np.array(self.cam_params[cam_name]["t"])
            self.transforms[cam_name] = (R, t)

        # QoS profile for best effort, low latency
        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )

        # Subscribe to RGB and depth topics
        self.rgb_subs = []
        self.depth_subs = []
        for i in [1, 2, 3]:
            rgb_sub = Subscriber(self, Image, f"/cam{i}/color/image_raw", qos_profile=qos)
            depth_sub = Subscriber(self, Image, f"/cam{i}/aligned_depth_to_color/image_raw", qos_profile=qos)
            self.rgb_subs.append(rgb_sub)
            self.depth_subs.append(depth_sub)

        # Synchronize all 6 topics with larger queue and more lenient timing
        all_subs = self.rgb_subs + self.depth_subs
        self.sync = ApproximateTimeSynchronizer(
            all_subs, queue_size=100, slop=0.05
        )
        self.sync.registerCallback(self.sync_callback)

        self.get_logger().info("PointCloudManager initialized")

    def rgbd_to_pointcloud(self, rgb, depth, cam_params):
        """Convert RGB-D images to point cloud using camera intrinsics."""
        # Downsample for faster processing
        if self.downsample > 1:
            rgb = rgb[::self.downsample, ::self.downsample]
            depth = depth[::self.downsample, ::self.downsample]

        h, w = depth.shape
        k = np.array(cam_params["k"]).reshape(3, 3)
        fx, fy = k[0, 0], k[1, 1]
        cx, cy = k[0, 2], k[1, 2]

        # Adjust intrinsics for downsampling
        if self.downsample > 1:
            fx /= self.downsample
            fy /= self.downsample
            cx /= self.downsample
            cy /= self.downsample

        # Create mesh grid
        u, v = np.meshgrid(np.arange(w), np.arange(h))

        # Convert depth to meters (assuming depth is in mm)
        z = depth.astype(np.float32) / 1000.0

        # Filter valid depths first
        valid = z > 0

        # Back-project only valid points (faster)
        u_valid = u[valid]
        v_valid = v[valid]
        z_valid = z[valid]

        x = (u_valid - cx) * z_valid / fx
        y = (v_valid - cy) * z_valid / fy

        points = np.stack([x, y, z_valid], axis=1)
        colors = rgb[valid].astype(np.float32) / 255.0

        return points, colors

    def transform_pointcloud(self, points, R, t):
        """Transform point cloud to robot frame using extrinsics."""
        # Apply transformation: p_world = R * p_cam + t
        return (R @ points.T).T + t

    def sync_callback(self, rgb1, rgb2, rgb3, depth1, depth2, depth3):
        """Process synchronized RGB-D images from all cameras."""
        try:
            # Convert ROS images to numpy
            self.rgb_images = [
                self.bridge.imgmsg_to_cv2(img, "rgb8")
                for img in [rgb1, rgb2, rgb3]
            ]
            self.depth_images = [
                self.bridge.imgmsg_to_cv2(img, "16UC1")
                for img in [depth1, depth2, depth3]
            ]

            
            # Log less frequently to reduce overhead
            self.callback_count += 1
            # if self.callback_count % 10 == 0:
            #     self.get_logger().info(f"Processed {self.callback_count} point clouds")

        except Exception as e:
            self.get_logger().error(f"Error in callback: {e}")

    def get_latest_pointcloud(self):
        """Return the latest merged point cloud."""
        self.rgb_images = []
        self.depth_images = []
        while self.rgb_images == [] or self.depth_images == []:
            time.sleep(0.01)  # Wait for first callback
            print("No point cloud received yet.")

        # Process each camera
        all_points = []
        all_colors = []

        for i, (rgb, depth) in enumerate(zip(self.rgb_images, self.depth_images), 1):
            cam_name = f"cam{i}"
            cam_params = self.cam_params[cam_name]
            R, t = self.transforms[cam_name]

            # Convert to point cloud
            points, colors = self.rgbd_to_pointcloud(rgb, depth, cam_params)

            # Transform to robot frame
            points = self.transform_pointcloud(points, R, t)

            all_points.append(points)
            all_colors.append(colors)

        # Merge point clouds
        merged_points = np.vstack(all_points)
        merged_colors = np.vstack(all_colors)

        latest_pcd = np.concatenate([merged_points, merged_colors], axis=1)

        return latest_pcd


# %%
def main():
    """Test the PointCloudManager."""
    rclpy.init()

    # Get config path
    config_path = Path(__file__).parent.parent / "config" / "camera_info.yaml"

    # Create manager with downsampling for faster processing
    manager = PointCloudManager(str(config_path), downsample=2)

    # Spin in background thread to receive messages
    import threading
    spin_thread = threading.Thread(target=rclpy.spin, args=(manager,), daemon=True)
    spin_thread.start()

    # Rate tracking
    last_pcd = None
    last_update_time = None
    update_times = []
    max_samples = 100
    check_count = 0
    start_time = time.time()
    duration = 10.0  # Run for 10 seconds

    # Wait and measure rate
    print("Measuring point cloud rate...")
    while time.time() - start_time < duration:
        time.sleep(0.01)  # Poll at 100 Hz
        pcd = manager.get_latest_pointcloud()
        check_count += 1

        # Track rate when new point cloud arrives
        if pcd is not None and pcd != last_pcd:
            current_time = time.time()
            if last_update_time is not None:
                dt = current_time - last_update_time
                update_times.append(dt)
                if len(update_times) > max_samples:
                    update_times.pop(0)

                # Print every 15 updates (approximately once per second at 15 Hz)
                if len(update_times) % 15 == 0:
                    avg_dt = np.mean(update_times[-30:])  # Use last 30 samples
                    rate = 1.0 / avg_dt if avg_dt > 0 else 0
                    print(f"Rate: {rate:.2f} Hz | Points: {len(pcd.points)} | Updates: {len(update_times)}")

            last_update_time = current_time
            last_pcd = pcd

    # Final stats
    if update_times:
        avg_rate = 1.0 / np.mean(update_times)
        print(f"\n=== Final Stats ===")
        print(f"Total updates: {len(update_times)}")
        print(f"Average rate: {avg_rate:.2f} Hz")
        print(f"Min interval: {min(update_times):.3f} s ({1.0/min(update_times):.1f} Hz)")
        print(f"Max interval: {max(update_times):.3f} s ({1.0/max(update_times):.1f} Hz)")
    else:
        print("No point clouds received!")

    # Cleanup
    manager.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
