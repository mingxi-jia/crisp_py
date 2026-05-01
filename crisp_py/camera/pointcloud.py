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
from sensor_msgs.msg import Image, JointState
from geometry_msgs.msg import WrenchStamped
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
        self.inhand_image = None
        self.inhand_depth = None
        self.downsample = downsample  # Downsample factor for faster processing
        self.callback_count = 0
        fps = 20.0  # Expected camera FPS

        # Load camera parameters
        with open(config_path, "r") as f:
            print(f"Loading camera config from: {config_path}")
            self.cam_params = yaml.safe_load(f)

        # Precompute rotation matrices for faster processing
        self.transforms = {}
        for cam_name in ["cam1", "cam2", "cam3"]:
            print(f"Camera {cam_name} extrinsics: {self.cam_params[cam_name]}")
            R = Rotation.from_quat(self.cam_params[cam_name]["q"]).as_matrix()
            t = np.array(self.cam_params[cam_name]["t"])
            self.transforms[cam_name] = (R, t)

        qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )

        # Subscribe to RGB and depth topics
        self.rgb_subs = []
        self.depth_subs = []
        self._external_cams = [3]
        self._name2idx = {f"cam{i}": k for k, i in enumerate(self._external_cams)}

        for i in self._external_cams:
            rgb_sub = Subscriber(self, Image, f"/cam{i}/cam{i}/color/image_raw", qos_profile=qos)
            depth_sub = Subscriber(self, Image, f"/cam{i}/cam{i}/aligned_depth_to_color/image_raw", qos_profile=qos)
            self.rgb_subs.append(rgb_sub)
            self.depth_subs.append(depth_sub)

        self.inhand_rgb_sub = Subscriber(self, Image, f"/cam4/cam4/color/image_rect_raw", qos_profile=qos)
        self.inhand_depth_sub = Subscriber(self, Image, f"/cam4/cam4/aligned_depth_to_color/image_raw", qos_profile=qos)
        self.inhand_sub = [self.inhand_rgb_sub, self.inhand_depth_sub]

        # Synchronize all 6 topics with larger queue and more lenient timing
        all_subs = self.rgb_subs + self.depth_subs + self.inhand_sub
        self.sync = ApproximateTimeSynchronizer(
            all_subs, queue_size=100, slop=0.5
        )
        self.sync.registerCallback(self.sync_callback)

        # Frequency monitoring
        self._freq_monitor_start_time = time.time()
        self._freq_monitor_count = 0
        self._freq_monitor_last_log_time = time.time()
        self._freq_monitor_interval = 5.0  # Log every 5 seconds
        self._freq_monitor_enabled = True

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

    def sync_callback(self, rgb3, depth3, inhand_rgb, inhand_depth):
        """Process synchronized RGB-D images from all cameras."""
        if self.callback_count % 30 == 0:
            print(f"[sync_callback] fired (count={self.callback_count})", flush=True)
        try:
            # Convert ROS images to numpy
            self.rgb_images = [
                self.bridge.imgmsg_to_cv2(img, "rgb8")
                for img in [rgb3]
            ]
            self.depth_images = [
                self.bridge.imgmsg_to_cv2(img, "16UC1")
                for img in [depth3]
            ]
            self.inhand_image = self.bridge.imgmsg_to_cv2(inhand_rgb, "rgb8")
            self.inhand_depth = self.bridge.imgmsg_to_cv2(inhand_depth, "16UC1")/1000.0

            
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
            time.sleep(0.002)  # Wait for first callback
            # print("No point cloud received yet.")

        # Process each camera
        all_points = []
        all_colors = []

        for i, (rgb, depth) in enumerate(zip(self.rgb_images, self.depth_images), 1):
            # if i != 3:
            #     continue
            # if i == 1:
            #     continue
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

    def get_latest_rgbd(self, cam_name: str, timeout: float = 5.0):
        """Return the latest RGB image from specified camera."""
        deadline = time.time() + timeout
        if cam_name == "cam4":
            while (self.inhand_image is None) or (self.inhand_depth is None):
                if time.time() > deadline:
                    print(f"[get_latest_rgbd] TIMEOUT cam4 after {timeout}s — sync_callback never fired")
                    return None, None
                time.sleep(0.01)
            return self.inhand_image, self.inhand_depth
        else:
            while self.rgb_images == [] or self.depth_images == []:
                if time.time() > deadline:
                    print(f"[get_latest_rgbd] TIMEOUT {cam_name} after {timeout}s — sync_callback never fired")
                    return None, None
                time.sleep(0.01)
            idx = self._name2idx[cam_name]
            return self.rgb_images[idx], self.depth_images[idx]

    def clear_cache(self):
        """Clear stored images to free memory."""
        self.rgb_images = []
        self.depth_images = []
        self.inhand_image = None
        self.inhand_depth = None


class ObservationManager(PointCloudManager):
    """Unified observation node: cameras + joint states + gripper + (optional) FT.

    Adds /joint_states, /gripper/gripper_state, and /ft/.../wrench subscriptions to
    the same ROS node so a single executor spin yields the complete observation.
    Pass `robot` and `gripper` (crisp_py.robot.Robot / crisp_py.gripper.Gripper) so
    EE pose and gripper command value are reachable via a single get_obs() call.
    """

    FRANKA_JOINTS = [
        "fr3_joint1", "fr3_joint2", "fr3_joint3", "fr3_joint4",
        "fr3_joint5", "fr3_joint6", "fr3_joint7",
    ]
    GRIPPER_JOINTS = ["gripper_joint"]

    def __init__(self,
                 config_path: str,
                 robot=None,
                 gripper=None,
                 ft_sensor_on: bool = False,
                 downsample: int = 2):
        super().__init__(config_path, downsample=downsample)
        self.robot = robot
        self.gripper = gripper
        self.ft_sensor_on = ft_sensor_on

        self._franka_received = False
        self._gripper_received = False
        self._ft_received = False

        self.franka_joint_array = None
        self.gripper_joint_array = None
        self.gripper_torque_array = None

        self.create_subscription(JointState, "/joint_states", self._franka_cb, 10)
        self.create_subscription(JointState, "/gripper/gripper_state", self._gripper_cb, 10)
        self.create_subscription(WrenchStamped,
                                 "/ft/robotiq_force_torque_sensor_broadcaster/wrench",
                                 self._ft_cb, 10)

    def _franka_cb(self, msg: JointState):
        arr = [p for n, p in zip(msg.name, msg.position) if n in self.FRANKA_JOINTS]
        if arr:
            self.franka_joint_array = np.asarray(arr, dtype=np.float32)
            self._franka_received = True

    def _gripper_cb(self, msg: JointState):
        arr = []
        for n, p in zip(msg.name, msg.position):
            if n in self.GRIPPER_JOINTS:
                arr.append(1.0 if p >= 0.95 else 0.0)
        if arr:
            self.gripper_joint_array = np.asarray(arr, dtype=np.float32)
            self._gripper_received = True

    def _ft_cb(self, msg: WrenchStamped):
        w = msg.wrench
        self.gripper_torque_array = np.asarray(
            [w.force.x, w.force.y, w.force.z,
             w.torque.x, w.torque.y, w.torque.z], dtype=np.float32)
        self._ft_received = True

    @property
    def is_ready(self) -> bool:
        if self.ft_sensor_on:
            return self._franka_received and self._gripper_received and self._ft_received
        return self._franka_received and self._gripper_received

    def get_obs(self, timeout: float = 5.0) -> dict:
        """Return a complete observation dict (raw numpy arrays, no encoding)."""
        deadline = time.time() + timeout
        while (self.rgb_images == [] or self.depth_images == []
               or self.inhand_image is None or self.inhand_depth is None):
            if time.time() > deadline:
                raise TimeoutError("Camera frames not received within timeout")
            time.sleep(0.005)

        rgb_list = [np.asarray(img) for img in self.rgb_images]
        depth_list = [np.asarray(img) for img in self.depth_images]
        inhand_rgb = np.asarray(self.inhand_image)
        inhand_depth = np.asarray(self.inhand_depth)

        ee_position = ee_quat_xyzw = joint_values = None
        if self.robot is not None:
            ee = self.robot.end_effector_pose
            ee_position = np.asarray(ee.position, dtype=np.float32)
            ee_quat_xyzw = ee.orientation.as_quat().astype(np.float32)
            joint_values = np.asarray(self.robot.joint_values, dtype=np.float32)

        gripper_value = None
        if self.gripper is not None and self.gripper.value is not None:
            gripper_value = float(self.gripper.value)

        return {
            "timestamp": time.time(),
            "rgb": rgb_list,
            "depth": depth_list,
            "inhand_rgb": inhand_rgb,
            "inhand_depth": inhand_depth,
            "joint_values": joint_values,
            "ee_position": ee_position,
            "ee_quat_xyzw": ee_quat_xyzw,
            "gripper_value": gripper_value,
            "gripper_state": self.gripper_joint_array,
            "ft_wrench": self.gripper_torque_array,
        }
# %%
DEFAULT_CONFIG_PATH = "/home/mingxi/mingxi_ws/handpi/diffusion_policy/robotool/robot_configs/camera_info.yaml"


def _summarize(name, arr):
    if arr is None:
        return f"  {name}: None"
    if isinstance(arr, np.ndarray):
        flat = arr.flatten()
        head = np.array2string(flat[:6], precision=3, suppress_small=True)
        return f"  {name}: shape={arr.shape} dtype={arr.dtype} head={head}"
    return f"  {name}: {arr}"


def test_pointcloud_manager(duration: float = 10.0,
                            config_path: str = DEFAULT_CONFIG_PATH):
    """Measure get_latest_rgbd rate for cam3 (cameras only)."""
    rclpy.init()
    manager = PointCloudManager(config_path, downsample=2)

    import threading
    threading.Thread(target=rclpy.spin, args=(manager,), daemon=True).start()

    update_times = []
    last_update_time = None
    start = time.time()
    test_cam = "cam3"
    print(f"Measuring get_latest_rgbd rate for {test_cam}...")
    while time.time() - start < duration:
        time.sleep(0.01)
        rgb, depth = manager.get_latest_rgbd(test_cam)
        if rgb is not None and depth is not None:
            now = time.time()
            if last_update_time is not None:
                update_times.append(now - last_update_time)
                if len(update_times) % 15 == 0:
                    rate = 1.0 / np.mean(update_times[-30:])
                    print(f"Rate: {rate:.2f} Hz | RGB: {rgb.shape} | Depth: {depth.shape}")
            last_update_time = now
        manager.clear_cache()

    if update_times:
        print(f"Average rate: {1.0/np.mean(update_times):.2f} Hz over {len(update_times)} updates")
    try:
        manager.destroy_node()
    except Exception:
        pass
    if rclpy.ok():
        rclpy.shutdown()


def test_observation_manager(duration: float = 10.0,
                             config_path: str = DEFAULT_CONFIG_PATH,
                             with_robot: bool = False,
                             ft_sensor_on: bool = False):
    """Validate ObservationManager: cameras + joint_states + gripper + (optional) FT and Robot/Gripper."""
    rclpy.init()

    robot = None
    gripper = None
    if with_robot:
        # Lazy imports — these pull in the full crisp_py control stack
        from crisp_py.gripper.gripper import Gripper, GripperConfig
        from crisp_py.robot import Robot
        from crisp_py.robot_config import FrankaConfig

        print("[test] Initializing Gripper...")
        gripper = Gripper(gripper_config=GripperConfig.from_yaml("./config/gripper_robotiq.yaml"))
        gripper.wait_until_ready()
        print("[test] Initializing Robot...")
        robot = Robot(namespace="", robot_config=FrankaConfig())
        robot.wait_until_ready()

    print("[test] Constructing ObservationManager...")
    manager = ObservationManager(
        config_path,
        robot=robot,
        gripper=gripper,
        ft_sensor_on=ft_sensor_on,
        downsample=2,
    )

    import threading
    threading.Thread(target=rclpy.spin, args=(manager,), daemon=True).start()

    print("[test] Waiting for joint_states + gripper_state (is_ready)...")
    deadline = time.time() + 15.0
    while not manager.is_ready:
        if time.time() > deadline:
            print("[test] WARNING: is_ready not reached in 15s — check that /joint_states and /gripper/gripper_state are publishing")
            break
        time.sleep(0.1)
    print(f"[test] is_ready={manager.is_ready}")

    expected_keys = {
        "timestamp", "rgb", "depth", "inhand_rgb", "inhand_depth",
        "joint_values", "ee_position", "ee_quat_xyzw",
        "gripper_value", "gripper_state", "ft_wrench",
    }
    expected_non_none = {"rgb", "depth", "inhand_rgb", "inhand_depth", "gripper_state"}
    if with_robot:
        expected_non_none |= {"joint_values", "ee_position", "ee_quat_xyzw", "gripper_value"}
    if ft_sensor_on:
        expected_non_none.add("ft_wrench")

    update_times = []
    last_t = None
    n_obs = 0
    start = time.time()
    print(f"[test] Polling get_obs() for {duration:.1f}s...")
    while time.time() - start < duration:
        try:
            obs = manager.get_obs(timeout=2.0)
        except TimeoutError as e:
            print(f"[test] {e}")
            manager.clear_cache()
            continue

        # Schema check on the first obs
        if n_obs == 0:
            missing = expected_keys - set(obs.keys())
            assert not missing, f"obs missing keys: {missing}"
            none_fields = {k for k in expected_non_none if obs.get(k) is None
                           or (isinstance(obs.get(k), list) and len(obs[k]) == 0)}
            if none_fields:
                print(f"[test] WARNING: expected non-None but got None for: {sorted(none_fields)}")
            print("[test] First obs:")
            print(f"  timestamp: {obs['timestamp']:.3f}")
            for i, (rgb, depth) in enumerate(zip(obs["rgb"], obs["depth"])):
                print(_summarize(f"rgb[{i}]", rgb))
                print(_summarize(f"depth[{i}]", depth))
            print(_summarize("inhand_rgb", obs["inhand_rgb"]))
            print(_summarize("inhand_depth", obs["inhand_depth"]))
            print(_summarize("joint_values", obs["joint_values"]))
            print(_summarize("ee_position", obs["ee_position"]))
            print(_summarize("ee_quat_xyzw", obs["ee_quat_xyzw"]))
            print(f"  gripper_value: {obs['gripper_value']}")
            print(_summarize("gripper_state", obs["gripper_state"]))
            print(_summarize("ft_wrench", obs["ft_wrench"]))

        n_obs += 1
        now = time.time()
        if last_t is not None:
            update_times.append(now - last_t)
            if len(update_times) % 15 == 0:
                rate = 1.0 / np.mean(update_times[-30:])
                jv = obs["joint_values"]
                ee = obs["ee_position"]
                print(f"[obs #{n_obs}] {rate:.2f} Hz | "
                      f"joint_values={'set' if jv is not None else 'None'} | "
                      f"ee={'set' if ee is not None else 'None'} | "
                      f"gripper_state={obs['gripper_state']}")
        last_t = now
        manager.clear_cache()

    print("\n=== Summary ===")
    print(f"Total observations: {n_obs}")
    if update_times:
        print(f"Average rate: {1.0/np.mean(update_times):.2f} Hz")
        print(f"Min/Max interval: {min(update_times):.3f}s / {max(update_times):.3f}s")
    else:
        print("No observations received!")

    if with_robot and robot is not None:
        try:
            robot.shutdown()
        except Exception:
            pass
    try:
        manager.destroy_node()
    except Exception:
        pass
    if rclpy.ok():
        rclpy.shutdown()


def main():
    import argparse
    parser = argparse.ArgumentParser(description="PointCloud / Observation manager tests")
    parser.add_argument("--mode", choices=["pcd", "obs"], default="obs",
                        help="pcd: test get_latest_rgbd. obs: test ObservationManager.get_obs (default)")
    parser.add_argument("--config", default=DEFAULT_CONFIG_PATH,
                        help="Path to camera_info.yaml")
    parser.add_argument("--duration", type=float, default=10.0)
    parser.add_argument("--with-robot", action="store_true",
                        help="Also instantiate crisp_py Robot+Gripper to populate joint_values/ee/gripper_value")
    parser.add_argument("--ft-sensor", action="store_true",
                        help="Require FT sensor messages before is_ready")
    args = parser.parse_args()

    if args.mode == "pcd":
        test_pointcloud_manager(duration=args.duration, config_path=args.config)
    else:
        test_observation_manager(duration=args.duration,
                                 config_path=args.config,
                                 with_robot=args.with_robot,
                                 ft_sensor_on=args.ft_sensor)


if __name__ == "__main__":
    main()
