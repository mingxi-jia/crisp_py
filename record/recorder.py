"""
Four-camera synchronized frame recorder for robot data collection.
Records RGB, depth images, gripper state, EEF pose, and joint states.
"""

import os
import sys
import queue
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from sensor_msgs.msg import Image, CameraInfo, JointState
from geometry_msgs.msg import PoseStamped, WrenchStamped
from std_msgs.msg import Float64MultiArray, Int32MultiArray
from cv_bridge import CvBridge, CvBridgeError
import message_filters
from pynput import keyboard
from scipy.spatial.transform import Rotation as R


@dataclass
class RecorderConfig:
    """Configuration for the frame recorder."""
    num_cameras: int = 4
    fps: int = 15
    # output_dir: str = './raw_datasets/episodes'
    # output_dir: str = '/mnt/c2b9de74-0cf1-492c-b46e-70d1bc9419fe/mingxi/XEMB/raw_datasets/episodes'
    output_dir: str = '/media/mingxi/daaata1/zilai_data/pick_coffee_pod_cog/play/'
    output_dir: str = '/media/mingxi/daaata1/duo_policy/data/raw_datasets/episodes'
    output_dir: str = '/mnt/c2b9de74-0cf1-492c-b46e-70d1bc9419fe/mingxi/XEMB/raw_datasets/episodes'
    queue_size: int = 20
    sync_slop: float = 0.05  # 50ms max time difference for sync (increased for 4 cameras)
    rgb_encoding: str = 'bgr8'
    depth_encoding: str = 'passthrough'
    save_format_rgb: str = 'png'
    save_format_depth: str = 'npy'
    # Depth visualization range (in mm)
    depth_min_mm: float = 200.0
    depth_max_mm: float = 2000.0
    # Keep recording briefly after gripper command / observed qpos changes.
    gripper_record_hold_sec: float = 0.5
    gripper_qpos_change_epsilon: float = 1e-4


@dataclass
class RecordingState:
    """Mutable state for ongoing recording."""
    is_recording: bool = False
    current_episode_path: Optional[str] = None
    dropped_bundle_count: int = 0
    gripper_states: list = field(default_factory=list)
    eef_poses: list = field(default_factory=list)
    joint_states: list = field(default_factory=list)
    intervention_states: list = field(default_factory=list)
    gripper_torques: list = field(default_factory=list)
    gripper_qpos: list = field(default_factory=list)
    last_frame_time: Optional[float] = None  # Timestamp of last recorded frame


class FrameRecorderNode(Node):
    def __init__(self):
        super().__init__('frame_recorder_node_sync3')

        self.config = RecorderConfig()
        self.state = RecordingState()
        self.bridge = CvBridge()

        # Thread synchronization
        self.recording_lock = threading.Lock()
        self.data_lock = threading.Lock()  # For gripper, eef_pose, joint_state, teleop
        self.writer_thread: Optional[threading.Thread] = None
        self.writer_running = threading.Event()
        self.frame_queue: queue.Queue = queue.Queue(maxsize=self.config.queue_size)

        # Camera info tracking
        self.camera_info = [None] * self.config.num_cameras  # (width, height) tuples
        self.camera_info_events = [threading.Event() for _ in range(self.config.num_cameras)]
        self.all_cameras_ready = threading.Event()

        # Current sensor data (protected by data_lock)
        self.current_gripper_state = 0
        self.current_gripper_qpos: Optional[float] = None
        self.current_eef_pose: Optional[np.ndarray] = None
        self.current_joint_state: Optional[np.ndarray] = None
        self.current_intervention_state = 0  # 0=idle, 1=policy, 2=intervention
        self.current_gripper_torque: Optional[np.ndarray] = None
        self.has_teleop_command = False
        self.gripper_record_until: float = 0.0

        # Build topic lists
        self.topics = self._build_topic_lists()

        # Setup
        os.makedirs(self.config.output_dir, exist_ok=True)
        self._setup_subscriptions()
        self._start_keyboard_listener()

        # Auto-start recording when cameras are ready
        threading.Thread(target=self._wait_and_start_recording, daemon=True).start()

        self.get_logger().info(f"Recorder initialized: {self.config.num_cameras} cameras, output={self.config.output_dir}")

    def _build_topic_lists(self) -> dict:
        """Build topic name lists for all cameras."""
        n = self.config.num_cameras
        topics = {
            'rgb': [f'/cam{i+1}/color/image_raw' for i in range(n - 1)],
            'depth': [f'/cam{i+1}/aligned_depth_to_color/image_raw' for i in range(n - 1)],
            'camera_info': [f'/cam{i+1}/color/camera_info' for i in range(n - 1)],
        }
        # Wrist camera (cam4) has different topic names
        topics['rgb'].append('/cam4/color/image_rect_raw')
        topics['depth'].append('/cam4/aligned_depth_to_color/image_raw')
        topics['camera_info'].append('/cam4/color/camera_info')
        return topics

    def _setup_subscriptions(self):
        """Setup all ROS subscriptions."""
        qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=self.config.queue_size,
            durability=DurabilityPolicy.VOLATILE
        )

        # Camera info subscriptions
        self.camera_info_subs = []
        for i, topic in enumerate(self.topics['camera_info']):
            sub = self.create_subscription(
                CameraInfo, topic,
                lambda msg, idx=i: self._on_camera_info(msg, idx), 10
            )
            self.camera_info_subs.append(sub)

        # Robot state subscriptions
        self.create_subscription(PoseStamped, '/current_pose', self._on_eef_pose, 10)
        self.create_subscription(JointState, '/joint_states', self._on_joint_states, 10)
        self.create_subscription(Float64MultiArray, '/gripper/gripper_position_controller/commands', self._on_gripper_command, 10)
        self.create_subscription(JointState, '/gripper/gripper_state', self._on_gripper_state, 10)
        self.create_subscription(WrenchStamped, '/ft/robotiq_force_torque_sensor_broadcaster/wrench', self._on_gripper_torque, 10)
        self.create_subscription(Int32MultiArray, '/teleop/signals', self._on_teleop_signal, 10)

        # Synchronized image subscriptions
        rgb_subs = [message_filters.Subscriber(self, Image, t, qos_profile=qos) for t in self.topics['rgb']]
        depth_subs = [message_filters.Subscriber(self, Image, t, qos_profile=qos) for t in self.topics['depth']]

        # Interleave: rgb1, depth1, rgb2, depth2, ...
        all_subs = []
        for rgb, depth in zip(rgb_subs, depth_subs):
            all_subs.extend([rgb, depth])

        self.time_sync = message_filters.ApproximateTimeSynchronizer(
            all_subs, queue_size=self.config.queue_size, slop=self.config.sync_slop
        )
        self.time_sync.registerCallback(self._on_synchronized_images)

    # --- Callbacks ---

    def _on_camera_info(self, msg: CameraInfo, camera_idx: int):
        """Handle camera info message."""
        if self.camera_info_events[camera_idx].is_set():
            return

        self.camera_info[camera_idx] = (msg.width, msg.height)
        self.camera_info_events[camera_idx].set()
        self.get_logger().info(f"Camera {camera_idx+1} info: {msg.width}x{msg.height}")

        # Destroy subscription once info is received
        if self.camera_info_subs[camera_idx]:
            self.destroy_subscription(self.camera_info_subs[camera_idx])
            self.camera_info_subs[camera_idx] = None

    def _on_eef_pose(self, msg: PoseStamped):
        """Handle end-effector pose message."""
        pose = np.array([
            msg.pose.position.x, msg.pose.position.y, msg.pose.position.z,
            msg.pose.orientation.x, msg.pose.orientation.y,
            msg.pose.orientation.z, msg.pose.orientation.w
        ], dtype=np.float32)
        # this is the original gripper tcp pose and we will calculate the finray gripper pose later

        with self.data_lock:
            self.current_eef_pose = pose

    def _on_joint_states(self, msg: JointState):
        """Handle joint states message."""
        # Sort by joint name for consistent ordering
        sorted_positions = [p for _, p in sorted(zip(msg.name, msg.position))]
        with self.data_lock:
            self.current_joint_state = np.array(sorted_positions, dtype=np.float32)

    def _on_gripper_command(self, msg: Float64MultiArray):
        """Handle gripper command message."""
        gripper_state = int(msg.data[0]) if msg.data else 0
        with self.data_lock:
            if gripper_state != self.current_gripper_state:
                self.gripper_record_until = time.time() + self.config.gripper_record_hold_sec
            self.current_gripper_state = gripper_state

    def _on_gripper_state(self, msg: JointState):
        """Handle gripper state message (raw qpos from /gripper/gripper_state)."""
        for i, name in enumerate(msg.name):
            if name == 'gripper_joint':
                with self.data_lock:
                    new_qpos = float(msg.position[i])
                    if (
                        self.current_gripper_qpos is None or
                        abs(new_qpos - self.current_gripper_qpos) >= self.config.gripper_qpos_change_epsilon
                    ):
                        self.gripper_record_until = time.time() + self.config.gripper_record_hold_sec
                    self.current_gripper_qpos = new_qpos
                break

    def _on_gripper_torque(self, msg: WrenchStamped):
        """Handle gripper torque message."""
        gripper_torque = np.array([
            msg.wrench.force.x,
            msg.wrench.force.y,
            msg.wrench.force.z,
            msg.wrench.torque.x,
            msg.wrench.torque.y,
            msg.wrench.torque.z
        ], dtype=np.float32)
        with self.data_lock:
            self.current_gripper_torque = gripper_torque

    def _on_teleop_signal(self, msg: Int32MultiArray):
        """Handle teleop signal message.

        Message format: [state, dx, dy, dz, droll, dpitch, dyaw, gripper, reset]
        where state: 0=idle, 1=policy action, 2=intervention action
        """
        if len(msg.data) < 9:
            return

        # Extract intervention state from first element
        intervention_state = msg.data[0]

        # Skip if reset is pressed (index 8)
        if msg.data[8]:
            return

        # Record if state is 1 (policy) or 2 (intervention)
        if intervention_state in [1, 2]:
            with self.data_lock:
                self.current_intervention_state = intervention_state
                self.has_teleop_command = True
        else:
            with self.data_lock:
                self.has_teleop_command = False

        print("self.current_intervention_state: ", self.current_intervention_state)

    def _on_synchronized_images(self, *msgs):
        """Handle synchronized image bundle from all cameras."""
        # print(f"[SYNC CALLBACK] Received {len(msgs)} messages")
        # print(f"[SYNC CALLBACK] is_recording={self.state.is_recording}")
        if not self.state.is_recording:
            return

        expected_count = self.config.num_cameras * 2
        print(f"Expected {expected_count} messages, got {len(msgs)}")
        if len(msgs) != expected_count:
            self.get_logger().error(f"Expected {expected_count} messages, got {len(msgs)}")
            return

        # Get current state snapshot
        with self.data_lock:
            has_teleop = self.has_teleop_command
            gripper = self.current_gripper_state
            eef_pose = self.current_eef_pose.copy() if self.current_eef_pose is not None else None
            joint_state = self.current_joint_state.copy() if self.current_joint_state is not None else None
            intervention_state = self.current_intervention_state
            gripper_torque = self.current_gripper_torque.copy() if self.current_gripper_torque is not None else None
            gripper_qpos = self.current_gripper_qpos
            gripper_record_until = self.gripper_record_until

        print(f"Intervention state: {self.current_intervention_state}")

        current_time = msgs[0].header.stamp.sec + msgs[0].header.stamp.nanosec * 1e-9
        should_record = has_teleop or (time.time() <= gripper_record_until)

        # Record while teleop is active, or during the short gripper hold window.
        if not should_record:
            print('idle, not started or inferencing')
            return

        # FPS enforcement: Check if enough time has elapsed since last frame
        min_interval = 1.0 / self.config.fps  # e.g., 0.2 seconds for 5 FPS

        if self.state.last_frame_time is not None:
            time_since_last = current_time - self.state.last_frame_time
            if time_since_last < min_interval:
                # Too soon, skip this frame
                return

        # Update last frame time
        self.state.last_frame_time = current_time

        self.get_logger().info("Synchronized bundle received, recording frame.")

        try:
            self.frame_queue.put_nowait((msgs, gripper, eef_pose, joint_state, intervention_state, gripper_torque, gripper_qpos))
        except queue.Full:
            self.state.dropped_bundle_count += 1
            if self.state.dropped_bundle_count % 100 == 0:
                self.get_logger().warn(f"Queue full, dropped {self.state.dropped_bundle_count} bundles")

        # Reset this to False to avoid image flooding 
        self.has_teleop_command = False

    # --- Recording Control ---

    def _wait_and_start_recording(self):
        """Wait for all camera info, then auto-start recording."""
        for event in self.camera_info_events:
            event.wait()
        self.all_cameras_ready.set()
        self.get_logger().info("All cameras ready. Auto-starting recording...")
        self._start_recording()

    def _start_recording(self):
        """Start a new recording episode."""
        if not self.all_cameras_ready.wait(timeout=3.0):
            self.get_logger().warn("Cannot start: cameras not ready")
            return

        with self.recording_lock:
            if self.state.is_recording:
                self.get_logger().warn("Already recording")
                return

            # Cleanup any existing writer thread
            if self.writer_thread and self.writer_thread.is_alive():
                self._stop_writer_thread()

            # Create episode directory
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
            self.state.current_episode_path = os.path.join(
                self.config.output_dir, f"episode_{timestamp}"
            )
            os.makedirs(self.state.current_episode_path, exist_ok=True)

            # Reset state
            self.state.dropped_bundle_count = 0
            self.state.gripper_states = []
            self.state.eef_poses = []
            self.state.joint_states = []
            self.state.intervention_states = []
            self.state.gripper_qpos = []
            self.state.last_frame_time = None  # Reset FPS timer
            self._clear_queue()

            with self.data_lock:
                self.has_teleop_command = False
                self.current_intervention_state = 0
                self.gripper_record_until = 0.0

            # Start writer thread
            self.writer_running.set()
            self.writer_thread = threading.Thread(target=self._writer_loop, daemon=True)
            self.writer_thread.start()
            self.state.is_recording = True

            self.get_logger().info(f"Recording started: {self.state.current_episode_path}")

    def _stop_recording(self):
        """Stop the current recording."""
        if not self.state.is_recording:
            self.get_logger().info("Not recording")
            return

        self.get_logger().info("Stopping recording...")
        self.state.is_recording = False
        self._stop_writer_thread()

        with self.recording_lock:
            if self.state.current_episode_path:
                self.get_logger().info(f"Episode saved: {self.state.current_episode_path}")
                if self.state.dropped_bundle_count > 0:
                    self.get_logger().warn(f"Dropped bundles: {self.state.dropped_bundle_count}")

    def _stop_writer_thread(self):
        """Stop the writer thread and wait for it to finish."""
        if not self.writer_thread or not self.writer_thread.is_alive():
            return

        self.writer_running.clear()
        self.frame_queue.join()
        self.writer_thread.join(timeout=25.0)

        if self.writer_thread.is_alive():
            self.get_logger().error("Writer thread did not stop cleanly")

        self.writer_thread = None
        self._clear_queue()

    def _clear_queue(self):
        """Clear the frame queue."""
        count = 0
        while not self.frame_queue.empty():
            try:
                self.frame_queue.get_nowait()
                self.frame_queue.task_done()
                count += 1
            except queue.Empty:
                break
        if count:
            self.get_logger().info(f"Cleared {count} queued bundles")

    # --- Writer Thread ---

    def _writer_loop(self):
        """Main loop for the writer thread."""
        episode_path = self.state.current_episode_path
        if not episode_path:
            self.get_logger().error("Writer started without episode path")
            return

        # Create directory structure
        cam_paths = self._create_episode_dirs(episode_path)
        if not cam_paths:
            return

        bundle_count = 0
        frame_counts = [[0, 0] for _ in range(self.config.num_cameras)]

        while self.writer_running.is_set() or not self.frame_queue.empty():
            try:
                bundle = self.frame_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            msgs, gripper, eef_pose, joint_state, intervention_state, gripper_torque, gripper_qpos = bundle

            # Track state for this frame
            self.state.gripper_states.append(1 - gripper)  # Invert gripper state
            self.state.eef_poses.append(eef_pose)
            self.state.joint_states.append(joint_state)
            # Convert intervention state: 2 -> 1 (intervention), 1 -> 0 (policy)
            self.state.intervention_states.append(1 if intervention_state == 2 else 0)
            self.state.gripper_torques.append(gripper_torque)
            self.state.gripper_qpos.append(gripper_qpos)


            # Save images for each camera
            for cam_idx in range(self.config.num_cameras):
                rgb_msg = msgs[cam_idx * 2]
                depth_msg = msgs[cam_idx * 2 + 1]

                if self._save_rgb_image(rgb_msg, cam_paths[cam_idx]['rgb'], cam_idx):
                    frame_counts[cam_idx][0] += 1
                if self._save_depth_image(depth_msg, cam_paths[cam_idx], cam_idx):
                    frame_counts[cam_idx][1] += 1

            self.frame_queue.task_done()
            bundle_count += 1

        # Save state data
        self._save_state_data(cam_paths['state'])

        # Log summary
        self.get_logger().info(f"Writer finished: {bundle_count} bundles")
        for i, (rgb, depth) in enumerate(frame_counts):
            self.get_logger().info(f"  Cam {i+1}: {rgb} RGB, {depth} depth")

    def _create_episode_dirs(self, episode_path: str) -> Optional[dict]:
        """Create directory structure for episode."""
        try:
            cam_paths = {}
            for i in range(self.config.num_cameras):
                cam_dir = os.path.join(episode_path, f"cam{i+1}")
                cam_paths[i] = {
                    'rgb': os.path.join(cam_dir, 'rgb'),
                    'depth': os.path.join(cam_dir, 'depth'),
                    'depth_vis': os.path.join(cam_dir, 'depth_vis'),
                }
                for path in cam_paths[i].values():
                    os.makedirs(path, exist_ok=True)

            cam_paths['state'] = os.path.join(episode_path, 'state')
            os.makedirs(cam_paths['state'], exist_ok=True)
            return cam_paths
        except OSError as e:
            self.get_logger().error(f"Failed to create directories: {e}")
            return None

    def _save_rgb_image(self, msg: Image, save_path: str, cam_idx: int) -> bool:
        """Save RGB image to disk."""
        if not msg or msg.height == 0 or msg.width == 0:
            return False

        try:
            # Handle BGRA to BGR conversion if needed
            if msg.encoding.lower() == 'bgra8' and self.config.rgb_encoding.lower() == 'bgr8':
                cv_image = cv2.cvtColor(
                    self.bridge.imgmsg_to_cv2(msg),
                    cv2.COLOR_BGRA2BGR
                )
            else:
                cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding=self.config.rgb_encoding)

            if cv_image is None or cv_image.size == 0:
                return False

            filename = self._timestamp_filename(msg, self.config.save_format_rgb)
            cv2.imwrite(os.path.join(save_path, filename), cv_image)
            return True
        except (CvBridgeError, Exception) as e:
            self.get_logger().error(f"Cam {cam_idx+1} RGB error: {e}")
            return False

    def _save_depth_image(self, msg: Image, cam_paths: dict, cam_idx: int) -> bool:
        """Save depth image (npy) and visualization (png) to disk."""
        if not msg or msg.height == 0 or msg.width == 0:
            return False

        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding=self.config.depth_encoding)
            if cv_image is None or cv_image.size == 0:
                return False

            # Save raw depth as npy
            filename_npy = self._timestamp_filename(msg, self.config.save_format_depth)
            np.save(os.path.join(cam_paths['depth'], filename_npy), cv_image)

            # Save visualization
            filename_vis = self._timestamp_filename(msg, 'png')
            depth_vis = self._create_depth_visualization(cv_image)
            cv2.imwrite(os.path.join(cam_paths['depth_vis'], filename_vis), depth_vis)

            return True
        except (CvBridgeError, Exception) as e:
            self.get_logger().error(f"Cam {cam_idx+1} depth error: {e}")
            return False

    def _create_depth_visualization(self, depth: np.ndarray) -> np.ndarray:
        """Create visualization image from depth data."""
        vis = np.copy(depth).astype(float)
        vis[np.isnan(vis)] = 0.0
        vis = np.clip(vis, self.config.depth_min_mm, self.config.depth_max_mm)
        vis = (vis - self.config.depth_min_mm) / (self.config.depth_max_mm - self.config.depth_min_mm) * 255.0
        return vis.astype(np.uint8)

    def _timestamp_filename(self, msg: Image, extension: str) -> str:
        """Generate filename from message timestamp."""
        sec = msg.header.stamp.sec
        nanosec = msg.header.stamp.nanosec
        return f"{sec:010d}_{nanosec:09d}.{extension}"

    def _save_state_data(self, state_path: str):
        """Save gripper states, EEF poses, joint states, and intervention states."""
        self._save_numpy_array(
            self.state.gripper_states, state_path, 'grasp.npy', np.int32, "gripper states"
        )
        self._save_numpy_array(
            self.state.eef_poses, state_path, 'pose_wrt_world.npy', np.float32, "EEF poses"
        )
        self._save_numpy_array(
            self.state.joint_states, state_path, 'joint_states.npy', np.float32, "joint states"
        )
        self._save_numpy_array(
            self.state.intervention_states, state_path, 'intervention.npy', np.int32, "intervention states"
        )
        self._save_numpy_array(
            self.state.gripper_torques, state_path, 'ft_sensor.npy', np.float32, "gripper torques"
        )
        self._save_numpy_array(
            self.state.gripper_qpos, state_path, 'gripper_qpos.npy', np.float32, "gripper qpos"
        )

    def _save_numpy_array(self, data_list: list, path: str, filename: str, dtype, name: str):
        """Save a list of data as numpy array."""
        valid_data = [d for d in data_list if d is not None]
        if not valid_data:
            self.get_logger().warn(f"No {name} to save")
            return

        try:
            arr = np.array(valid_data, dtype=dtype)
            filepath = os.path.join(path, filename)
            np.save(filepath, arr)
            self.get_logger().info(f"Saved {len(valid_data)} {name} to {filepath}")
        except Exception as e:
            self.get_logger().error(f"Error saving {name}: {e}")

    # --- Keyboard Control ---

    def _start_keyboard_listener(self):
        """Start keyboard listener for recording control."""
        self.keyboard_listener = keyboard.Listener(on_press=self._on_key_press)
        self.keyboard_listener.start()
        self.get_logger().info("Press 'r' to save current episode and start next one")

    def _on_key_press(self, key):
        """Handle keyboard press events."""
        try:
            if hasattr(key, 'char') and key.char == 'r':
                self._reset_episode()
            elif hasattr(key, 'char') and key.char == 'f':
                self.get_logger().info("Bad Quality Data Marked, reset episode.")

                # Create bad.txt file in state directory
                if self.state.current_episode_path:
                    state_dir = os.path.join(self.state.current_episode_path, 'state')
                    os.makedirs(state_dir, exist_ok=True)
                    bad_file_path = os.path.join(state_dir, 'bad.txt')
                    with open(bad_file_path, 'w') as f:
                        f.write('Bad quality data marked by user\n')
                    self.get_logger().info(f"Created bad.txt marker at {bad_file_path}")

                self._reset_episode()

        except AttributeError:
            pass

    def _reset_episode(self):
        """Save current episode and start a new one. Delete if fewer than 16 frames."""
        episode_path = self.state.current_episode_path
        if self.state.is_recording:
            self._stop_recording()

        # Check and delete episode if fewer than 16 frames
        if episode_path:
            self._cleanup_short_episode(episode_path, min_frames=16)

        self._start_recording()

    def _cleanup_short_episode(self, episode_path: str, min_frames: int):
        """Delete episode folder if it has fewer than min_frames."""
        try:
            # Count frames by checking gripper states length (one per frame)
            state_file = os.path.join(episode_path, 'state', 'grasp.npy')
            if os.path.exists(state_file):
                frame_count = len(np.load(state_file))
            else:
                # Fallback: count RGB images in cam1
                rgb_path = os.path.join(episode_path, 'cam1', 'rgb')
                if os.path.exists(rgb_path):
                    frame_count = len([f for f in os.listdir(rgb_path) if f.endswith('.png')])
                else:
                    frame_count = 0

            if frame_count < min_frames:
                import shutil
                shutil.rmtree(episode_path)
                self.get_logger().warn(f"Deleted episode with only {frame_count} frames (min: {min_frames}): {episode_path}")
            else:
                self.get_logger().info(f"Episode kept: {frame_count} frames")
        except Exception as e:
            self.get_logger().error(f"Error checking/cleaning episode: {e}")

    # --- Cleanup ---

    def destroy_node(self):
        """Clean shutdown of the node."""
        self.get_logger().info("Shutting down...")

        if hasattr(self, 'keyboard_listener') and self.keyboard_listener:
            self.keyboard_listener.stop()

        self.state.is_recording = False
        self._stop_writer_thread()

        if hasattr(self, 'time_sync'):
            self.time_sync.callbacks = {}

        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = FrameRecorderNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        print("\nShutting down...")
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
    finally:
        if node:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
