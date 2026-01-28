from abc import ABC, abstractmethod
import time
import numpy as np
from pathlib import Path
import threading
import shutil
from datetime import datetime
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt
from matplotlib import cm, ticker

from diff_eval_utils.diffusion_transforms import (
    get_pose_from_robot,
    convert_action_from_fingertip_to_gripper,
    franka_obs_to_diff_obs
)
from diff_eval_utils.diffusion_visualization import visualize_pcd_and_actions
from diff_eval_utils.diffusion_constants import GRIPPER_NORM_CONST
from std_msgs.msg import Int32MultiArray



class RobotController(ABC):
    """Abstract base class for robot controllers."""

    def __init__(self, robot, gripper, obs_manager, joint_state_subscriber,
                 policy_client, rotation_transformer,
                 config, ctrl_freq=10.0):
        """Initialize the controller.

        Args:
            robot: Robot instance
            gripper: Gripper instance
            obs_manager: PointCloudManager instance
            joint_state_subscriber: JointStateSubscriber instance
            policy_client: PolicyClient instance
            pcd_client: PcdProcessingClient instance
            rotation_transformer: RotationTransformer for 6D rotations
            config: DPEvalConfig instance
            ctrl_freq: Control frequency in Hz
        """
        self.robot = robot
        self.gripper = gripper
        self.obs_manager = obs_manager
        self.joint_state_subscriber = joint_state_subscriber
        self.policy_client = policy_client
        # self.pcd_client = pcd_client
        self.rotation_transformer = rotation_transformer
        self.config = config
        self.ctrl_freq = ctrl_freq

        # Common state
        self.target_pose = robot.end_effector_pose.copy()
        self.arm_rate = robot.node.create_rate(ctrl_freq)
        self.gripper_rate = gripper.node.create_rate(ctrl_freq)
        self.prev_grasp_value = 0.0

        # Unified observation buffer (stores all observation data per timestep)
        self.obs_buffer = []
        self._buffer_lock = threading.Lock()

        # Background buffer update thread
        self._buffer_update_thread = None
        self._buffer_update_running = False
        self._buffer_update_requested = threading.Event()
        self._buffer_update_complete = threading.Event()

        self.latest_obs_timestamp = None

        # Initial synchronous buffer update
        self._update_buffer_sync()

    def _start_background_buffer_thread(self):
        """Start the background buffer update thread."""
        if self._buffer_update_thread is not None and self._buffer_update_thread.is_alive():
            return

        self._buffer_update_running = True
        self._buffer_update_thread = threading.Thread(
            target=self._background_buffer_worker,
            daemon=True
        )
        self._buffer_update_thread.start()

    def _stop_background_buffer_thread(self):
        """Stop the background buffer update thread."""
        self._buffer_update_running = False
        self._buffer_update_requested.set()  # Wake up the thread to exit
        if self._buffer_update_thread is not None:
            self._buffer_update_thread.join(timeout=1.0)

    def _background_buffer_worker(self):
        """Background worker that updates buffer when requested."""
        while self._buffer_update_running:
            # Wait for update request
            self._buffer_update_requested.wait()

            if not self._buffer_update_running:
                break

            self._buffer_update_requested.clear()

            # Perform the buffer update
            self._update_buffer_sync()

            # Signal that update is complete
            self._buffer_update_complete.set()

    def _update_buffer_sync(self):
        """Synchronously update the observation buffer (called by background thread)."""
        joint_state = self.joint_state_subscriber.joint_values
        gripper_state = self.prev_grasp_value

        eef_pose = get_pose_from_robot(self.robot.end_effector_pose)

        # Capture current observation snapshot
        obs_snapshot = {
            'eef_pos': eef_pose[:3].astype(np.float32),
            'eef_quat': eef_pose[3:].astype(np.float32),
            'gripper_qpos': np.array([gripper_state, gripper_state], dtype=np.float32),
            'joint_pos': joint_state.astype(np.float32),
        }

        # Capture images based on policy type
        cam3_rgb, _ = self.obs_manager.get_latest_rgbd('cam3')
        cam4_rgb, cam4_depth = self.obs_manager.get_latest_rgbd('cam4')
        obs_snapshot['cam4_rgb'] = cam4_rgb.copy()
        obs_snapshot['cam4_depth'] = cam4_depth.copy()
        obs_snapshot['cam3_rgb'] = cam3_rgb.copy()
        obs_snapshot['pcd'] = self.obs_manager.get_latest_pointcloud()
        update_end_time = time.time()
        obs_snapshot['timestamp'] = update_end_time
        # Thread-safe buffer update
        with self._buffer_lock:
            self.obs_buffer.append(obs_snapshot)

            # Initialize buffer with duplicate if needed (for policies requiring 2 frames)
            while len(self.obs_buffer) < 2:
                self.obs_buffer.insert(0, obs_snapshot.copy())

    def _update_buffer(self):
        """Request a buffer update (non-blocking, runs in background thread)."""
        # Make sure background thread is running
        if not self._buffer_update_running:
            self._start_background_buffer_thread()

        # Clear completion flag and request update
        self._buffer_update_complete.clear()
        self._buffer_update_requested.set()

    def _wait_for_buffer_update(self, timeout=1.0):
        """Wait for the background buffer update to complete.

        Args:
            timeout: Maximum time to wait in seconds

        Returns:
            bool: True if update completed, False if timeout
        """
        return self._buffer_update_complete.wait(timeout=timeout)

    @abstractmethod
    def run(self, n_steps: int):
        """Execute the controller for n_steps.

        Args:
            n_steps: Number of control steps to execute
        """
        pass

    def _get_observation(self):
        # Keep buffer size limited
        with self._buffer_lock:
            if len(self.obs_buffer) > 20:
                self.obs_buffer.pop(0)

            # Make a copy of buffer for processing
            buffer_copy = [obs.copy() for obs in self.obs_buffer]
        
        latest_timestamp = buffer_copy[-1]['timestamp']
        obs_dict = franka_obs_to_diff_obs(
            buffer_copy,
            img_policy=self.config.img_policy,
            visualize=self.config.visualize
        )
        self.latest_obs_timestamp = latest_timestamp
        return obs_dict

    def _get_inhand_rgb(self):
        return self.obs_manager.get_latest_rgbd('cam4')[0]

    def _execute_action(self, action, move_to=False):
        """Execute a single action.

        Args:
            action: 10-element action array [x, y, z, rot6d(6), grasp(1)]
        """
        action, gripper_pose = convert_action_from_fingertip_to_gripper(action, self.rotation_transformer)

        # Collect debug data if enabled
        if hasattr(self, 'debug_data_collection') and self.debug_data_collection is not None:
            timestamp_ms = int(time.time() * 1000)

            # Save executed action (before transformations)
            self.debug_data_collection['executed_actions'].append(
                (timestamp_ms, action.copy())
            )

            # Save EEF pose
            current_pose = self.robot.end_effector_pose
            pose_dict = {
                'position': current_pose.position.copy(),
                'orientation_quat': current_pose.orientation.as_quat().copy(),
                'orientation_matrix': current_pose.orientation.as_matrix().copy()
            }
            self.debug_data_collection['eef_poses'].append(
                (timestamp_ms, pose_dict)
            )

            # Save actual EEF position
            self.debug_data_collection['actual_eef_pos'].append(
                (timestamp_ms, current_pose.position.copy())
            )

        # Safety check: verify pose change is reasonable
        new_position = action[:3]
        print(f"obs & action timestamp {self.latest_obs_timestamp}\t{time.time()}")
        self.target_pose.position = new_position
        # self.target_pose.orientation = R.from_euler('XYZ', [np.pi, 0, 0])
        # gripper_pose=R.from_euler('XYZ', [np.pi, 0, 0])
        print(f"Moving to position: {new_position}, orientation (euler): {gripper_pose.as_euler('XYZ')}")
        self.target_pose.orientation = gripper_pose
        if not move_to:
            self.robot.set_target(pose=self.target_pose)
            self.arm_rate.sleep()
        else:
            self.robot.move_to(pose=self.target_pose, speed=0.15)

        grasp_value = np.round(np.clip(action[-1], 0, 1))
        if grasp_value != self.prev_grasp_value:
            self.gripper.set_target(1 - grasp_value)
            self.gripper_rate.sleep()
            time.sleep(1.0)  # Wait for gripper (Franka driver limitation)
        self.prev_grasp_value = grasp_value

        # Trigger async buffer update (non-blocking)
        self._update_buffer()

    def _switch_to_impedance_controller(self):
        self.robot.controller_switcher_client.switch_controller("cartesian_impedance_controller")
        self.robot.cartesian_controller_parameters_client.load_param_config(
            file_path="config/control/spacemouse_cartesian_impedance.yaml"
        )

    def cleanup(self):
        """Clean up resources. Call this when done with the controller."""
        self.gripper.set_target(1.0)  # Open gripper
        self._stop_background_buffer_thread()
