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
import copy
from scipy.spatial.transform import Slerp


from diff_eval_utils.diffusion_transforms import (
    get_pose_from_robot,
    ten_d_action_to_pose,
    convert_action_from_fingertip_to_gripper,
    franka_obs_to_diff_obs,
    ten_d_action_to_pose_batch,
    convert_action_from_fingertip_to_gripper_batch
)
from diff_eval_utils.diffusion_visualization import visualize_pcd_and_actions
from diff_eval_utils.diffusion_constants import GRIPPER_NORM_CONST
from std_msgs.msg import Int32MultiArray
from crisp_py.robot import Pose


class RobotController(ABC):
    """Abstract base class for robot controllers."""

    def __init__(self, robot, gripper, obs_manager, joint_state_subscriber,
                 policy_client, config, ctrl_freq=10.0, ik_client=None):
        """Initialize the controller.

        Args:
            robot: Robot instance
            gripper: Gripper instance
            obs_manager: PointCloudManager instance
            joint_state_subscriber: JointStateSubscriber instance
            policy_client: PolicyClient instance
            config: DPEvalConfig instance
            ctrl_freq: Control frequency in Hz
            ik_client: PinkIKClient instance (required when ctrl_space='joint')
        """
        self.robot = robot
        self.gripper = gripper
        self.obs_manager = obs_manager
        self.joint_state_subscriber = joint_state_subscriber
        self.policy_client = policy_client
        self.ik_client = ik_client
        self.config = config
        self.control_space = config.ctrl_space  # 'joint' or 'cartesian'
        if self.control_space not in ['joint', 'cartesian']:
            raise ValueError(f"Invalid ctrl_space: {self.control_space}")
        if self.control_space == 'joint':
            self.ctrl_freq = config.joint_ctrl_freq
            print(self.ctrl_freq)
        else:
            self.ctrl_freq = config.ctrl_freq
        print(f"Current Control Frequency: {self.ctrl_freq}")
        self.n_interpolation = config.n_interpolation

        if self.control_space == 'joint' and self.ik_client is None:
            raise ValueError("ik_client is required when ctrl_space='joint'")

        # Common state
        self.target_pose = robot.end_effector_pose.copy()
        self.arm_rate = robot.node.create_rate(self.ctrl_freq)
        self.gripper_rate = gripper.node.create_rate(self.ctrl_freq)
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

        self._last_joint_target = None
        self._joint_exec_time = None
        self._joint_exec_time_tolerance = config.joint_exec_time_tolerance

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
        self.obs_manager.clear_cache()
        while not self.joint_state_subscriber.is_ready:
            time.sleep(0.01)
        joint_state = self.joint_state_subscriber.joint_values
        gripper_state = self.joint_state_subscriber.gripper_state[0]
        # gripper_state = 0 
        # print(f"gripper_state {gripper_state}")

        obs_snapshot = {}
        # Capture images based on policy type
        cam3_rgb, _ = self.obs_manager.get_latest_rgbd('cam3')
        cam4_rgb, cam4_depth = self.obs_manager.get_latest_rgbd('cam4')
        obs_snapshot['cam4_rgb'] = cam4_rgb.copy()
        obs_snapshot['cam4_depth'] = cam4_depth.copy()
        obs_snapshot['cam3_rgb'] = cam3_rgb.copy()
        obs_snapshot['pcd'] = self.obs_manager.get_latest_pointcloud()
        
        eef_pose = get_pose_from_robot(copy.deepcopy(self.robot.end_effector_pose))
        # Capture current observation snapshot
        obs_snapshot = {
            'eef_pos_raw': copy.deepcopy(self.robot.end_effector_pose.position.astype(np.float32)),
            'eef_pos': eef_pose[:3].astype(np.float32),
            'eef_quat': eef_pose[3:].astype(np.float32),
            'gripper_qpos': np.array([gripper_state, gripper_state], dtype=np.float32),
            'joint_pos': joint_state.astype(np.float32),
            **obs_snapshot
        }

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
        print(f"eef_pos_raw: {buffer_copy[-1]['eef_pos_raw']}")
        obs_dict = franka_obs_to_diff_obs(
            buffer_copy,
            img_policy=self.config.img_policy,
            visualize=self.config.visualize
        )
        self.latest_obs_timestamp = latest_timestamp
        return obs_dict

    def _get_inhand_rgb(self):
        return self.obs_manager.get_latest_rgbd('cam4')[0]
    
    def _post_process_action(self, actions):
        # t_start = time.time()
        poses, grasps = ten_d_action_to_pose_batch(actions)
        gripper_positions, gripper_rotations = convert_action_from_fingertip_to_gripper_batch(poses)
        actions_ret = []
        for idx in range(len(grasps)):
            actions_ret.append((gripper_positions[idx], gripper_rotations[idx], grasps[idx]))
        # print(f"post process took {(time.time() - t_start) * 1000} ms")
        return actions_ret
    
    def _execute_gripper_action(self, grasp_value):
        grasp_value = np.round(np.clip(grasp_value, 0, 1))
        if grasp_value != self.prev_grasp_value:
            self.gripper.set_target(1 - grasp_value)
            self.gripper_rate.sleep()
            # time.sleep(1.0)  # Wait for gripper (Franka driver limitation)
        self.prev_grasp_value = grasp_value
    
    def _execute_cartesian_action(self, action, gripper_pose, move_to=False):
        assert len(action) >= 3, "Action must have at least 3 elements for position"
        new_position = action[:3]
        current_position = copy.deepcopy(self.robot.end_effector_pose.position)
        print(f"{new_position[0]:.3f}, {new_position[1]:.3f}, {new_position[2]:.3f}\t{current_position[0]:.3f}, {current_position[1]:.3f}, {current_position[2]:.3f}")
        current_orientation = copy.deepcopy(self.robot.end_effector_pose.orientation)

        if move_to:
            self.target_pose.position = new_position
            self.target_pose.orientation = gripper_pose
            self.robot.move_to(pose=self.target_pose, speed=0.15)
            return
        
        # Create SLERP interpolator for orientation

        if self.n_interpolation == 0:
            self.target_pose.position = new_position
            self.target_pose.orientation = gripper_pose
            self.robot.set_target(pose=self.target_pose)
            self.arm_rate.sleep()
            return

        key_rots = R.concatenate([current_orientation, gripper_pose])
        slerp = Slerp([0, 1], key_rots)
        
        for step in range(self.n_interpolation):
            alpha = (step + 1) / self.n_interpolation
            # Interpolate position (linear)
            interp_position = (1 - alpha) * current_position + alpha * new_position
            # Interpolate orientation (SLERP)
            interp_orientation = slerp(alpha)

            self.target_pose.position = interp_position
            self.target_pose.orientation = interp_orientation
            self.robot.set_target(pose=self.target_pose)
            self.arm_rate.sleep()

        # Set final pose
        self.target_pose.position = new_position
        self.target_pose.orientation = gripper_pose


    def _execute_joint_action(self, action, gripper_pose):
        assert len(action) >= 3, "Action must have at least 3 elements for position"
        if self._last_joint_target is None:
            q_current = self.robot.joint_values.copy()
        else: 
            q_current = self._last_joint_target
        target_pos = action[:3]
        target_quat = gripper_pose.as_quat()
        
        if self._joint_exec_time is not None:
            t_since_last_exec = (time.time() - self._joint_exec_time) * 1000 # convert to ms
            if t_since_last_exec > self._joint_exec_time_tolerance:
                print(f"WARNING: Control loop too slow, took {t_since_last_exec} for new action to arrive")
    
        q_target, success, timing = self.ik_client.solve_ik(target_pos, target_quat, q_current)

        # print(self.robot.end_effector_pose.position, target_pos)
        # print(f"IK time={timing.get('total_time_ms', 0):.4f}ms, iters={timing.get('num_iterations', 0)}, success: {success}")
        
        if not success:
            print(f"  WARNING: IK did not converge")
            time.sleep(4.0) 

        if self.n_interpolation == 0:
            self.robot.set_target_joint(q_target)
            self.arm_rate.sleep()
        else:
            for step in range(self.n_interpolation):
                alpha = (step + 1) / self.n_interpolation
                q_interp = (1 - alpha) * q_current + alpha * q_target
                self.robot.set_target_joint(q_interp)
                self.arm_rate.sleep()

        self._last_joint_target = q_target
        self._joint_exec_time = time.time()


    def _execute_action(self, action, move_to=False):
        """Execute a single action.

        Args:
            
        """
        gripper_position, gripper_rotation, gripper_action = action

        if self.control_space == 'cartesian':
            self._execute_cartesian_action(gripper_position, gripper_rotation, move_to=move_to)

        elif self.control_space == 'joint':
            assert move_to == False, "move_to not supported in joint control space"
            self._execute_joint_action(gripper_position, gripper_rotation)
        
        else:
            raise NotImplementedError(f"Invalid control space: {self.control_space}")
        
        self._execute_gripper_action(gripper_action)

        # Trigger async buffer update (non-blocking)
        self._update_buffer()

    def _switch_to_impedance_controller(self):
        if self.control_space == 'cartesian':
            self.robot.controller_switcher_client.switch_controller("cartesian_impedance_controller")
            self.robot.cartesian_controller_parameters_client.load_param_config(
                file_path="config/control/spacemouse_cartesian_impedance.yaml"
            )
        elif self.control_space == 'joint':
            self.robot.controller_switcher_client.switch_controller("joint_impedance_controller")
            self.robot.joint_controller_parameters_client.load_param_config(
                file_path="config/control/joint_impedance_controller.yaml"
            )
        else:
            raise NotImplementedError(f"Invalid control space: {self.control_space}")

    def cleanup(self):
        """Clean up resources. Call this when done with the controller."""
        self.gripper.set_target(1.0)  # Open gripper
        self._stop_background_buffer_thread()
