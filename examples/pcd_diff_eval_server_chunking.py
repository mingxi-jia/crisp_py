"""A ROS2-based point cloud processing example with policy server."""

# %%
import os
import time
from pathlib import Path
import shutil
import rclpy
import open3d as o3d

from crisp_py.camera.pointcloud import PointCloudManager
from crisp_py.robot import Robot, Pose
from crisp_py.gripper.gripper import Gripper, GripperConfig
from sensor_msgs.msg import JointState

from collections import deque

import sys
toolbox_path = '/home/mingxi/mingxi_ws/handpi/robot-vision-toolbox'
sys.path.append(toolbox_path)

# Diffusion Imports / inits
sys.path.append('/home/mingxi/mingxi_ws/handpi/diffusion_policy')
from diffusion_policy.workspace.base_workspace import BaseWorkspace
from diffusion_policy.policy.base_image_policy import BaseImagePolicy
from diffusion_policy.real_world.real_inference_util import get_real_obs_dict
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.model.common.rotation_transformer import RotationTransformer

from scipy.spatial.transform import Rotation as R

import torch
import numpy as np
import dill
import hydra
import requests
import base64
import matplotlib.pyplot as plt
from matplotlib import cm, ticker


finger_hand_offset = 0.06  # Distance from finger tip to gripper base along z-axis
gripper_norm_const = 0.05  # Normalization constant for gripper value
home_position = np.array([0.60, 0., 0.32 + finger_hand_offset])
joint_thresholds = {
        "panda_link0": 0.08,
        "panda_link1": 0.08,
        "panda_link2": 0.08,
        "panda_link3": 0.08,
        "panda_link4": 0.08,
        "panda_link5": 0.08,
        "panda_link6": 0.08,
        "panda_link7": 0.08,
        "panda_link8": 0.08,
        "panda_hand": 0.02,
        "panda_leftfinger": 0.02,
        "panda_rightfinger": 0.02,
    }

class JointStateSubscriber:
    """A simple ROS2 subscriber to get joint states directly from /joint_states topic."""

    def __init__(self, node, joint_names: list[str], topic: str = "/joint_states"):
        """Initialize the joint state subscriber.

        Args:
            node: ROS2 node to attach the subscription to.
            joint_names: List of joint names to track (in desired order).
            topic: Topic name to subscribe to.
        """
        self._node = node
        self._received = False
        self.joint_array = None

        self._subscription = node.create_subscription(
            JointState,
            topic,
            self._callback,
            10
        )

    def _callback(self, msg: JointState):
        """Update joint positions from the message."""
        joint_names = msg.name
        joint_positions = msg.position
        sorted_indices = sorted(range(len(joint_names)), key=lambda i: joint_names[i])
        sorted_positions = [joint_positions[i] for i in sorted_indices]
        self.joint_array = np.array(sorted_positions, dtype=np.float32)
        self._received = True

    @property
    def joint_values(self) -> np.ndarray:
        """Get joint values in the order specified by joint_names."""
        return self.joint_array[1:]

    @property
    def is_ready(self) -> bool:
        """Check if at least one message has been received."""
        return self._received

def visualize_pcd(pcd: np.ndarray, robot_pcd=None):
    """Visualize point cloud using Open3D.

    Args:
        pcd: Point cloud as a numpy array of shape (N, 6) where the first 3 columns are XYZ and the next 3 are RGB.
        robot_pcd: Optional robot point cloud geometry to visualize alongside the scene.
    """
    pcd_o3d = o3d.geometry.PointCloud()
    pcd_o3d.points = o3d.utility.Vector3dVector(pcd[:, :3])
    pcd_o3d.colors = o3d.utility.Vector3dVector(pcd[:, 3:])

    geometries = [pcd_o3d]
    if robot_pcd is not None:
        geometries.append(robot_pcd)

    o3d.visualization.draw_geometries(geometries)

def visualize_robot_pcd(raw_pcd, robot_seg, joint_state):

    joint_names = sorted([j.name for j in robot_seg.robot_urdf.actuated_joints])
    joint_angles = dict(zip(joint_names, joint_state))
    # Sample robot points
    robot_mesh_dict = robot_seg.robot_urdf.visual_trimesh_fk(cfg=joint_angles)
    sampled_points = []
    for mesh, pose in robot_mesh_dict.items():
        transformed = mesh.copy()
        transformed.apply_transform(pose)
        transformed.apply_transform(robot_seg.T_world_urdf)
        sampled_points.append(transformed.sample(2*2500))

    robot_points = np.vstack(sampled_points)
    robot_pcd = o3d.geometry.PointCloud()
    robot_pcd.points = o3d.utility.Vector3dVector(robot_points)
    robot_pcd.paint_uniform_color([0.75, 0.75, 0.75])  # Grey color

    pcd_o3d = o3d.geometry.PointCloud()
    pcd_o3d.points = o3d.utility.Vector3dVector(raw_pcd[:, :3])
    pcd_o3d.colors = o3d.utility.Vector3dVector(raw_pcd[:, 3:])

    # o3d.visualization.draw_geometries([pcd_o3d, robot_pcd])
    # o3d.visualization.draw_geometries([pcd_o3d])
    return robot_pcd

def visualize_pcd_and_actions(pcd, actions, robot_pcd=None):

    action_frames = []
    for action in actions:
        x, y, z = action[:3]
        rot6d = action[3:9]
        rotation_transformer = RotationTransformer(
            from_rep='rotation_6d', to_rep='matrix')
        rotmat = rotation_transformer.forward(rot6d.reshape(1, 6))

        # Create coordinate frame at action pose
        action_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05)
        action_transform = np.eye(4)
        action_transform[:3, :3] = rotmat
        action_transform[:3, 3] = np.array([x, y, z])
        action_frame.transform(action_transform)
        action_frames.append(action_frame)

    # Visualize filtered pcd with robot model and EEF frame
    pcd_o3d = o3d.geometry.PointCloud()
    pcd_o3d.points = o3d.utility.Vector3dVector(pcd[:, :3])
    pcd_o3d.colors = o3d.utility.Vector3dVector(pcd[:, 3:])
    if robot_pcd is not None:
        o3d.visualization.draw_geometries([pcd_o3d, robot_pcd] + action_frames)
    else:
        o3d.visualization.draw_geometries([pcd_o3d] + action_frames)

def franka_obs_to_diff_obs(obs_manager: PointCloudManager, eef_pose, gripper_state, pcd_client, joint_state, visualize=False):
    """Convert Franka observation to diffusion model observation format.

    Args:
        obs_manager: Point cloud manager for getting raw point clouds
        eef_pose: End effector pose
        gripper_state: Gripper state
        pcd_client: PcdProcessingClient for processing point clouds
        joint_state: Joint state array
        visualize: Whether to visualize

    Returns:
        dict: A dictionary containing the formatted observation.
    """
    t0 = time.time()
    pcd = obs_manager.get_latest_pointcloud()
    t_latest_pcd = time.time() - t0
    # print(f"Time to get latest pcd: {t_latest_pcd*1000:6.1f} ms")

    # if visualize:
    #     robo_pcd = visualize_robot_pcd(pcd, pcd_processor.robot_filter, joint_state)

    t0 = time.time()
    pcd, render_pcd = pcd_client.process_pcd(pcd, eef_pose, joint_state)
    t_process_pcd = time.time() - t0
    # print(f"Time to process pcd (total): {t_process_pcd*1000:6.1f} ms")

    t0 = time.time()
    inhand_cam = 'cam4'
    ih_rgb, ih_depth = obs_manager.get_latest_rgbd(inhand_cam)
    rgb_dict, depth_dict = {inhand_cam: ih_rgb}, {inhand_cam: ih_depth}
    rgb_dict, depth_dict = pcd_client.process_images(rgb_dict, depth_dict)
    t_process_images = time.time() - t0
    # print(f"Time to process images (total): {t_process_images*1000:6.1f} ms")
    print(rgb_dict[inhand_cam].max())

    robot0_eef_pos = eef_pose[:3]
    robot0_eef_quat = eef_pose[3:]  # Assuming quaternion is in (x, y, z, w) format
    robot0_gripper_qpos = np.array([gripper_state, gripper_state], dtype=int) # !!! Check

    # Visualize for debugging
    if visualize:
        pcd_visualize = o3d.geometry.PointCloud()
        pcd_visualize.points = o3d.utility.Vector3dVector(render_pcd[:, :3])
        pcd_visualize.colors = o3d.utility.Vector3dVector(render_pcd[:, 3:])
        o3d.visualization.draw_geometries([pcd_visualize, robo_pcd])

    # Create observation dictionary
    obs = {
        'pcd': pcd, # !!! Check the format, diffusion expects [1024, 6]
        'render_pcd': render_pcd,
        'robot0_eye_in_hand_image': np.transpose(rgb_dict[inhand_cam], (2, 0, 1)) / 255.0,
        'robot0_eef_pos': robot0_eef_pos.astype(np.float32),
        'robot0_eef_quat': robot0_eef_quat.astype(np.float32),
        'robot0_gripper_qpos': robot0_gripper_qpos.astype(np.float32),
    }

    return obs

class PolicyClient:
    """Client for communicating with the policy server."""

    def __init__(self, server_url: str = "http://localhost:5000"):
        self.server_url = server_url
        self._check_health()

    def _check_health(self):
        """Check if server is healthy."""
        try:
            response = requests.get(f"{self.server_url}/health", timeout=5)
            if response.status_code == 200:
                print("Connected to policy server successfully")
            else:
                raise ConnectionError("Policy server unhealthy")
        except Exception as e:
            raise ConnectionError(f"Cannot connect to policy server: {e}")

    def predict_action(self, obs_dict: dict) -> np.ndarray:
        """Get action prediction from server.

        Args:
            obs_dict: Dictionary of observations

        Returns:
            Action array
        """
        # Encode observations as base64
        data = {}
        for key, value in obs_dict.items():
            array_bytes = value.tobytes()
            array_b64 = base64.b64encode(array_bytes).decode('utf-8')
            data[key] = {
                'data': array_b64,
                'dtype': str(value.dtype),
                'shape': value.shape
            }

        # Send request
        response = requests.post(
            f"{self.server_url}/predict",
            json=data,
            timeout=30
        )

        if response.status_code != 200:
            raise RuntimeError(f"Server error: {response.json()}")

        result = response.json()

        # Decode action
        action_bytes = base64.b64decode(result['action']['data'])
        action = np.frombuffer(action_bytes, dtype=result['action']['dtype'])
        action = action.reshape(result['action']['shape'])
        print(f"action shape: {action.shape}")

        # Print timing info
        timing = result['timing']
        # print(f"  [Policy server timing]")
        # print(f"    to GPU:          {timing['to_gpu']:6.1f} ms")
        # print(f"    predict_action:  {timing['predict']:6.1f} ms")
        # print(f"    to CPU:          {timing['to_cpu']:6.1f} ms")

        return action

    def reset(self):
        """Reset the policy."""
        response = requests.post(f"{self.server_url}/reset", timeout=5)
        if response.status_code != 200:
            raise RuntimeError(f"Server error: {response.json()}")

class PcdProcessingClient:
    """Client for communicating with the point cloud processing server."""

    def __init__(self, server_url: str = "http://localhost:5001"):
        self.server_url = server_url
        self._check_health()

    def _check_health(self):
        """Check if server is healthy."""
        try:
            response = requests.get(f"{self.server_url}/health", timeout=5)
            if response.status_code == 200:
                print("Connected to PCD processing server successfully")
            else:
                raise ConnectionError("PCD processing server unhealthy")
        except Exception as e:
            raise ConnectionError(f"Cannot connect to PCD processing server: {e}")

    def process_pcd(self, pcd: np.ndarray, eef_pose: np.ndarray, joint_state: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Process point cloud on the server.

        Args:
            pcd: Raw point cloud array
            eef_pose: End-effector pose
            joint_state: Joint state array

        Returns:
            Tuple of (processed_pcd, render_pcd)
        """
        # Encode inputs as base64
        data = {
            'pcd': {
                'data': base64.b64encode(pcd.tobytes()).decode('utf-8'),
                'dtype': str(pcd.dtype),
                'shape': pcd.shape
            },
            'eef_pose': {
                'data': base64.b64encode(eef_pose.tobytes()).decode('utf-8'),
                'dtype': str(eef_pose.dtype),
                'shape': eef_pose.shape
            },
            'joint_state': {
                'data': base64.b64encode(joint_state.tobytes()).decode('utf-8'),
                'dtype': str(joint_state.dtype),
                'shape': joint_state.shape
            }
        }

        # Send request
        response = requests.post(
            f"{self.server_url}/process_pcd",
            json=data,
            timeout=30
        )

        if response.status_code != 200:
            raise RuntimeError(f"Server error: {response.json()}")

        result = response.json()

        # Decode processed point clouds
        processed_pcd_bytes = base64.b64decode(result['processed_pcd']['data'])
        processed_pcd = np.frombuffer(processed_pcd_bytes, dtype=result['processed_pcd']['dtype'])
        processed_pcd = processed_pcd.reshape(result['processed_pcd']['shape'])

        render_pcd_bytes = base64.b64decode(result['render_pcd']['data'])
        render_pcd = np.frombuffer(render_pcd_bytes, dtype=result['render_pcd']['dtype'])
        render_pcd = render_pcd.reshape(result['render_pcd']['shape'])

        # Print timing info
        timing = result['timing']
        # print(f"  [PCD processing server timing]")
        # print(f"    process:         {timing['process']:6.1f} ms")

        return processed_pcd, render_pcd

    def process_images(self, rgb_dict: dict, depth_dict: dict) -> tuple[dict, dict]:
        """Process RGB and depth images on the server.

        Args:
            rgb_dict: Dictionary of RGB images
            depth_dict: Dictionary of depth images

        Returns:
            Tuple of (processed_rgb_dict, processed_depth_dict)
        """
        # Encode inputs as base64
        data = {
            'rgb_dict': {},
            'depth_dict': {}
        }

        for cam_name, img in rgb_dict.items():
            data['rgb_dict'][cam_name] = {
                'data': base64.b64encode(img.tobytes()).decode('utf-8'),
                'dtype': str(img.dtype),
                'shape': img.shape
            }

        for cam_name, img in depth_dict.items():
            data['depth_dict'][cam_name] = {
                'data': base64.b64encode(img.tobytes()).decode('utf-8'),
                'dtype': str(img.dtype),
                'shape': img.shape
            }

        # Send request
        response = requests.post(
            f"{self.server_url}/process_images",
            json=data,
            timeout=30
        )

        if response.status_code != 200:
            raise RuntimeError(f"Server error: {response.json()}")

        result = response.json()

        # Decode processed images
        processed_rgb = {}
        processed_depth = {}

        for cam_name, img_data in result['rgb_dict'].items():
            img_bytes = base64.b64decode(img_data['data'])
            processed_rgb[cam_name] = np.frombuffer(img_bytes, dtype=img_data['dtype']).reshape(img_data['shape'])

        for cam_name, img_data in result['depth_dict'].items():
            img_bytes = base64.b64decode(img_data['data'])
            processed_depth[cam_name] = np.frombuffer(img_bytes, dtype=img_data['dtype']).reshape(img_data['shape'])

        # Print timing info
        timing = result['timing']
        # print(f"  [Image processing server timing]")
        # print(f"    process:         {timing['process']:6.1f} ms")

        return processed_rgb, processed_depth

def get_action(obs, policy_client):
    """Get action from policy server."""
    t0 = time.time()
    action = policy_client.predict_action(obs)
    t_total = time.time() - t0

    print(f"  [get_action total]:  {t_total*1000:6.1f} ms")

    return action

def get_pose_from_robot(robot_pose: Pose):
    xyz = robot_pose.position
    rotmat = robot_pose.orientation.as_matrix()
    eef_pose = np.eye(4)
    eef_pose[:3, :3] = rotmat
    eef_pose[:3, 3] = xyz

    gripper_offset = np.eye(4)
    gripper_offset[:3, 3] = np.array([0, 0, finger_hand_offset])  # 10 cm offset along z-axis

    gripper_pose = eef_pose @ gripper_offset
    gripper_xyz = gripper_pose[:3, 3]
    gripper_orientation = R.from_matrix(gripper_pose[:3, :3]).as_quat()
    return np.concatenate([gripper_xyz, gripper_orientation], axis=0)

def convert_action_from_fingertip_to_gripper(action, rot6d_to_mat):
    # print(f"Original action: {action}")
    # Ensure action is writable (make a copy if needed)
    if not action.flags.writeable:
        action = action.copy()

    rot6d = action[3:9]
    rotmat = rot6d_to_mat.forward(rot6d.reshape(1, 6))

    action[2] = np.clip(action[2], 0.0, 0.6)
    finger_pose = np.eye(4)
    finger_pose[:3, :3] = rotmat[0]
    finger_pose[:3, 3] = action[:3]

    gripper_offset = np.eye(4)
    gripper_offset[:3, 3] = np.array([0, 0, -finger_hand_offset])  # 10 cm offset along z-axis

    gripper_pose = finger_pose @ gripper_offset
    gripper_xyz = gripper_pose[:3, 3]
    gripper_rot6d = rot6d_to_mat.inverse(gripper_pose[:3, :3].reshape(1, 3, 3))[0]
    grasp = action[-1:]

    action_converted = np.concatenate([gripper_xyz, gripper_rot6d, grasp], axis=0)
    # print(f"Converted action: {action_converted}")
    return action_converted

# %%
class RealTimeChunkingController:
    policy_delay = 4 # 4 steps under 10 Hz
    horizon = 16
    action_exec_size = 4 # Number of actions to execute per chunk
    debug_output_dir = Path("debug_plots")  # Directory for debug plots

    def __init__(self, robot, gripper, obs_manager, joint_state_subscriber,
                 policy_client, pcd_client, rotation_transformer, ctrl_freq=10.0,
                 debug_plotting=False):
        self.robot = robot
        self.gripper = gripper
        self.obs_manager = obs_manager
        self.joint_state_subscriber = joint_state_subscriber
        self.policy_client = policy_client
        self.pcd_client = pcd_client
        self.rotation_transformer = rotation_transformer
        self.ctrl_freq = ctrl_freq
        self.debug_plotting = debug_plotting

        self.action_queue = deque()
        self.target_pose = robot.end_effector_pose.copy()
        self.arm_rate = robot.node.create_rate(ctrl_freq)
        self.gripper_rate = gripper.node.create_rate(ctrl_freq)
        self.prev_grasp_value = 0.0

        self.last_inference_time = None
        self.inference_count = 0

        # Debug plotting data
        if self.debug_plotting:
            self._setup_debug_output_dir()
            self.predicted_actions = []  # List of (inference_idx, timestep, action)
            self.actual_poses = []  # List of (timestep, x, y, z, yaw)
            self.control_loop_times = []  # List of control loop execution times
            self.all_predicted_chunks = []  # List of (inference_idx, full 16-action chunk)
            self.start_time = None

    def _setup_debug_output_dir(self):
        """Create or clear the debug output directory."""
        if self.debug_output_dir.exists():
            print(f"Clearing existing debug output directory: {self.debug_output_dir}")
            shutil.rmtree(self.debug_output_dir)
        self.debug_output_dir.mkdir(parents=True, exist_ok=True)
        print(f"Debug plots will be saved to: {self.debug_output_dir}")


    def _need_inference(self):
        # When the current action buffer is below policy_delay, we need to get new actions
        if len(self.action_queue) <= self.policy_delay:
            return True
        return False

    def _run_inference(self):
        # Track inference timing
        current_time = time.time()
        if self.last_inference_time is not None:
            time_since_last = current_time - self.last_inference_time
            print(f"\n{'='*50}")
            print(f"[Inference #{self.inference_count}] Time since last inference: {time_since_last*1000:6.1f} ms")
            print(f"{'='*50}")

        # Prepare robot obs - get joint state from ROS2 subscriber
        joint_state = self.joint_state_subscriber.joint_values
        gripper_val = self.gripper.value
        joint_state = np.concatenate([joint_state, [gripper_norm_const * gripper_val]])
        gripper_state = not self.gripper.is_open()


        eef_pose = get_pose_from_robot(self.robot.end_effector_pose)

        # Get observation
        obs_dict = franka_obs_to_diff_obs(self.obs_manager, eef_pose, gripper_state,
                                         self.pcd_client, joint_state)

        # Get actions from policy server
        actions = self.policy_client.predict_action(obs_dict)

        # Record predicted actions for debug plotting (only the executed subset)
        if self.debug_plotting and self.start_time is not None:
            current_elapsed = time.time() - self.start_time
            for i, action in enumerate(actions[self.policy_delay: self.policy_delay + self.action_exec_size]):
                # Extract x, y, z, yaw from action
                xyz = action[:3]
                rot6d = action[3:9]
                rotmat = self.rotation_transformer.forward(rot6d.reshape(1, 6))[0]
                yaw = np.arctan2(rotmat[1, 0], rotmat[0, 0])

                # Store: (inference_idx, predicted_timestep, x, y, z, yaw)
                predicted_timestep = current_elapsed + (i + 1) / self.ctrl_freq
                self.predicted_actions.append((self.inference_count, predicted_timestep, xyz[0], xyz[1], xyz[2], yaw))

        # Store all 16 actions for the chunk visualization (regardless of start_time)
        if self.debug_plotting:
            self.all_predicted_chunks.append((self.inference_count, actions.copy()))

        if len(self.action_queue) == 0:
            # If queue is empty, just add all actions
            for action in actions[:self.action_exec_size]:
                self.action_queue.append(action.copy())
        # Add actions to queue
        for action in actions[self.policy_delay: self.policy_delay + self.action_exec_size]:
            self.action_queue.append(action.copy())

        # Update timing
        self.last_inference_time = time.time()
        print(f"Inference took {(self.last_inference_time - current_time)*1000:6.1f} ms")
        self.inference_count += 1

    def run(self, n_steps):
        n_steps_done = 0

        # Start timing for debug plotting
        if self.debug_plotting:
            self.start_time = time.time()

        self._run_inference()  # Initial inference
        while n_steps_done < n_steps:
            loop_start = time.time()

            # Check if we need to run inference
            if self._need_inference():
                self._run_inference()

            # Get action from the queue and execute
            if len(self.action_queue) > 0:
                action = self.action_queue.popleft()
                action = convert_action_from_fingertip_to_gripper(action, self.rotation_transformer)

                self.target_pose.position = action[:3]
                self.target_pose.orientation = R.from_euler('XYZ', [np.pi, 0, 0])

                self.robot.set_target(pose=self.target_pose)

                # Record actual pose for debug plotting
                if self.debug_plotting:
                    current_pose = self.robot.end_effector_pose
                    rotmat = current_pose.orientation.as_matrix()
                    yaw = np.arctan2(rotmat[1, 0], rotmat[0, 0])
                    elapsed = time.time() - self.start_time
                    self.actual_poses.append((elapsed, current_pose.position[0],
                                             current_pose.position[1], current_pose.position[2], yaw))

                self.arm_rate.sleep()

                grasp_value = np.round(np.clip(action[-1], 0, 1))
                if grasp_value != self.prev_grasp_value:
                    self.gripper.set_target(1-grasp_value)
                    self.gripper_rate.sleep()
                    time.sleep(1.0)
                self.prev_grasp_value = grasp_value

                n_steps_done += 1

            # Record control loop time
            if self.debug_plotting:
                loop_time = (time.time() - loop_start) * 1000  # Convert to ms
                self.control_loop_times.append(loop_time)

        # Generate debug plots after execution
        if self.debug_plotting:
            self._plot_debug_results()
            self._plot_action_chunks()
            print(f"\n{'='*60}")
            print(f"All debug plots saved to: {self.debug_output_dir.absolute()}")
            print(f"{'='*60}")

    def _plot_debug_results(self):
        """Generate plots showing predicted vs actual trajectories and control loop timing."""
        print("\nGenerating debug plots...")

        # Convert data to numpy arrays
        actual = np.array(self.actual_poses)  # (N, 5): time, x, y, z, yaw
        predicted = np.array(self.predicted_actions)  # (M, 6): inference_idx, time, x, y, z, yaw
        loop_times = np.array(self.control_loop_times)  # (N,): loop execution times in ms

        # Create colormap for different inference chunks
        n_inferences = int(predicted[:, 0].max()) + 1
        colors = cm.rainbow(np.linspace(0, 1, n_inferences))

        # ===== Figure 1: Trajectory comparison (2x2) =====
        fig1, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig1.suptitle('Predicted Actions vs Actual Execution', fontsize=14, fontweight='bold')

        labels = ['X Position (m)', 'Y Position (m)', 'Z Position (m)', 'Yaw (rad)']
        actual_indices = [1, 2, 3, 4]  # Column indices in actual array
        pred_indices = [2, 3, 4, 5]  # Column indices in predicted array (inference_idx, time, x, y, z, yaw)

        for idx, (ax, label, actual_col, pred_col) in enumerate(zip(axes.flatten(), labels, actual_indices, pred_indices)):
            # Plot actual trajectory (ground truth) in black
            ax.plot(actual[:, 0], actual[:, actual_col], 'k-', linewidth=2, label='Actual (Ground Truth)', zorder=10)

            # Plot predicted actions colored by inference chunk
            for inf_idx in range(n_inferences):
                mask = predicted[:, 0] == inf_idx
                chunk_data = predicted[mask]
                if len(chunk_data) > 0:
                    ax.scatter(chunk_data[:, 1], chunk_data[:, pred_col],
                              color=colors[inf_idx], s=30, alpha=0.7,
                              label=f'Inference #{inf_idx}', zorder=5)

            ax.set_xlabel('Time (s)', fontsize=10)
            ax.set_ylabel(label, fontsize=10)
            ax.xaxis.set_major_locator(ticker.MultipleLocator(4))
            ax.grid(True, alpha=0.3)
            ax.set_title(label, fontsize=11, fontweight='bold')

            # Only show legend for first plot (to avoid clutter)
            if idx == 0:
                ax.legend(loc='best', fontsize=8, ncol=2)

        plt.tight_layout()

        # Save trajectory plot
        filename1 = self.debug_output_dir / "trajectory_comparison.png"
        plt.savefig(filename1, dpi=150, bbox_inches='tight')
        print(f"Trajectory plot saved to: {filename1}")

        # ===== Figure 2: Control loop timing histogram =====
        fig2, ax = plt.subplots(1, 1, figsize=(10, 6))

        # Create histogram
        ax.hist(loop_times, bins=30, color='steelblue', alpha=0.7, edgecolor='black')

        # Add statistics
        mean_time = np.mean(loop_times)
        median_time = np.median(loop_times)
        std_time = np.std(loop_times)
        min_time = np.min(loop_times)
        max_time = np.max(loop_times)

        # Add vertical lines for mean and median
        ax.axvline(mean_time, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_time:.2f} ms')
        ax.axvline(median_time, color='green', linestyle='--', linewidth=2, label=f'Median: {median_time:.2f} ms')

        # Add text box with statistics
        stats_text = f'Statistics:\n' \
                    f'Mean: {mean_time:.2f} ms\n' \
                    f'Median: {median_time:.2f} ms\n' \
                    f'Std Dev: {std_time:.2f} ms\n' \
                    f'Min: {min_time:.2f} ms\n' \
                    f'Max: {max_time:.2f} ms\n' \
                    f'Total loops: {len(loop_times)}'

        ax.text(0.98, 0.97, stats_text, transform=ax.transAxes,
               verticalalignment='top', horizontalalignment='right',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
               fontsize=10, family='monospace')

        ax.set_xlabel('Control Loop Execution Time (ms)', fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)
        ax.set_title('Control Loop Timing Distribution', fontsize=14, fontweight='bold')
        ax.xaxis.set_major_locator(ticker.MultipleLocator(4))
        ax.grid(True, alpha=0.3, axis='y')
        ax.legend(loc='upper left', fontsize=10)

        plt.tight_layout()

        # Save timing plot
        filename2 = self.debug_output_dir / "control_loop_timing.png"
        plt.savefig(filename2, dpi=150, bbox_inches='tight')
        print(f"Timing histogram saved to: {filename2}")

        plt.show(block=False)
        plt.pause(0.1)

    def _plot_action_chunks(self):
        """Generate plots showing all 16 predicted actions from each inference chunk."""
        print("\nGenerating action chunk plots...")

        if len(self.all_predicted_chunks) == 0:
            print("No action chunks to plot")
            return

        # Create colormap for different inference chunks
        n_inferences = len(self.all_predicted_chunks)
        colors = cm.rainbow(np.linspace(0, 1, n_inferences))

        # Create 2x2 subplot
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('All Predicted Action Chunks (16 actions per inference)', fontsize=14, fontweight='bold')

        labels = ['X Position (m)', 'Y Position (m)', 'Z Position (m)', 'Yaw (rad)']

        for ax, label in zip(axes.flatten(), labels):
            # Plot each inference chunk
            for inf_idx, (chunk_id, actions) in enumerate(self.all_predicted_chunks):
                # Calculate starting n_step for this chunk
                # First chunk starts at 0, subsequent chunks start at 8 * idx
                if inf_idx == 0:
                    start_step = 0
                else:
                    start_step = self.action_exec_size * inf_idx

                # Extract all 16 actions
                x_vals = []
                y_vals = []

                for action_idx in range(len(actions)):
                    action = actions[action_idx]
                    n_step = start_step + action_idx

                    # Extract the value based on which subplot we're in
                    if label == 'X Position (m)':
                        value = action[0]
                    elif label == 'Y Position (m)':
                        value = action[1]
                    elif label == 'Z Position (m)':
                        value = action[2]
                    else:  # Yaw
                        rot6d = action[3:9]
                        rotmat = self.rotation_transformer.forward(rot6d.reshape(1, 6))[0]
                        value = np.arctan2(rotmat[1, 0], rotmat[0, 0])

                    x_vals.append(n_step)
                    y_vals.append(value)

                # Plot the chunk
                ax.plot(x_vals, y_vals, 'o-', color=colors[inf_idx],
                       linewidth=2, markersize=6, alpha=0.7,
                       label=f'Inference #{chunk_id}')

            ax.set_xlabel('n_step', fontsize=11, fontweight='bold')
            ax.set_ylabel(label, fontsize=11, fontweight='bold')
            ax.xaxis.set_major_locator(ticker.MultipleLocator(4))
            ax.grid(True, alpha=0.3)
            ax.set_title(label, fontsize=12, fontweight='bold')
            ax.legend(loc='best', fontsize=8, ncol=2)

        plt.tight_layout()

        # Save plot
        filename = self.debug_output_dir / "action_chunks.png"
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        print(f"Action chunks plot saved to: {filename}")

        plt.show(block=False)
        plt.pause(0.1)


def main():

    ctrl_freq = 10.0 # Hz
    policy_server_port = 5000
    pcd_server_port = 5001
    debug_plotting = True  # Enable/disable debug plotting

    ### ---- Client Setup ----- ###
    policy_client = PolicyClient(f"http://localhost:{policy_server_port}")
    pcd_client = PcdProcessingClient(f"http://localhost:{pcd_server_port}")
    rotation_transformer = RotationTransformer(
            from_rep='rotation_6d', to_rep='matrix')

    ### ---- Robot Setup ----- ###
    """Test the PointCloudManager."""
    rclpy.init()
    # Initialize robot
    robot = Robot(namespace="")
    robot.wait_until_ready()
    print(f"Robot ready. Current joint values: {robot.joint_values}")
    print(robot.end_effector_pose)
    print(robot.joint_values)
    print("Going to home position...")
    robot.home()

    robot.controller_switcher_client.switch_controller("cartesian_impedance_controller")
    robot.cartesian_controller_parameters_client.load_param_config(
        file_path="config/control/default_cartesian_impedance.yaml"
    )

    print("Going to start position...")
    robot.move_to(position=home_position, speed=0.15)

    # Initialize gripper
    gripper_config = GripperConfig.from_yaml("./config/gripper_right.yaml")
    gripper = Gripper(gripper_config=gripper_config, namespace="/right/gripper")
    gripper.wait_until_ready()
    gripper.set_target(1.0)
    print("Gripper ready")

    ### ---- Point Cloud Setup ----- ###
    # Initialize robot filter
    config_path = os.path.join(toolbox_path, "configs", "camera_info.yaml")
    manager = PointCloudManager(config_path) # Create point cloud manager

    # Get joint names - we need to load the URDF temporarily to get the joint names
    # This is only needed for the JointStateSubscriber
    from robot_filter.arm_segmentor import RobotArmSegmentation
    temp_robot_seg = RobotArmSegmentation()
    joint_names = sorted([j.name for j in temp_robot_seg.robot_urdf.actuated_joints])
    joint_state_subscriber = JointStateSubscriber(manager, joint_names, topic="/joint_states")

    import threading # Spin in background thread to receive messages
    spin_thread = threading.Thread(target=rclpy.spin, args=(manager,), daemon=True)
    # spin_thread = threading.Thread(target=rclpy.spin, args=(manager,))
    spin_thread.start()

    # Wait for joint states to be received
    while not joint_state_subscriber.is_ready:
        time.sleep(0.1)
    print(f"Joint state subscriber ready. Joint names: {joint_names}")


    ### ---- Inference Loop ----- ###
    n_steps_todo = 60 # !! Number of steps to run

    print("Starting inference loop...\n====================\n====================")

    # Create controller
    controller = RealTimeChunkingController(
        robot=robot,
        gripper=gripper,
        obs_manager=manager,
        joint_state_subscriber=joint_state_subscriber,
        policy_client=policy_client,
        pcd_client=pcd_client,
        rotation_transformer=rotation_transformer,
        ctrl_freq=ctrl_freq,
        debug_plotting=debug_plotting
    )

    # Run controller
    controller.run(n_steps=n_steps_todo)

    # Cleanup
    robot.home()
    robot.shutdown()
    manager.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nKeyboard interrupt received, shutting down...")
        try:
            rclpy.shutdown()
        except Exception:
            pass
        try:
            sys.exit(0)
        except SystemExit:
            pass
