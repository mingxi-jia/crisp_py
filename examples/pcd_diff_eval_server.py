"""A ROS2-based point cloud processing example with policy server."""

# %%
import os
import time
from pathlib import Path
import rclpy
import open3d as o3d

from crisp_py.camera.pointcloud import PointCloudManager
from crisp_py.robot import Robot, Pose
from crisp_py.gripper.gripper import Gripper, GripperConfig
from sensor_msgs.msg import JointState

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
    print(f"Time to get latest pcd: {t_latest_pcd*1000:6.1f} ms")

    # if visualize:
    #     robo_pcd = visualize_robot_pcd(pcd, pcd_processor.robot_filter, joint_state)

    t0 = time.time()
    pcd, render_pcd = pcd_client.process_pcd(pcd, eef_pose, joint_state)
    t_process_pcd = time.time() - t0
    print(f"Time to process pcd (total): {t_process_pcd*1000:6.1f} ms")

    t0 = time.time()
    inhand_cam = 'cam4'
    ih_rgb, ih_depth = obs_manager.get_latest_rgbd(inhand_cam)
    rgb_dict, depth_dict = {inhand_cam: ih_rgb}, {inhand_cam: ih_depth}
    rgb_dict, depth_dict = pcd_client.process_images(rgb_dict, depth_dict)
    t_process_images = time.time() - t0
    print(f"Time to process images (total): {t_process_images*1000:6.1f} ms")
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

        # Print timing info
        timing = result['timing']
        print(f"  [Policy server timing]")
        print(f"    to GPU:          {timing['to_gpu']:6.1f} ms")
        print(f"    predict_action:  {timing['predict']:6.1f} ms")
        print(f"    to CPU:          {timing['to_cpu']:6.1f} ms")

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
        print(f"  [PCD processing server timing]")
        print(f"    process:         {timing['process']:6.1f} ms")

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
        print(f"  [Image processing server timing]")
        print(f"    process:         {timing['process']:6.1f} ms")

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
def main():

    ctrl_freq = 10.0 # Hz
    policy_server_port = 5000
    pcd_server_port = 5001

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

    target_pose = robot.end_effector_pose.copy()
    print(f"target_pose: {target_pose}")
    arm_rate = robot.node.create_rate(ctrl_freq)
    gripper_rate = gripper.node.create_rate(ctrl_freq)

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
    n_steps_done = 0 # -1 means warm up
    n_steps_todo = 40 # !! Number of steps to run
    prev_grasp_value = 0.0 # Initialize previous grasp value !! Check value

    print("Starting inference loop...\n====================\n====================")
    while True:
        iter_start = time.time()

        if n_steps_done >= n_steps_todo:
            break

        # Prepare robot obs - get joint state from ROS2 subscriber
        joint_state = joint_state_subscriber.joint_values
        gripper_val = gripper.value
        # print(f"gripper_val: {gripper_val}")
        joint_state = np.concatenate([joint_state, [gripper_norm_const * gripper_val]])
        gripper_state = not gripper.is_open()

        # Prepare robot obs
        time.sleep(0.05) # wait till robot is stable (TODO: better way to do this)

        eef_pose = get_pose_from_robot(robot.end_effector_pose)

        t0 = time.time()
        obs_dict = franka_obs_to_diff_obs(manager, eef_pose, gripper_state, pcd_client, joint_state)
        t_pointcloud = time.time() - t0
        print(f"Time to prepare observation: {t_pointcloud*1000:6.1f} ms")

        # Get action from policy server
        t0 = time.time()
        actions = get_action(obs=obs_dict, policy_client=policy_client)
        t_inference = time.time() - t0

        # print(f"Policy inference: {actions}")
        # visualize_pcd_and_actions(pcd=obs_dict['pcd'], actions=actions)

        t0 = time.time()
        for i, action in enumerate(actions):
            # Make a copy since the array from server is read-only
            action = action.copy()
            action = convert_action_from_fingertip_to_gripper(action, rotation_transformer)

            target_pose.position = action[:3]
            target_pose.orientation = R.from_euler('XYZ', [np.pi, 0, 0])

            # robot.move_to(position=np.array([x, y, z]), speed=0.15)
            robot.set_target(pose=target_pose)
            arm_rate.sleep()

            grasp_value = np.round(np.clip(action[-1], 0, 1))
            if grasp_value != prev_grasp_value:
                gripper.set_target(1-grasp_value)
                gripper_rate.sleep()
                time.sleep(1.0)  # wait for gripper to move (due to franka driver limitation)
            prev_grasp_value = grasp_value
        t_execution = time.time() - t0

        iter_total = time.time() - iter_start
        print(f"\n=== TIMING [step {n_steps_done}] ===")
        print(f"  Point cloud :    {t_pointcloud*1000:6.1f} ms")
        print(f"  Policy infer:    {t_inference*1000:6.1f} ms")
        print(f"  Action exec:     {t_execution*1000:6.1f} ms ({len(actions)} actions)")
        print(f"  TOTAL:           {iter_total*1000:6.1f} ms")
        print("=" * 30 + "\n")

        n_steps_done += 1

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
