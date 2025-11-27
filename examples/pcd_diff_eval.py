"""A ROS2-based point cloud processing example."""

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
from robot_filter.arm_segmentor import RobotArmSegmentation

from utils.pcd_utils import render_pcd_from_pose
from hand.trajectory_loader import ObservationProcessor

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

def franka_obs_to_diff_obs(obs_manager: PointCloudManager, eef_pose, gripper_state, pcd_processor, joint_state):
    """Convert Franka observation to diffusion model observation format.
    
    Args:
        pcd (np.ndarray): The input point cloud.
        end_effector_pose (np.ndarray): The robot's end effector pose.
    
    Returns:
        dict: A dictionary containing the formatted observation.
    """
    t0 = time.time()
    pcd = obs_manager.get_latest_pointcloud()
    t_latest_pcd = time.time() - t0
    print(f"Time to get latest pcd: {t_latest_pcd*1000:6.1f} ms")

    # robo_pcd = visualize_robot_pcd(pcd, pcd_processor.robot_filter, joint_state)

    t0 = time.time()
    pcd, render_pcd = pcd_processor.get_policy_obs(pcd, eef_pose, joint_state)
    # print(pcd.shape)
    t_process_pcd = time.time() - t0
    print(f"Time to process pcd: {t_process_pcd*1000:6.1f} ms")

    t0 = time.time()
    inhand_cam = 'cam4'
    ih_rgb, ih_depth = obs_manager.get_latest_rgbd(inhand_cam)
    rgb_dict, depth_dict = {inhand_cam: ih_rgb}, {inhand_cam: ih_depth}
    # print(rgb_dict[inhand_cam].max())
    rgb_dict, depth_dict, is_contact = pcd_processor.get_policy_images(rgb_dict, depth_dict)
    t_process_images = time.time() - t0
    print(f"Time to process images: {t_process_images*1000:6.1f} ms")
    print(rgb_dict[inhand_cam].max())
    
    robot0_eef_pos = eef_pose[:3]
    robot0_eef_quat = eef_pose[3:]  # Assuming quaternion is in (x, y, z, w) format   
    robot0_gripper_qpos = np.array([gripper_state, gripper_state], dtype=int) # !!! Check
    
    # Visualize for debugging
    # pcd_visualize = o3d.geometry.PointCloud()
    # pcd_visualize.points = o3d.utility.Vector3dVector(render_pcd[:, :3])
    # pcd_visualize.colors = o3d.utility.Vector3dVector(render_pcd[:, 3:])
    # o3d.visualization.draw_geometries([pcd_visualize, robo_pcd])

    # Create observation dictionary
    obs = {
        # 'pcd': pcd, # !!! Check the format, diffusion expects [1024, 6]
        'render_pcd': render_pcd,
        # 'robot0_eye_in_hand_image': np.transpose(rgb_dict[inhand_cam], (2, 0, 1)) / 255.0,  
        'robot0_eef_pos': robot0_eef_pos.astype(np.float32), 
        'robot0_eef_quat': robot0_eef_quat.astype(np.float32),
        'robot0_gripper_qpos': robot0_gripper_qpos.astype(np.float32),
    }

    return obs

# !! Not sure how necessary this is 
def warm_up_policy(obs, policy, cfg):
    with torch.no_grad():
        policy.reset()
        device = torch.device('cuda')
        obs_dict_np = get_real_obs_dict(
            env_obs=obs, shape_meta=cfg.task.shape_meta)
        obs_dict = dict_apply(obs_dict_np, 
            lambda x: torch.from_numpy(x).unsqueeze(0).unsqueeze(1).to(device))
        result = policy.predict_action(obs_dict)
        action = result['action'][0].detach().to('cpu').numpy()
        assert action.shape[-1] == 2
        del result


def get_action(obs, policy, cfg):
    with torch.no_grad():
        device = torch.device('cuda')

        # Time obs_dict preparation
        t0 = time.time()
        # obs_dict_np = get_real_obs_dict(
        #     env_obs=obs, shape_meta=cfg.task.shape_meta)
        t_obs_dict = time.time() - t0

        # Time tensor conversion and GPU transfer
        t0 = time.time()
        obs_dict = dict_apply(obs,
            lambda x: torch.from_numpy(x).unsqueeze(0).unsqueeze(1).to(device))
        torch.cuda.synchronize()  # Ensure GPU transfer is complete
        t_to_gpu = time.time() - t0

        # Time policy inference
        t0 = time.time()
        result = policy.predict_action(obs_dict)
        torch.cuda.synchronize()  # Ensure inference is complete
        t_predict = time.time() - t0

        # Time result transfer back to CPU
        t0 = time.time()
        action = result['action'][0].detach().to('cpu').numpy()
        t_to_cpu = time.time() - t0

        print(f"  [get_action breakdown]")
        print(f"    obs_dict prep:   {t_obs_dict*1000:6.1f} ms")
        print(f"    to GPU:          {t_to_gpu*1000:6.1f} ms")
        print(f"    predict_action:  {t_predict*1000:6.1f} ms")
        print(f"    to CPU:          {t_to_cpu*1000:6.1f} ms")

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
    ckpt_path = "/home/mingxi/Downloads/epoch=0080-val_loss=0.011.ckpt"
    # ckpt_path = "/home/mingxi/mingxi_ws/handpi/data/data/outputs/2025.11.20/19.43.13_diff_voxel_lift_block_realworld_38_None/checkpoints/epoch=0110-val_loss=0.032.ckpt"

    ### ---- Policy Setup ----- ###
    # Load the diffusion model
    payload = torch.load(open(ckpt_path, 'rb'), pickle_module=dill)
    cfg = payload['cfg']
    cfg.logging.resume = False
    cfg.logging.mode = 'offline'  # Disable logging
    cfg.real_robot_eval = True  # Enable real robot eval mode
    cls = hydra.utils.get_class(cfg._target_)
    workspace = cls(cfg)
    workspace: BaseWorkspace
    workspace.load_payload(payload, exclude_keys=None, include_keys=None)

    # Setup the diffusion model
    delta_action = False
    policy: BaseImagePolicy
    policy = workspace.model # !!! didn't include use ema
    device = torch.device('cuda')
    policy.eval()
    policy.to(device)
    policy.num_inference_steps = 20 
    n_action_steps = 8
    policy.n_action_steps = n_action_steps
    policy.reset()
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

    # robot_seg = RobotArmSegmentation(joint_thresholds=joint_thresholds, urdf_path=toolbox_path + "/robot_filter/panda_description/urdf/panda_arm_hand_finray.urdf")
    # robot_seg.load_urdf(toolbox_path + "/robot_filter/panda_description/urdf/panda_arm_hand_finray.urdf")

    pcd_processor = ObservationProcessor()

    # Get joint names from the robot segmentor's URDF (sorted alphabetically)
    joint_names = sorted([j.name for j in pcd_processor.robot_filter.robot_urdf.actuated_joints])
    joint_state_subscriber = JointStateSubscriber(manager, joint_names, topic="/joint_states")

    import threading # Spin in background thread to receive messages
    # spin_thread = threading.Thread(target=rclpy.spin, args=(manager,), daemon=True)
    spin_thread = threading.Thread(target=rclpy.spin, args=(manager,))
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
        # joint_state = np.concatenate([joint_state, [gripper_norm_const * gripper_val]])
        gripper_state = not gripper.is_open()

        # Prepare robot obs
        time.sleep(0.05) # wait till robot is stable (TODO: better way to do this)

        eef_pose = get_pose_from_robot(robot.end_effector_pose)

        t0 = time.time()
        obs_dict = franka_obs_to_diff_obs(manager, eef_pose, gripper_state, pcd_processor, joint_state)
        # robot_pcd = visualize_robot_pcd(pcd, pcd_processor.robot_filter, joint_state)
        t_pointcloud = time.time() - t0

        if n_steps_done == -1: # Warm up policy
            # !! Update the obs input to the policy
            warm_up_policy(obs=obs_dict, policy=policy, cfg=cfg)  # Replace None with actual observation if available
            n_steps_done += 1
            continue

        # Get action from policy
        t0 = time.time()
        actions = get_action(obs=obs_dict, policy=policy, cfg=cfg)
        t_inference = time.time() - t0

        # print(f"Policy inference: {actions}")
        # visualize_pcd_and_actions(pcd=obs_dict['pcd'], actions=actions)

        t0 = time.time()
        for action in actions:
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
