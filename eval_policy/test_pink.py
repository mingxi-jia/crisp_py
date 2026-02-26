"""Test Pink IK solver with joint impedance control.

This script:
1. Loads trajectory from HDF5 (same as replay_hand_poses.py)
2. Sends poses to Pink IK server to get joint solutions
3. Executes joint trajectory using joint_impedance_controller

Usage:
    1. Start the Pink IK server (in pink_solver env):
       conda activate pink_solver
       python pink_ik_server.py --port 5002

    2. Run this script (in ROS2/crisp_py env):
       python test_pink.py
"""

import time
import requests
import numpy as np
import h5py

from scipy.spatial.transform import Rotation as R

from crisp_py.robot import Robot
from crisp_py.robot_config import FrankaConfig
from crisp_py.gripper.gripper import Gripper, GripperConfig

from diff_eval_utils.diffusion_constants import DPEvalConfig
from diff_eval_utils.diffusion_transforms import ten_d_action_to_pose, convert_action_from_fingertip_to_gripper, mat_to_rot6d

# Configuration
PINK_SERVER_URL = "http://localhost:5002"
HDF5_FILE = "/media/mingxi/T7/XEMB_Experiment/test/test_coffee_prep_d1_realworld_pretrain.hdf5"


class PinkIKClient:
    """Client for the Pink IK server."""

    def __init__(self, server_url: str):
        self.server_url = server_url

    def health_check(self) -> bool:
        try:
            resp = requests.get(f"{self.server_url}/health", timeout=2)
            return resp.status_code == 200
        except Exception:
            return False

    def solve_ik(self, position, orientation_quat, q_init=None):
        """Solve IK for a single pose.

        Args:
            position: [x, y, z]
            orientation_quat: [qx, qy, qz, qw]
            q_init: Initial joint configuration (optional)

        Returns:
            q: Joint configuration (7 DOF for Panda arm)
            success: Whether IK converged
        """
        payload = {
            "position": position.tolist() if isinstance(position, np.ndarray) else position,
            "orientation": orientation_quat.tolist() if isinstance(orientation_quat, np.ndarray) else orientation_quat,
        }
        if q_init is not None:
            payload["q_init"] = q_init.tolist() if isinstance(q_init, np.ndarray) else q_init

        resp = requests.post(f"{self.server_url}/solve_ik", json=payload)
        data = resp.json()
        return np.array(data.get("q", [])), data.get("success", False)

    def solve_trajectory(self, poses, q_init=None):
        """Solve IK for a trajectory of poses.

        Args:
            poses: List of (position, orientation_quat) tuples
            q_init: Initial joint configuration (optional)

        Returns:
            trajectory: List of joint configurations
            success: Whether all IK solved successfully
        """
        pose_dicts = []
        for pos, quat in poses:
            pose_dicts.append({
                "position": pos.tolist() if isinstance(pos, np.ndarray) else pos,
                "orientation": quat.tolist() if isinstance(quat, np.ndarray) else quat,
            })

        payload = {"poses": pose_dicts}
        if q_init is not None:
            payload["q_init"] = q_init.tolist() if isinstance(q_init, np.ndarray) else q_init

        resp = requests.post(f"{self.server_url}/solve_trajectory", json=payload)
        data = resp.json()
        trajectory = [np.array(q) for q in data.get("trajectory", [])]
        return trajectory, data.get("success", False)


def setup_robot():
    """Initialize robot with joint impedance controller."""
    config = DPEvalConfig()

    # Setup gripper
    gripper_config = GripperConfig.from_yaml("./config/gripper_right.yaml")
    gripper = Gripper(gripper_config=gripper_config, namespace="/right/gripper")
    gripper.wait_until_ready()
    gripper.set_target(1.0)  # Open

    # Setup robot
    config_franka = FrankaConfig()
    config_franka.home_config = config.home_joint_position
    robot = Robot(namespace="", robot_config=config_franka)
    robot.wait_until_ready()

    print(f"Robot ready. Joint values: {robot.joint_values}")
    print(f"End effector pose: {robot.end_effector_pose}")

    # Home the robot first using joint trajectory controller
    print("Homing robot...")
    robot.home()

    # Switch to joint impedance controller for compliant motion
    print("Switching to joint_impedance_controller...")
    robot.controller_switcher_client.switch_controller("joint_impedance_controller")
    robot.joint_controller_parameters_client.load_param_config(
        "./config/control/joint_impedance_controller.yaml"
    )
    time.sleep(0.5)

    return robot, gripper, config


def load_trajectory_from_hdf5(file_path: str):
    """Load hand poses from HDF5 file (same format as replay_hand_poses.py)."""
    dataset = h5py.File(file_path, "r")
    data = dataset['data']['demo_0']

    hand_pos = data['obs']['robot0_eef_pos'][:]
    hand_quat = data['obs']['robot0_eef_quat'][:]
    hand_grasp = data['obs']['robot0_gripper_qpos'][:, 0]

    hand_poses = np.hstack((hand_pos, hand_quat)).astype(float)

    assert len(hand_poses) == len(hand_grasp), f"Length mismatch: {len(hand_poses)} vs {len(hand_grasp)}"

    dataset.close()
    return hand_poses, hand_grasp


def convert_poses_to_gripper_frame(hand_poses):
    """Convert fingertip poses to gripper frame (same as replay_hand_poses.py)."""
    gripper_poses = []

    for hand_pose in hand_poses:
        # Build 10D action from hand pose
        action = np.concatenate([
            hand_pose[:3],
            mat_to_rot6d.forward(
                R.from_quat(hand_pose[3:7]).as_matrix().reshape(1, 3, 3).astype(np.float32)
            )[0],
            [0.0]  # grasp placeholder
        ])

        # Convert to Pose then to gripper frame
        pose, grasp = ten_d_action_to_pose(action, clip=False)
        gripper_pos, gripper_rotation = convert_action_from_fingertip_to_gripper(pose)

        gripper_quat = gripper_rotation.as_quat()  # [qx, qy, qz, qw]
        gripper_poses.append((gripper_pos, gripper_quat))

    return gripper_poses


def main():
    config = DPEvalConfig()
    CTRL_FREQ = config.joint_ctrl_freq

    # Initialize Pink IK client
    print("Connecting to Pink IK server...")
    ik_client = PinkIKClient(PINK_SERVER_URL)

    if not ik_client.health_check():
        print(f"ERROR: Pink IK server not available at {PINK_SERVER_URL}")
        print("Start it with: conda activate pink_solver && python pink_ik_server.py")
        return

    print("Pink IK server connected!")

    # Load trajectory from HDF5
    print(f"\nLoading trajectory from {HDF5_FILE}...")
    hand_poses, hand_grasp = load_trajectory_from_hdf5(HDF5_FILE)
    print(f"Loaded {len(hand_poses)} poses")

    # Convert to gripper frame
    print("Converting poses to gripper frame...")
    gripper_poses = convert_poses_to_gripper_frame(hand_poses)

    # Setup robot
    print("\nInitializing robot...")
    robot, gripper, _ = setup_robot()

    # Get current joint state as initial guess for IK
    q_init = robot.joint_values.copy()

    # Solve IK for entire trajectory
    print("\nSolving IK for trajectory...")
    # gripper_poses = gripper_poses[:30]
    t_start = time.time()
    joint_trajectory, success = ik_client.solve_trajectory(gripper_poses, q_init)
    t_end = time.time()
    print(f"Got IK solving time: {(t_end - t_start)*1000:.4f} milliseconds")

    if not success:
        print("WARNING: Some IK solutions did not converge, but continuing...")
    print(f"Solved {len(joint_trajectory)} joint configurations")

    # Create rate for timing
    rate = robot.node.create_rate(CTRL_FREQ)

    # Move to first pose with interpolation from home
    print("\nMoving to initial pose...")

    # Interpolate from current (home) position to first trajectory point
    num_interp_steps = int(20)  # 2 seconds worth of steps
    q_start = q_init
    q_end = joint_trajectory[0]

    for step in range(num_interp_steps):
        alpha = (step + 1) / num_interp_steps
        q_interp = (1 - alpha) * q_start + alpha * q_end
        robot.set_target_joint(q_interp)
        rate.sleep()

    # Execute trajectory
    print("\nExecuting trajectory...")
    prev_grasp = hand_grasp[0]

    for i, (q_target, grasp_value) in enumerate(zip(joint_trajectory, hand_grasp)):
        print(f"Step {i+1}/{len(joint_trajectory)}: q[0]={q_target[0]:.3f}")

        if i != 0:
            num_interp_steps = int(2)  # 2 seconds worth of steps
            q_start = joint_trajectory[i-1]
            q_end = q_target

            for step in range(num_interp_steps):
                alpha = (step + 1) / num_interp_steps
                q_interp = (1 - alpha) * q_start + alpha * q_end
                robot.set_target_joint(q_interp)
                rate.sleep()

        else:
            # Send joint target
            robot.set_target_joint(q_target)
            rate.sleep()
        # time.sleep(0.1)

        # Handle gripper
        # grasp_cmd = np.round(np.clip(grasp_value, 0, 1))
        # if grasp_cmd != prev_grasp:
        #     gripper.set_target(1 - grasp_cmd)
        #     time.sleep(1.0)  # Wait for gripper
        # prev_grasp = grasp_cmd

    print("\nTrajectory complete. Homing robot...")
    robot.home()
    robot.shutdown()
    print("Done!")


if __name__ == "__main__":
    main()
