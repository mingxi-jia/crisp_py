"""Visualize robot trajectory with Pink IK solver using local URDF.

This script:
1. Loads trajectory from HDF5 (same as test_pink.py)
2. Solves IK using Pink with local fr3_robot.urdf
3. Optionally applies low pass filter to smooth the trajectory
4. Visualizes the robot trajectory in meshcat (browser-based)
5. Optionally plots IK error (ground truth vs solved) over time
6. Optionally plots trajectory comparison (ground truth vs IK solved)

Run in pink_solver environment:
    conda activate pink_solver
    python test_pink_sim_2.py

    # With IK error plot:
    python test_pink_sim_2.py --plot-error

    # With trajectory comparison plot (recommended for accuracy analysis):
    python test_pink_sim_2.py --plot-comparison

    # Both plots:
    python test_pink_sim_2.py --plot-error --plot-comparison

    # Skip meshcat visualization, only show plots:
    python test_pink_sim_2.py --plot-comparison --no-viz

    # Apply low pass filter to smooth trajectory:
    python test_pink_sim_2.py --lowpass

    # Customize low pass filter (cutoff freq in Hz, filter order):
    python test_pink_sim_2.py --lowpass --lowpass-cutoff 1.5 --lowpass-order 3

Opens visualization at: http://127.0.0.1:7000/static/
"""

import os
import time
import argparse
import numpy as np
import h5py
import matplotlib.pyplot as plt
import sys
from scipy.signal import butter, filtfilt

os.environ['PYOPENGL_PLATFORM'] = 'egl'

from scipy.spatial.transform import Rotation as R
from diff_eval_utils.diffusion_constants import DPEvalConfig

config = DPEvalConfig
toolbox_path = config.toolbox_path
robot_filter_path = toolbox_path + "/robot_filter"
sys.path.append(robot_filter_path)

import pinocchio as pin
import pink
from pink import solve_ik
from pink.tasks import FrameTask

from pink_realtime.util import load_robot_from_urdf

# Try to import meshcat visualizer
try:
    from pinocchio.visualize import MeshcatVisualizer
    HAS_MESHCAT = True
except ImportError:
    HAS_MESHCAT = False
    print("WARNING: MeshcatVisualizer not available. Install with: pip install meshcat")

# Configuration
SCRIPT_DIR = robot_filter_path + "/panda_description/urdf"
URDF_PATH = os.path.join(SCRIPT_DIR, "panda_arm_robotiq.urdf")
HDF5_FILE = "/media/mingxi/T7/XEMB_Experiment/coffee_prep/replay_hand_test/test_2_smoothed.hdf5"
END_EFFECTOR_FRAME = "robotiq_tcp_link"  # End effector frame from local URDF
CTRL_FREQ = 10.0  # Hz

# Fingertip to gripper offset
FINGER_HAND_OFFSET = 0.20

# Home joint position from DPEvalConfig (valid for FR3 joint limits)
HOME_JOINT_POSITION = np.array([
    0.0015795138042423453, 0.11460111156789562, 0.00012723805921852443,
    -1.9088541631957334, 0.007562769235242739, 2.16821183580947, 0.7848162419679248, 0, 0, 0, 0, 0, 0
])


def load_trajectory_from_hdf5(file_path: str):
    """Load hand poses from HDF5 file."""
    dataset = h5py.File(file_path, "r")
    data = dataset['data']['demo_0']

    hand_pos = data['obs']['robot0_eef_pos'][:]
    hand_quat = data['obs']['robot0_eef_quat'][:]
    hand_grasp = data['obs']['robot0_gripper_qpos'][:, 0]

    hand_poses = np.hstack((hand_pos, hand_quat)).astype(float)

    dataset.close()
    return hand_poses, hand_grasp


def convert_action_from_fingertip_to_gripper_local(position, quat):
    """Convert fingertip pose to gripper frame (simplified local version)."""
    # Build fingertip pose matrix
    finger_pose = np.eye(4)
    finger_pose[:3, :3] = R.from_quat(quat).as_matrix()
    finger_pose[:3, 3] = position

    # Apply offset (fingertip to gripper)
    gripper_offset = np.eye(4)
    gripper_offset[:3, 3] = np.array([0, 0, -FINGER_HAND_OFFSET])

    gripper_pose = finger_pose @ gripper_offset
    gripper_pos = gripper_pose[:3, 3]
    gripper_quat = R.from_matrix(gripper_pose[:3, :3]).as_quat()

    return gripper_pos, gripper_quat


def convert_poses_to_gripper_frame(hand_poses):
    """Convert all fingertip poses to gripper frame."""
    gripper_poses = []
    for hand_pose in hand_poses:
        pos = hand_pose[:3]
        quat = hand_pose[3:7]
        gripper_pos, gripper_quat = convert_action_from_fingertip_to_gripper_local(pos, quat)
        gripper_poses.append((gripper_pos, gripper_quat))
    return gripper_poses


def interpolate_trajectory(trajectory, num_interp=5):
    """Interpolate between consecutive frames to smooth the trajectory.

    Args:
        trajectory: List of joint configurations (numpy arrays)
        num_interp: Number of frames to insert between each pair of consecutive frames

    Returns:
        Interpolated trajectory with (len(trajectory)-1) * (num_interp+1) + 1 frames
    """
    if len(trajectory) < 2 or num_interp < 1:
        return trajectory

    interpolated = []

    for i in range(len(trajectory) - 1):
        q_start = np.array(trajectory[i])
        q_end = np.array(trajectory[i + 1])

        # Add start frame
        interpolated.append(q_start)

        # Add interpolated frames
        for j in range(1, num_interp + 1):
            alpha = j / (num_interp + 1)
            q_interp = (1 - alpha) * q_start + alpha * q_end
            interpolated.append(q_interp)

    # Add final frame
    interpolated.append(np.array(trajectory[-1]))

    return interpolated


def lowpass_filter_trajectory(trajectory, cutoff_freq, sample_freq, order=2):
    """Apply a Butterworth low pass filter to the trajectory.

    Args:
        trajectory: List of joint configurations (numpy arrays)
        cutoff_freq: Cutoff frequency in Hz (frequencies above this are attenuated)
        sample_freq: Sampling frequency of the trajectory in Hz
        order: Order of the Butterworth filter (default: 2)

    Returns:
        Filtered trajectory as list of numpy arrays
    """
    if len(trajectory) < 4:
        print("Warning: Trajectory too short for filtering, returning original")
        return trajectory

    # Convert to numpy array for easier processing
    traj_array = np.array(trajectory)  # Shape: (n_frames, n_joints)
    n_frames, n_joints = traj_array.shape

    # Normalize cutoff frequency (Nyquist frequency = sample_freq / 2)
    nyquist = sample_freq / 2.0
    if cutoff_freq >= nyquist:
        print(f"Warning: Cutoff freq ({cutoff_freq} Hz) >= Nyquist freq ({nyquist} Hz). "
              f"Reducing to {nyquist * 0.9:.2f} Hz")
        cutoff_freq = nyquist * 0.9

    normalized_cutoff = cutoff_freq / nyquist

    # Design Butterworth low pass filter
    b, a = butter(order, normalized_cutoff, btype='low', analog=False)

    # Apply filter to each joint dimension
    filtered_array = np.zeros_like(traj_array)
    for j in range(n_joints):
        # Use filtfilt for zero-phase filtering (no phase delay)
        filtered_array[:, j] = filtfilt(b, a, traj_array[:, j])

    # Convert back to list of arrays
    filtered_trajectory = [filtered_array[i, :] for i in range(n_frames)]

    return filtered_trajectory


def solve_trajectory_ik(robot, gripper_poses, q_init):
    """Solve IK for entire trajectory.

    Returns:
        trajectory: List of joint configurations
        solved_poses: List of (position, quaternion) tuples for the solved end-effector poses
    """
    model = robot.model
    data = robot.data

    # Initialize
    config = pink.Configuration(model, data, q_init)

    # Create end-effector task
    end_effector_task = FrameTask(
        END_EFFECTOR_FRAME,
        position_cost=1.0,
        orientation_cost=1.0,
    )

    trajectory = []
    solved_poses = []
    q_current = q_init.copy()

    print(f"Solving IK for {len(gripper_poses)} poses...")

    for i, (pos, quat) in enumerate(gripper_poses):
        # Build target pose
        rot_matrix = R.from_quat(quat).as_matrix()
        target_pose = pin.SE3(rot_matrix, np.array(pos))
        end_effector_task.set_target(target_pose)

        # Reset configuration
        config = pink.Configuration(model, data, q_current)

        # Solve IK iteratively (tighter convergence for smoother trajectory)
        max_iters = 500
        ik_dt = 1 / max_iters
        tol = 1e-7

        for _ in range(max_iters):

            postural_task = pink.tasks.PostureTask(
                cost=1e-2,  # Adjust this: higher = stiffer/smaller movements
            )
            postural_task.set_target(q_current)

            velocity = solve_ik(
                config,
                [end_effector_task],
                # [end_effector_task, postural_task],
                ik_dt,
                solver="quadprog",
            )

            q = pin.integrate(model, config.q, velocity * ik_dt)
            config = pink.Configuration(model, data, q)

            current_pose = config.get_transform_frame_to_world(END_EFFECTOR_FRAME)
            pos_error = np.linalg.norm(current_pose.translation - target_pose.translation)
            rot_error = np.linalg.norm(pin.log3(current_pose.rotation.T @ target_pose.rotation))

            if pos_error < tol and rot_error < tol:
                break

        trajectory.append(q.copy())
        q_current = q

        # Store the solved end-effector pose
        solved_se3 = config.get_transform_frame_to_world(END_EFFECTOR_FRAME)
        solved_pos = solved_se3.translation.copy()
        solved_quat = R.from_matrix(solved_se3.rotation).as_quat()
        solved_poses.append((solved_pos, solved_quat))

        if (i + 1) % 50 == 0:
            print(f"  Solved {i + 1}/{len(gripper_poses)} poses")

    print("IK solving complete.")

    return trajectory, solved_poses


def plot_trajectory_comparison(gripper_poses, solved_poses, dt):
    """Plot ground truth vs solved trajectories for visual comparison.

    Args:
        gripper_poses: List of (position, quaternion) tuples - ground truth targets
        solved_poses: List of (position, quaternion) tuples - solved IK results
        dt: Time step between poses
    """
    n_poses = len(gripper_poses)
    times = np.arange(n_poses) * dt

    # Extract positions
    target_positions = np.array([p[0] for p in gripper_poses])
    solved_positions = np.array([p[0] for p in solved_poses])

    # Extract orientations (convert to Euler for visualization)
    target_eulers = np.array([R.from_quat(p[1]).as_euler('xyz', degrees=True) for p in gripper_poses])
    solved_eulers = np.array([R.from_quat(p[1]).as_euler('xyz', degrees=True) for p in solved_poses])

    # Create figure with 3 rows: XYZ position, RPY orientation, 3D trajectory
    fig = plt.figure(figsize=(16, 14))

    # ============ Row 1: Position comparison (3 subplots for X, Y, Z) ============
    ax_x = fig.add_subplot(3, 3, 1)
    ax_x.plot(times, target_positions[:, 0], 'b-', label='Ground Truth', linewidth=1.5)
    ax_x.plot(times, solved_positions[:, 0], 'r--', label='IK Solved', linewidth=1.5, alpha=0.8)
    ax_x.set_ylabel('X Position [m]')
    ax_x.set_title('X Position Comparison')
    ax_x.legend(loc='upper right')
    ax_x.grid(True, alpha=0.3)

    ax_y = fig.add_subplot(3, 3, 2)
    ax_y.plot(times, target_positions[:, 1], 'b-', label='Ground Truth', linewidth=1.5)
    ax_y.plot(times, solved_positions[:, 1], 'r--', label='IK Solved', linewidth=1.5, alpha=0.8)
    ax_y.set_ylabel('Y Position [m]')
    ax_y.set_title('Y Position Comparison')
    ax_y.legend(loc='upper right')
    ax_y.grid(True, alpha=0.3)

    ax_z = fig.add_subplot(3, 3, 3)
    ax_z.plot(times, target_positions[:, 2], 'b-', label='Ground Truth', linewidth=1.5)
    ax_z.plot(times, solved_positions[:, 2], 'r--', label='IK Solved', linewidth=1.5, alpha=0.8)
    ax_z.set_ylabel('Z Position [m]')
    ax_z.set_title('Z Position Comparison')
    ax_z.legend(loc='upper right')
    ax_z.grid(True, alpha=0.3)

    # ============ Row 2: Orientation comparison (3 subplots for Roll, Pitch, Yaw) ============
    ax_roll = fig.add_subplot(3, 3, 4)
    ax_roll.plot(times, target_eulers[:, 0], 'b-', label='Ground Truth', linewidth=1.5)
    ax_roll.plot(times, solved_eulers[:, 0], 'r--', label='IK Solved', linewidth=1.5, alpha=0.8)
    ax_roll.set_xlabel('Time [s]')
    ax_roll.set_ylabel('Roll [deg]')
    ax_roll.set_title('Roll Comparison')
    ax_roll.legend(loc='upper right')
    ax_roll.grid(True, alpha=0.3)

    ax_pitch = fig.add_subplot(3, 3, 5)
    ax_pitch.plot(times, target_eulers[:, 1], 'b-', label='Ground Truth', linewidth=1.5)
    ax_pitch.plot(times, solved_eulers[:, 1], 'r--', label='IK Solved', linewidth=1.5, alpha=0.8)
    ax_pitch.set_xlabel('Time [s]')
    ax_pitch.set_ylabel('Pitch [deg]')
    ax_pitch.set_title('Pitch Comparison')
    ax_pitch.legend(loc='upper right')
    ax_pitch.grid(True, alpha=0.3)

    ax_yaw = fig.add_subplot(3, 3, 6)
    ax_yaw.plot(times, target_eulers[:, 2], 'b-', label='Ground Truth', linewidth=1.5)
    ax_yaw.plot(times, solved_eulers[:, 2], 'r--', label='IK Solved', linewidth=1.5, alpha=0.8)
    ax_yaw.set_xlabel('Time [s]')
    ax_yaw.set_ylabel('Yaw [deg]')
    ax_yaw.set_title('Yaw Comparison')
    ax_yaw.legend(loc='upper right')
    ax_yaw.grid(True, alpha=0.3)

    # ============ Row 3: 3D trajectory visualization ============
    ax_3d = fig.add_subplot(3, 3, 7, projection='3d')
    ax_3d.plot(target_positions[:, 0], target_positions[:, 1], target_positions[:, 2],
               'b-', label='Ground Truth', linewidth=2)
    ax_3d.plot(solved_positions[:, 0], solved_positions[:, 1], solved_positions[:, 2],
               'r--', label='IK Solved', linewidth=2, alpha=0.8)
    # Mark start and end points
    ax_3d.scatter(*target_positions[0], c='green', s=100, marker='o', label='Start')
    ax_3d.scatter(*target_positions[-1], c='purple', s=100, marker='s', label='End')
    ax_3d.set_xlabel('X [m]')
    ax_3d.set_ylabel('Y [m]')
    ax_3d.set_zlabel('Z [m]')
    ax_3d.set_title('3D Trajectory Comparison')
    ax_3d.legend(loc='upper right')

    # ============ Row 3: Position error magnitude over time ============
    pos_errors = np.linalg.norm(solved_positions - target_positions, axis=1) * 1000  # in mm
    ax_pos_err = fig.add_subplot(3, 3, 8)
    ax_pos_err.plot(times, pos_errors, 'k-', linewidth=1.5)
    ax_pos_err.fill_between(times, 0, pos_errors, alpha=0.3)
    ax_pos_err.set_xlabel('Time [s]')
    ax_pos_err.set_ylabel('Position Error [mm]')
    ax_pos_err.set_title(f'Position Error Magnitude (Mean: {np.mean(pos_errors):.3f} mm, Max: {np.max(pos_errors):.3f} mm)')
    ax_pos_err.grid(True, alpha=0.3)

    # ============ Row 3: Orientation error magnitude over time ============
    orient_errors = []
    for i in range(n_poses):
        target_rot = R.from_quat(gripper_poses[i][1])
        solved_rot = R.from_quat(solved_poses[i][1])
        # Compute relative rotation and get angle
        rel_rot = target_rot.inv() * solved_rot
        angle = np.abs(rel_rot.magnitude()) * 180 / np.pi  # in degrees
        orient_errors.append(angle)
    orient_errors = np.array(orient_errors)

    ax_orient_err = fig.add_subplot(3, 3, 9)
    ax_orient_err.plot(times, orient_errors, 'k-', linewidth=1.5)
    ax_orient_err.fill_between(times, 0, orient_errors, alpha=0.3)
    ax_orient_err.set_xlabel('Time [s]')
    ax_orient_err.set_ylabel('Orientation Error [deg]')
    ax_orient_err.set_title(f'Orientation Error Magnitude (Mean: {np.mean(orient_errors):.4f}°, Max: {np.max(orient_errors):.4f}°)')
    ax_orient_err.grid(True, alpha=0.3)

    fig.suptitle('Ground Truth vs Pink IK Solved Trajectory Comparison', fontsize=14, fontweight='bold')
    fig.tight_layout()

    # Print accuracy summary
    print("\n" + "=" * 60)
    print("TRAJECTORY COMPARISON SUMMARY")
    print("=" * 60)
    print(f"Number of poses: {n_poses}")
    print(f"Duration: {times[-1]:.2f} s")
    print(f"\nPosition Accuracy:")
    print(f"  Mean error: {np.mean(pos_errors):.4f} mm")
    print(f"  Max error:  {np.max(pos_errors):.4f} mm")
    print(f"  Std dev:    {np.std(pos_errors):.4f} mm")
    print(f"\nOrientation Accuracy:")
    print(f"  Mean error: {np.mean(orient_errors):.6f} deg")
    print(f"  Max error:  {np.max(orient_errors):.6f} deg")
    print(f"  Std dev:    {np.std(orient_errors):.6f} deg")

    plt.show()


def plot_ik_error(gripper_poses, solved_poses, dt):
    """Plot the difference between ground truth target poses and solved IK positions.

    Args:
        gripper_poses: List of (position, quaternion) tuples - ground truth targets
        solved_poses: List of (position, quaternion) tuples - solved IK results
        dt: Time step between poses
    """
    n_poses = len(gripper_poses)
    times = np.arange(n_poses) * dt

    # Extract positions and compute errors
    target_positions = np.array([p[0] for p in gripper_poses])
    solved_positions = np.array([p[0] for p in solved_poses])
    pos_errors = solved_positions - target_positions

    # Extract orientations and compute errors (in Euler angles)
    target_eulers = np.array([R.from_quat(p[1]).as_euler('xyz', degrees=True) for p in gripper_poses])
    solved_eulers = np.array([R.from_quat(p[1]).as_euler('xyz', degrees=True) for p in solved_poses])

    # Compute orientation errors (handling angle wrapping)
    orient_errors = solved_eulers - target_eulers
    # Wrap angles to [-180, 180]
    orient_errors = np.mod(orient_errors + 180, 360) - 180

    # Create figure with 2 rows: position errors and orientation errors
    fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

    # Position errors (in mm for better readability)
    ax_pos = axes[0]
    ax_pos.plot(times, pos_errors[:, 0] * 1000, label='X error', linewidth=1.5, color='r')
    ax_pos.plot(times, pos_errors[:, 1] * 1000, label='Y error', linewidth=1.5, color='g')
    ax_pos.plot(times, pos_errors[:, 2] * 1000, label='Z error', linewidth=1.5, color='b')
    ax_pos.set_ylabel('Position Error [mm]')
    ax_pos.set_title('IK Position Error (Solved - Target)')
    ax_pos.legend(loc='upper right')
    ax_pos.grid(True, alpha=0.3)
    ax_pos.axhline(y=0, color='k', linestyle='-', linewidth=0.5)

    # Add RMS error annotation
    pos_rms = np.sqrt(np.mean(pos_errors**2, axis=0)) * 1000
    ax_pos.text(0.02, 0.98, f'RMS: X={pos_rms[0]:.3f}mm, Y={pos_rms[1]:.3f}mm, Z={pos_rms[2]:.3f}mm',
                transform=ax_pos.transAxes, fontsize=9, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # Orientation errors (in degrees)
    ax_orient = axes[1]
    ax_orient.plot(times, orient_errors[:, 0], label='Roll error', linewidth=1.5, color='r')
    ax_orient.plot(times, orient_errors[:, 1], label='Pitch error', linewidth=1.5, color='g')
    ax_orient.plot(times, orient_errors[:, 2], label='Yaw error', linewidth=1.5, color='b')
    ax_orient.set_xlabel('Time [s]')
    ax_orient.set_ylabel('Orientation Error [deg]')
    ax_orient.set_title('IK Orientation Error (Solved - Target)')
    ax_orient.legend(loc='upper right')
    ax_orient.grid(True, alpha=0.3)
    ax_orient.axhline(y=0, color='k', linestyle='-', linewidth=0.5)

    # Add RMS error annotation
    orient_rms = np.sqrt(np.mean(orient_errors**2, axis=0))
    ax_orient.text(0.02, 0.98, f'RMS: Roll={orient_rms[0]:.4f}°, Pitch={orient_rms[1]:.4f}°, Yaw={orient_rms[2]:.4f}°',
                   transform=ax_orient.transAxes, fontsize=9, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    fig.tight_layout()

    # Also create a combined 6-DOF plot
    fig2, ax_combined = plt.subplots(figsize=(14, 6))

    # Plot all 6 dimensions on the same axis with different y-scales
    ax_combined.plot(times, pos_errors[:, 0] * 1000, label='X [mm]', linewidth=1.5, linestyle='-', color='r')
    ax_combined.plot(times, pos_errors[:, 1] * 1000, label='Y [mm]', linewidth=1.5, linestyle='-', color='g')
    ax_combined.plot(times, pos_errors[:, 2] * 1000, label='Z [mm]', linewidth=1.5, linestyle='-', color='b')

    # Create secondary y-axis for orientation
    ax_orient2 = ax_combined.twinx()
    ax_orient2.plot(times, orient_errors[:, 0], label='Roll [deg]', linewidth=1.5, linestyle='--', color='r', alpha=0.7)
    ax_orient2.plot(times, orient_errors[:, 1], label='Pitch [deg]', linewidth=1.5, linestyle='--', color='g', alpha=0.7)
    ax_orient2.plot(times, orient_errors[:, 2], label='Yaw [deg]', linewidth=1.5, linestyle='--', color='b', alpha=0.7)

    ax_combined.set_xlabel('Time [s]')
    ax_combined.set_ylabel('Position Error [mm]')
    ax_orient2.set_ylabel('Orientation Error [deg]')
    ax_combined.set_title('IK Error: Ground Truth vs Solved (Solid=Position, Dashed=Orientation)')

    # Combine legends
    lines1, labels1 = ax_combined.get_legend_handles_labels()
    lines2, labels2 = ax_orient2.get_legend_handles_labels()
    ax_combined.legend(lines1 + lines2, labels1 + labels2, loc='upper right', ncol=2)

    ax_combined.grid(True, alpha=0.3)
    ax_combined.axhline(y=0, color='k', linestyle='-', linewidth=0.5)

    fig2.tight_layout()

    plt.show()

    # Print summary statistics
    print("\n" + "=" * 60)
    print("IK ERROR STATISTICS")
    print("=" * 60)
    print(f"Position Error (mm):")
    print(f"  X: min={pos_errors[:, 0].min()*1000:.4f}, max={pos_errors[:, 0].max()*1000:.4f}, RMS={pos_rms[0]:.4f}")
    print(f"  Y: min={pos_errors[:, 1].min()*1000:.4f}, max={pos_errors[:, 1].max()*1000:.4f}, RMS={pos_rms[1]:.4f}")
    print(f"  Z: min={pos_errors[:, 2].min()*1000:.4f}, max={pos_errors[:, 2].max()*1000:.4f}, RMS={pos_rms[2]:.4f}")
    print(f"  Total RMS: {np.sqrt(np.sum(pos_rms**2)):.4f}")
    print(f"\nOrientation Error (deg):")
    print(f"  Roll:  min={orient_errors[:, 0].min():.6f}, max={orient_errors[:, 0].max():.6f}, RMS={orient_rms[0]:.6f}")
    print(f"  Pitch: min={orient_errors[:, 1].min():.6f}, max={orient_errors[:, 1].max():.6f}, RMS={orient_rms[1]:.6f}")
    print(f"  Yaw:   min={orient_errors[:, 2].min():.6f}, max={orient_errors[:, 2].max():.6f}, RMS={orient_rms[2]:.6f}")


def visualize_trajectory(robot, trajectory, hand_grasp, dt):
    """Visualize the trajectory in meshcat."""
    if not HAS_MESHCAT:
        print("Cannot visualize: meshcat not available")
        return

    print("\nInitializing meshcat visualizer...")
    print("Open in browser: http://127.0.0.1:7000/static/")

    viz = MeshcatVisualizer(robot.model, robot.collision_model, robot.visual_model)
    viz.initViewer(open=True)
    viz.loadViewerModel()

    # Display initial pose
    viz.display(trajectory[0])
    time.sleep(1.0)

    print("\n" + "=" * 60)
    print("TRAJECTORY VISUALIZATION")
    print("=" * 60)
    print(f"Total poses: {len(trajectory)}")
    print(f"Control frequency: {1/dt:.1f} Hz")
    print(f"Duration: {len(trajectory) * dt:.2f}s")

    print("\nPress Enter to start animation (or 'q' to quit)...")
    user_input = input()
    if user_input.lower() == 'q':
        return

    print("\nPlaying trajectory...")

    for i, q in enumerate(trajectory):
        print(f"Step {i+1}/{len(trajectory)}", end='\r')
        viz.display(q)
        time.sleep(dt)

    print("\n\nTrajectory playback complete!")
    print("Press Enter to replay, 's' for slow motion, or 'q' to quit...")

    while True:
        user_input = input()
        if user_input.lower() == 'q':
            break
        elif user_input.lower() == 's':
            print("Playing in slow motion (0.5x speed)...")
            for i, q in enumerate(trajectory):
                viz.display(q)
                time.sleep(dt * 2)
            print("Slow motion complete. Enter to replay, 's' for slow, 'q' to quit...")
        else:
            print("Replaying...")
            for i, q in enumerate(trajectory):
                viz.display(q)
                time.sleep(dt)
            print("Replay complete. Enter to replay, 's' for slow, 'q' to quit...")


def main():
    parser = argparse.ArgumentParser(description='Visualize Pink IK trajectory with local URDF')
    parser.add_argument('--urdf', type=str, default=URDF_PATH, help='URDF file path')
    parser.add_argument('--hdf5', type=str, default=HDF5_FILE, help='HDF5 file path')
    parser.add_argument('--freq', type=float, default=CTRL_FREQ, help='Control frequency (Hz)')
    parser.add_argument('--interp', type=int, default=5,
                        help='Number of frames to interpolate between waypoints (0=no interpolation)')
    parser.add_argument('--no-viz', action='store_true', help='Skip visualization, just solve IK')
    parser.add_argument('--plot-error', action='store_true',
                        help='Plot IK error (difference between target and solved poses)')
    parser.add_argument('--plot-comparison', action='store_true',
                        help='Plot trajectory comparison (ground truth vs IK solved)')
    parser.add_argument('--mesh-dirs', type=str, nargs='+', default=robot_filter_path,
                        help='Additional directories to search for mesh packages')
    parser.add_argument('--lowpass', action='store_true',
                        help='Apply low pass filter to smooth the trajectory')
    parser.add_argument('--lowpass-cutoff', type=float, default=2.0,
                        help='Low pass filter cutoff frequency in Hz (default: 2.0)')
    parser.add_argument('--lowpass-order', type=int, default=2,
                        help='Low pass filter order (default: 2)')
    args = parser.parse_args()

    dt = 1.0 / args.freq

    # Load robot from local URDF
    print("Loading FR3 robot from local URDF...")
    mesh_dirs = args.mesh_dirs
    print(f"mesh_dirs: {mesh_dirs}")
    robot = load_robot_from_urdf(args.urdf, mesh_package_dirs=mesh_dirs)

    print(f"Robot DOF: {robot.model.nq}")
    print(f"End effector: {END_EFFECTOR_FRAME}")

    # List available frames for debugging
    print("\nAvailable frames in model:")
    for i, frame in enumerate(robot.model.frames):
        print(f"  {i}: {frame.name}")

    # Load trajectory from HDF5
    print(f"\nLoading trajectory from {args.hdf5}...")
    hand_poses, hand_grasp = load_trajectory_from_hdf5(args.hdf5)
    print(f"Loaded {len(hand_poses)} poses")

    # Convert to gripper frame
    print("Converting poses to gripper frame...")
    gripper_poses = convert_poses_to_gripper_frame(hand_poses)

    # Solve IK using home joint position as initial config
    print(f"\nSolving IK (initial config from DPEvalConfig)...")
    trajectory, solved_poses = solve_trajectory_ik(robot, gripper_poses, HOME_JOINT_POSITION)

    # Apply low pass filter if requested
    if args.lowpass:
        print(f"\nApplying low pass filter (cutoff: {args.lowpass_cutoff} Hz, order: {args.lowpass_order})...")
        trajectory = lowpass_filter_trajectory(
            trajectory,
            cutoff_freq=args.lowpass_cutoff,
            sample_freq=args.freq,
            order=args.lowpass_order
        )
        print("Low pass filtering complete.")

    # Plot trajectory comparison if requested
    if args.plot_comparison:
        print("\nPlotting trajectory comparison...")
        plot_trajectory_comparison(gripper_poses, solved_poses, dt)

    # Plot IK error if requested
    if args.plot_error:
        print("\nPlotting IK error...")
        plot_ik_error(gripper_poses, solved_poses, dt)

    # Interpolate trajectory if requested
    if args.interp > 0:
        print(f"\nInterpolating trajectory (inserting {args.interp} frames between waypoints)...")
        original_len = len(trajectory)
        trajectory = interpolate_trajectory(trajectory, num_interp=args.interp)
        # Adjust dt for interpolated trajectory to maintain same overall duration
        dt = dt / (args.interp + 1)
        print(f"Trajectory: {original_len} -> {len(trajectory)} frames (effective freq: {1/dt:.1f} Hz)")

    # Print summary
    print("\n" + "=" * 60)
    print("TRAJECTORY SUMMARY")
    print("=" * 60)
    print(f"Total waypoints: {len(trajectory)}")
    print(f"Duration: {len(trajectory) * dt:.2f}s at {1/dt:.1f} Hz")

    # Visualize
    if not args.no_viz:
        visualize_trajectory(robot, trajectory, hand_grasp, dt)
    else:
        print("\nSkipping visualization (--no-viz)")


if __name__ == "__main__":
    main()
