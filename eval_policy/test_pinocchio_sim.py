"""Visualize robot trajectory with Pinocchio IK solver (CLIK) before running on real robot.

This script:
1. Loads trajectory from HDF5 (same as test_pink_sim.py)
2. Solves IK using Pinocchio CLIK (locally, no server needed)
3. Visualizes the robot trajectory in meshcat (browser-based)

This mirrors test_pink_sim.py but uses pure Pinocchio for performance comparison.

Usage:
    python test_pinocchio_sim.py
    python test_pinocchio_sim.py --hdf5 /path/to/file.hdf5

Opens visualization at: http://127.0.0.1:7000/static/
"""

import time
import argparse
import numpy as np
import h5py

from scipy.spatial.transform import Rotation as R

import pinocchio as pin

from robot_descriptions.loaders.pinocchio import load_robot_description

# Try to import meshcat visualizer
try:
    from pinocchio.visualize import MeshcatVisualizer
    HAS_MESHCAT = True
except ImportError:
    HAS_MESHCAT = False
    print("WARNING: MeshcatVisualizer not available. Install with: pip install meshcat")

# Configuration
HDF5_FILE = "/media/mingxi/T7/XEMB_Experiment/coffee_prep/replay_hand_test/test_2.hdf5"
END_EFFECTOR_FRAME = "fr3_link7"  # End effector frame for FR3
CTRL_FREQ = 10.0  # Hz

# Fingertip to gripper offset
FINGER_HAND_OFFSET = 0.06

# Home joint position from DPEvalConfig (valid for FR3 joint limits)
HOME_JOINT_POSITION = np.array([
    0.0015795138042423453, 0.11460111156789562, 0.00012723805921852443,
    -1.9088541631957334, 0.007562769235242739, 2.16821183580947, 0.7848162419679248
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


def solve_single_ik_clik(model, data, frame_id, position, orientation_quat, q_init,
                          max_iters=100, dt=0.01, tol=1e-4, damping=1e-4,
                          position_weight=0.5, orientation_weight=4.0):
    """Solve IK for a single pose using CLIK (Closed-Loop Inverse Kinematics).

    Args:
        model: Pinocchio model
        data: Pinocchio data
        frame_id: End-effector frame ID
        position: [x, y, z] target position
        orientation_quat: [qx, qy, qz, qw] target orientation (quaternion)
        q_init: Initial joint configuration
        max_iters: Maximum iterations
        dt: Integration timestep / step size (matched to Pink's 1/max_iters)
        tol: Convergence tolerance
        damping: Damping factor for numerical stability
        position_weight: Weight for position error (matched to Pink)
        orientation_weight: Weight for orientation error (matched to Pink)

    Returns:
        q: Joint configuration
        success: Whether IK converged
    """
    q = np.array(q_init, dtype=np.float64)

    # Build target SE3 pose
    rot_matrix = R.from_quat(orientation_quat).as_matrix()
    target_pose = pin.SE3(rot_matrix, np.array(position))

    # Weight matrix (matches Pink's position_cost=0.5, orientation_cost=4.0)
    W = np.diag([position_weight] * 3 + [orientation_weight] * 3)

    # CLIK iteration
    for i in range(max_iters):
        # Forward kinematics
        pin.forwardKinematics(model, data, q)
        pin.updateFramePlacement(model, data, frame_id)

        # Get current frame placement
        current_pose = data.oMf[frame_id]

        # Position error
        pos_error = target_pose.translation - current_pose.translation

        # Orientation error using log map (rotation error in tangent space)
        rot_error = pin.log3(current_pose.rotation.T @ target_pose.rotation)

        # Combined error
        error = np.concatenate([pos_error, rot_error])

        # Check convergence
        pos_error_norm = np.linalg.norm(pos_error)
        rot_error_norm = np.linalg.norm(rot_error)

        if pos_error_norm < tol and rot_error_norm < tol:
            return q, True

        # Compute frame Jacobian (in LOCAL_WORLD_ALIGNED frame)
        J = pin.computeFrameJacobian(
            model, data, q, frame_id,
            pin.ReferenceFrame.LOCAL_WORLD_ALIGNED
        )

        # Weighted Jacobian and error for task prioritization
        J_weighted = W @ J
        error_weighted = W @ error

        # Damped least squares (Levenberg-Marquardt)
        # v = (J^T W J + lambda I)^{-1} J^T W e
        JtWJ = J_weighted.T @ J_weighted
        damped_JtWJ = JtWJ + damping * np.eye(model.nv)

        try:
            v = np.linalg.solve(damped_JtWJ, J_weighted.T @ error_weighted)
        except np.linalg.LinAlgError:
            # Fallback to pseudo-inverse if solve fails
            J_pinv = np.linalg.lstsq(J_weighted, error_weighted, rcond=None)[0]
            v = J_pinv

        # Integrate velocity to get new configuration
        q = pin.integrate(model, q, v * dt)

        # Clip to joint limits
        q = np.clip(q, model.lowerPositionLimit, model.upperPositionLimit)

    # Return best solution even if not converged
    return q, False


def solve_trajectory_ik(robot, gripper_poses, q_init):
    """Solve IK for entire trajectory using Pinocchio CLIK."""
    model = robot.model
    data = robot.model.createData()

    # Get end-effector frame ID
    frame_id = model.getFrameId(END_EFFECTOR_FRAME)
    if frame_id >= model.nframes:
        raise ValueError(f"Frame '{END_EFFECTOR_FRAME}' not found in robot model")

    trajectory = []
    q_current = q_init.copy()
    converged_count = 0

    # IK parameters matched to Pink (test_pink_sim.py)
    max_iters = 100
    ik_dt = 1.0 / max_iters  # 0.01, same as Pink
    tol = 1e-4
    damping = 1e-4
    position_weight = 0.5    # Matched to Pink's position_cost
    orientation_weight = 4.0  # Matched to Pink's orientation_cost

    print(f"Solving IK for {len(gripper_poses)} poses using Pinocchio CLIK...")
    print(f"  Parameters: max_iters={max_iters}, dt={ik_dt}, tol={tol}")
    print(f"  Weights: position={position_weight}, orientation={orientation_weight}")
    start_time = time.time()

    for i, (pos, quat) in enumerate(gripper_poses):
        q, success = solve_single_ik_clik(
            model, data, frame_id,
            pos, quat, q_current,
            max_iters=max_iters,
            dt=ik_dt,
            tol=tol,
            damping=damping,
            position_weight=position_weight,
            orientation_weight=orientation_weight
        )

        trajectory.append(q.copy())
        q_current = q

        if success:
            converged_count += 1

        if (i + 1) % 50 == 0:
            print(f"  Solved {i + 1}/{len(gripper_poses)} poses")

    elapsed = time.time() - start_time
    print(f"IK solving complete in {elapsed:.2f}s ({1000*elapsed/len(gripper_poses):.2f}ms per pose)")
    print(f"Convergence rate: {converged_count}/{len(gripper_poses)} ({100*converged_count/len(gripper_poses):.1f}%)")

    return trajectory


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
    print("TRAJECTORY VISUALIZATION (Pinocchio CLIK)")
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
    parser = argparse.ArgumentParser(description='Visualize Pinocchio CLIK IK trajectory')
    parser.add_argument('--hdf5', type=str, default=HDF5_FILE, help='HDF5 file path')
    parser.add_argument('--freq', type=float, default=CTRL_FREQ, help='Control frequency (Hz)')
    parser.add_argument('--interp', type=int, default=0,
                        help='Number of frames to interpolate between waypoints (0=no interpolation)')
    parser.add_argument('--no-viz', action='store_true', help='Skip visualization, just solve IK')
    args = parser.parse_args()

    dt = 1.0 / args.freq

    # Load robot from robot_descriptions package
    print("Loading FR3 robot...")
    robot = load_robot_description("fr3_mj_description")
    print(f"Robot DOF: {robot.model.nq}")
    print(f"End effector: {END_EFFECTOR_FRAME}")

    # Load trajectory from HDF5
    print(f"\nLoading trajectory from {args.hdf5}...")
    hand_poses, hand_grasp = load_trajectory_from_hdf5(args.hdf5)
    print(f"Loaded {len(hand_poses)} poses")

    # Convert to gripper frame
    print("Converting poses to gripper frame...")
    gripper_poses = convert_poses_to_gripper_frame(hand_poses)

    # Solve IK using home joint position as initial config
    print(f"\nSolving IK with Pinocchio CLIK (initial config from DPEvalConfig)...")
    trajectory = solve_trajectory_ik(robot, gripper_poses, HOME_JOINT_POSITION)

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
    print("TRAJECTORY SUMMARY (Pinocchio CLIK)")
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
