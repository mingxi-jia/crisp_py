"""Pink IK Solver Server.

A simple HTTP server that solves inverse kinematics using Pink.
Run this in the pink_solver conda environment.

Usage (run from the repository root, so the `control` package is importable):
    conda activate pink_solver
    python -m control.pink_ik_server [--port 5002] [--urdf other.urdf]

Self-contained: the robot model is control/fr3_robot.urdf and mesh paths
resolve from the standard ROS share directories. No external toolbox.

API:
    POST /solve_ik
        Input: {"position": [x, y, z], "orientation": [qx, qy, qz, qw], "q_init": [...]}
        Output: {"q": [...], "success": true/false}

    POST /solve_trajectory
        Input: {"poses": [{"position": [...], "orientation": [...]}, ...], "q_init": [...]}
        Output: {"trajectory": [[...], ...], "success": true/false}
"""

import os
import argparse
import numpy as np
from flask import Flask, request, jsonify

import pinocchio as pin
import pink
from pink import solve_ik
from pink.tasks import FrameTask
from pink.barriers import BodySphericalBarrier
import sys

import time

from control.pink_realtime.util import load_robot_from_urdf

# Try to import robot_descriptions for fallback
try:
    from robot_descriptions.loaders.pinocchio import load_robot_description
    HAS_ROBOT_DESCRIPTIONS = True
except ImportError:
    HAS_ROBOT_DESCRIPTIONS = False

app = Flask(__name__)

# Global robot and configuration
robot = None
config = None
end_effector_task = None
posture_task_1 = None
self_collision_barrier = None

END_EFFECTOR_FRAME = "fr3_hand_tcp"  # End effector frame name for local URDF


# Home joint position for the 7 arm joints (reduced model)
HOME_JOINT_POSITION = np.array([
    0.0015795138042423453, 0.11460111156789562, 0.00012723805921852443,
    -1.9088541631957334, 0.007562769235242739, 2.16821183580947, 0.7848162419679248
])

GOOD_JOINT_POSITION_1 = np.array([
    0.04461413917090776, -0.48244150863625346, -0.12345185881715248, -2.7744763331266284, -1.9715285002710865, 2.6243887446139884, 1.3522823784989295
])

GOOD_JOINT_POSITION_2 = np.array([
    -0.22961019174013284, -0.4996252881560531, 0.12971274782612352, -2.8556380654427507, -0.39038999253732737, 3.2497209233852304, 0.19488255825328224
])

def setup_collision_barrier(robot):
    # 1. Define Frame for Link 2
    # We place the sphere center at the end of link 2
    l2_placement = pin.SE3.Identity()
    l2_placement.translation = np.array([0, 0, 0.15]) # Adjust offset to center of link
    
    frame_l2 = pin.Frame(
        "link2_barrier_frame",
        robot.model.getJointId("fr3_joint2"), # Parent joint
        robot.model.getFrameId("fr3_link2"),  # Parent frame
        l2_placement,
        pin.FrameType.OP_FRAME,
    )
    robot.model.addFrame(frame_l2)

    # 2. Define Frame for Link 5
    l5_placement = pin.SE3.Identity()
    l5_placement.translation = np.array([0, 0, 0.15]) # Adjust offset
    
    frame_l5 = pin.Frame(
        "link5_barrier_frame",
        robot.model.getJointId("fr3_joint5"), 
        robot.model.getFrameId("fr3_link5"),
        l5_placement,
        pin.FrameType.OP_FRAME,
    )
    robot.model.addFrame(frame_l5)

    # Re-generate robot data to include new frames
    robot.data = pin.Data(robot.model)

    # 3. Create the Barrier Task
    # d_min is the sum of the radii of the two virtual spheres
    self_collision_barrier = BodySphericalBarrier(
        ("link2_barrier_frame", "link5_barrier_frame"),
        d_min=0.10,      # Minimum distance in meters
        gain=10.0,      # How "hard" the barrier pushes back
    )
    
    return self_collision_barrier


def init_robot(urdf_path: str = None, mesh_dirs: list = None, ee_frame: str = None):
    """Initialize the robot model and Pink configuration.

    Args:
        urdf_path: Path to local URDF file. If None, uses robot_descriptions package.
        mesh_dirs: List of directories to search for mesh packages (for local URDF).
        ee_frame: End effector frame name. If None, uses default based on URDF source.
    """
    global robot, config, end_effector_task, posture_task_1, posture_task_2, posture_task_home, self_collision_barrier, END_EFFECTOR_FRAME

    if urdf_path is not None:
        # Load from local URDF file
        print(f"Loading FR3 robot from local URDF: {urdf_path}")
        robot = load_robot_from_urdf(urdf_path, mesh_package_dirs=mesh_dirs)
        END_EFFECTOR_FRAME = ee_frame or "fr3_hand_tcp"
    else:
        # Load from robot_descriptions package (fallback)
        if not HAS_ROBOT_DESCRIPTIONS:
            raise RuntimeError(
                "robot_descriptions package not available. "
                "Please provide a URDF path with --urdf or install robot_descriptions."
            )
        print("Loading FR3 robot from robot_descriptions...")
        robot = load_robot_description("fr3_mj_description")
        END_EFFECTOR_FRAME = ee_frame or "fr3_link7"

    # Build reduced model: lock all joints with index > 7 (gripper joints)
    joints_to_lock = [i for i in range(8, robot.model.njoints)]
    if joints_to_lock:
        q_ref = np.concatenate((HOME_JOINT_POSITION, np.zeros(robot.model.nq - 7)))
        robot.model, [robot.visual_model, robot.collision_model] = pin.buildReducedModel(
            robot.model, [robot.visual_model, robot.collision_model], joints_to_lock, q_ref)
        robot.data = robot.model.createData()
        print(f"Built reduced model: locked joints {joints_to_lock}, DOF: {robot.model.nq}")

    # Initialize configuration to home pose
    config = pink.Configuration(robot.model, robot.data, HOME_JOINT_POSITION)

    # Create end-effector task
    end_effector_task = FrameTask(
        END_EFFECTOR_FRAME,
        position_cost=1.0,
        orientation_cost=1.0,
    )
    
    posture_task_1 = pink.tasks.PostureTask(
        cost=1e-6,  # Keep this low so it doesn't fight the End Effector task
    )
    posture_task_1.set_target(GOOD_JOINT_POSITION_1)

    posture_task_2 = pink.tasks.PostureTask(
        cost=1e-6,  # Keep this low so it doesn't fight the End Effector task
    )
    posture_task_2.set_target(GOOD_JOINT_POSITION_2)

    posture_task_home = pink.tasks.PostureTask(
        cost=1e-6,  # Keep this low so it doesn't fight the End Effector task
    )
    posture_task_home.set_target(HOME_JOINT_POSITION)

    # Setup self-collision barrier
    self_collision_barrier = setup_collision_barrier(robot)

    print(f"Robot loaded. DOF: {robot.model.nq}")
    print(f"End effector frame: {END_EFFECTOR_FRAME}")

    # List all joints with their indices and DOF info
    print("\n" + "=" * 60)
    print("JOINTS USED FOR IK OPTIMIZATION:")
    print("=" * 60)
    for i in range(1, robot.model.njoints):  # Skip universe joint (index 0)
        joint = robot.model.joints[i]
        joint_name = robot.model.names[i]
        # Get joint limits
        idx_q = joint.idx_q
        nq = joint.nq
        if nq > 0:
            lower = robot.model.lowerPositionLimit[idx_q:idx_q+nq]
            upper = robot.model.upperPositionLimit[idx_q:idx_q+nq]
            print(f"  Joint {i}: {joint_name:30s} | idx_q={idx_q:2d} | nq={nq} | limits=[{lower[0]:.3f}, {upper[0]:.3f}]")
        else:
            print(f"  Joint {i}: {joint_name:30s} | idx_q={idx_q:2d} | nq={nq} (fixed)")
    print("=" * 60)

    # List available frames
    print("\nAvailable frames:")
    for i, frame in enumerate(robot.model.frames):
        print(f"  {i}: {frame.name}")


def solve_single_ik(position, orientation_quat, q_init=None, max_iters=1000, dt=0.001, tol=1e-6):
    """Solve IK for a single pose.

    Args:
        position: [x, y, z] target position
        orientation_quat: [qx, qy, qz, qw] target orientation (quaternion)
        q_init: Initial joint configuration (optional)
        max_iters: Maximum iterations
        dt: Integration timestep
        tol: Convergence tolerance

    Returns:
        q: Joint configuration
        success: Whether IK converged
        timing_stats: Dictionary with timing information
    """
    global config, end_effector_task, posture_task_2, posture_task_home

    timing_stats = {}
    total_start = time.time()

    # Reset configuration
    setup_start = time.time()
    if q_init is not None:
        q = np.array(q_init)
    else:
        q = HOME_JOINT_POSITION.copy()

    config = pink.Configuration(robot.model, robot.data, q)

    # Build target SE3 pose
    from scipy.spatial.transform import Rotation as R
    rot_matrix = R.from_quat(orientation_quat).as_matrix()
    target_pose = pin.SE3(rot_matrix, np.array(position))

    # Set target
    end_effector_task.set_target(target_pose)
    timing_stats['setup_time_ms'] = (time.time() - setup_start) * 1000

    # Solve IK iteratively
    solve_start = time.time()
    num_iters = 0
    for i in range(max_iters):
        num_iters = i + 1

        # Compute velocity
        velocity = solve_ik(
            config,
            [end_effector_task],
            # [end_effector_task, posture_task_home],
            dt,
            solver="quadprog",
        )

        # Integrate
        q = pin.integrate(robot.model, config.q, velocity * dt)
        config = pink.Configuration(robot.model, robot.data, q)

        # Check convergence
        current_pose = config.get_transform_frame_to_world(END_EFFECTOR_FRAME)
        pos_error = np.linalg.norm(current_pose.translation - target_pose.translation)
        rot_error = np.linalg.norm(pin.log3(current_pose.rotation.T @ target_pose.rotation))

        if pos_error < tol and rot_error < tol:
            timing_stats['solve_time_ms'] = (time.time() - solve_start) * 1000
            timing_stats['total_time_ms'] = (time.time() - total_start) * 1000
            timing_stats['num_iterations'] = num_iters
            timing_stats['pos_error'] = pos_error
            timing_stats['rot_error'] = rot_error
            return q.tolist(), True, timing_stats

    timing_stats['solve_time_ms'] = (time.time() - solve_start) * 1000
    timing_stats['total_time_ms'] = (time.time() - total_start) * 1000
    timing_stats['num_iterations'] = num_iters
    timing_stats['pos_error'] = pos_error
    timing_stats['rot_error'] = rot_error
    return q.tolist(), False, timing_stats


@app.route('/solve_ik', methods=['POST'])
def handle_solve_ik():
    """Solve IK for a single pose."""
    data = request.json

    position = data.get('position')
    orientation = data.get('orientation')  # [qx, qy, qz, qw]
    q_init = data.get('q_init', None)

    if position is None or orientation is None:
        return jsonify({'error': 'Missing position or orientation', 'success': False}), 400

    try:
        q, success, timing_stats = solve_single_ik(position, orientation, q_init)
        return jsonify({
            'q': q,
            'success': success,
            'timing': timing_stats
        })
    except Exception as e:
        return jsonify({'error': str(e), 'success': False}), 500


@app.route('/solve_trajectory', methods=['POST'])
def handle_solve_trajectory():
    """Solve IK for a trajectory of poses."""
    data = request.json

    poses = data.get('poses')  # List of {"position": [...], "orientation": [...]}
    q_init = data.get('q_init', None)

    if poses is None or len(poses) == 0:
        return jsonify({'error': 'Missing or empty poses', 'success': False}), 400

    try:
        total_start = time.time()
        trajectory = []
        per_pose_timing = []
        q_current = q_init
        all_success = True
        total_iterations = 0

        for i, pose in enumerate(poses):
            position = pose['position']
            orientation = pose['orientation']

            q, success, timing_stats = solve_single_ik(position, orientation, q_current)
            trajectory.append(q)
            per_pose_timing.append(timing_stats)
            total_iterations += timing_stats['num_iterations']
            q_current = q  # Use previous solution as initial guess for next

            if not success:
                all_success = False

        total_time_ms = (time.time() - total_start) * 1000
        solve_times = [t['solve_time_ms'] for t in per_pose_timing]

        timing_summary = {
            'total_time_ms': total_time_ms,
            'num_poses': len(poses),
            'avg_time_per_pose_ms': total_time_ms / len(poses) if poses else 0,
            'min_solve_time_ms': min(solve_times) if solve_times else 0,
            'max_solve_time_ms': max(solve_times) if solve_times else 0,
            'avg_solve_time_ms': sum(solve_times) / len(solve_times) if solve_times else 0,
            'total_iterations': total_iterations,
            'avg_iterations_per_pose': total_iterations / len(poses) if poses else 0,
        }

        return jsonify({
            'trajectory': trajectory,
            'success': all_success,
            'timing': timing_summary,
            'per_pose_timing': per_pose_timing
        })
    except Exception as e:
        return jsonify({'error': str(e), 'success': False}), 500


@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint."""
    return jsonify({'status': 'ok', 'end_effector': END_EFFECTOR_FRAME})


@app.route('/robot_info', methods=['GET'])
def robot_info():
    """Get robot information."""
    return jsonify({
        'nq': robot.model.nq,
        'nv': robot.model.nv,
        'end_effector': END_EFFECTOR_FRAME,
        'home_position': HOME_JOINT_POSITION.tolist(),
        'joint_names': [robot.model.names[i] for i in range(1, robot.model.njoints)],
    })


def main():
    # Default URDF path (same directory as this script)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_urdf = os.path.join(script_dir, "fr3_robot.urdf")

    parser = argparse.ArgumentParser(description='Pink IK Solver Server')
    parser.add_argument('--port', type=int, default=5002, help='Server port')
    parser.add_argument('--host', type=str, default='0.0.0.0', help='Server host')
    parser.add_argument('--urdf', type=str, default=None,
                        help='URDF to load (default: control/fr3_robot.urdf)')
    args = parser.parse_args()


    # The robot model is the URDF in this directory; mesh paths resolve from
    # the standard ROS share dirs, so no external toolbox is involved.
    ee_frame = "fr3_hand_tcp"
    urdf_path = args.urdf or default_urdf

    # Initialize robot
    init_robot(urdf_path=urdf_path, ee_frame=ee_frame)

    print(f"\nStarting Pink IK server on {args.host}:{args.port}")
    print("Endpoints:")
    print("  POST /solve_ik - Solve IK for single pose")
    print("  POST /solve_trajectory - Solve IK for trajectory")
    print("  GET  /health - Health check")
    print("  GET  /robot_info - Robot information")

    app.run(host=args.host, port=args.port, debug=False)


if __name__ == '__main__':
    main()
