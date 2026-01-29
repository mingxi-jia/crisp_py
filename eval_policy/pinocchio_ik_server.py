"""Pinocchio IK Solver Server.

A simple HTTP server that solves inverse kinematics using Pinocchio (CLIK method).
This provides the same API as pink_ik_server.py for performance comparison.

Usage:
    python pinocchio_ik_server.py [--port 5003]

API:
    POST /solve_ik
        Input: {"position": [x, y, z], "orientation": [qx, qy, qz, qw], "q_init": [...]}
        Output: {"q": [...], "success": true/false}

    POST /solve_trajectory
        Input: {"poses": [{"position": [...], "orientation": [...]}, ...], "q_init": [...]}
        Output: {"trajectory": [[...], ...], "success": true/false}
"""

import argparse
import numpy as np
from flask import Flask, request, jsonify

import pinocchio as pin
from scipy.spatial.transform import Rotation as R

from robot_descriptions.loaders.pinocchio import load_robot_description

app = Flask(__name__)

# Global robot and configuration
robot = None
data = None
END_EFFECTOR_FRAME = "fr3_link7"  # End effector frame name for FR3
end_effector_frame_id = None

# Home joint position from DPEvalConfig (valid for FR3 joint limits)
HOME_JOINT_POSITION = np.array([
    0.0015795138042423453, 0.11460111156789562, 0.00012723805921852443,
    -1.9088541631957334, 0.007562769235242739, 2.16821183580947, 0.7848162419679248
])


def init_robot():
    """Initialize the robot model."""
    global robot, data, end_effector_frame_id

    print("Loading FR3 robot description...")
    robot = load_robot_description("fr3_mj_description")
    data = robot.model.createData()

    # Get end-effector frame ID
    end_effector_frame_id = robot.model.getFrameId(END_EFFECTOR_FRAME)
    if end_effector_frame_id >= robot.model.nframes:
        raise ValueError(f"Frame '{END_EFFECTOR_FRAME}' not found in robot model")

    print(f"Robot loaded. DOF: {robot.model.nq}")
    print(f"End effector frame: {END_EFFECTOR_FRAME} (id: {end_effector_frame_id})")
    print(f"Joint names: {[robot.model.names[i] for i in range(1, robot.model.njoints)]}")


def solve_single_ik(position, orientation_quat, q_init=None, max_iters=200, dt=0.01,
                    tol=1e-6, damping=1e-4, position_weight=0.5, orientation_weight=4.0):
    """Solve IK for a single pose using CLIK (Closed-Loop Inverse Kinematics).

    Args:
        position: [x, y, z] target position
        orientation_quat: [qx, qy, qz, qw] target orientation (quaternion)
        q_init: Initial joint configuration (optional)
        max_iters: Maximum iterations
        dt: Integration timestep / step size
        tol: Convergence tolerance
        damping: Damping factor for numerical stability (damped least squares)
        position_weight: Weight for position error (matched to Pink)
        orientation_weight: Weight for orientation error (matched to Pink)

    Returns:
        q: Joint configuration
        success: Whether IK converged
    """
    global robot, data, end_effector_frame_id

    model = robot.model

    # Initialize q
    if q_init is not None:
        q = np.array(q_init, dtype=np.float64)
    else:
        q = HOME_JOINT_POSITION.copy()

    # Build target SE3 pose
    rot_matrix = R.from_quat(orientation_quat).as_matrix()
    target_pose = pin.SE3(rot_matrix, np.array(position))

    # Weight matrix (matches Pink's position_cost and orientation_cost)
    W = np.diag([position_weight] * 3 + [orientation_weight] * 3)

    # CLIK iteration
    for i in range(max_iters):
        # Forward kinematics
        pin.forwardKinematics(model, data, q)
        pin.updateFramePlacement(model, data, end_effector_frame_id)

        # Get current frame placement
        current_pose = data.oMf[end_effector_frame_id]

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
            return q.tolist(), True

        # Compute frame Jacobian (in LOCAL_WORLD_ALIGNED frame)
        J = pin.computeFrameJacobian(
            model, data, q, end_effector_frame_id,
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
            v = np.linalg.lstsq(J_weighted, error_weighted, rcond=None)[0]

        # Integrate velocity to get new configuration
        q = pin.integrate(model, q, v * dt)

        # Clip to joint limits
        q = np.clip(q, model.lowerPositionLimit, model.upperPositionLimit)

    # Return best solution even if not converged
    return q.tolist(), False


def solve_single_ik_nullspace(position, orientation_quat, q_init=None, max_iters=200,
                               dt=0.01, tol=1e-6, damping=1e-4,
                               position_weight=0.5, orientation_weight=4.0,
                               nullspace_gain=0.5):
    """Solve IK with nullspace optimization to stay close to initial config.

    This variant uses the nullspace of the Jacobian to minimize joint movement
    from the initial configuration while achieving the target pose.
    """
    global robot, data, end_effector_frame_id

    model = robot.model

    # Initialize q
    if q_init is not None:
        q = np.array(q_init, dtype=np.float64)
        q_ref = q.copy()  # Reference configuration for nullspace
    else:
        q = HOME_JOINT_POSITION.copy()
        q_ref = q.copy()

    # Build target SE3 pose
    rot_matrix = R.from_quat(orientation_quat).as_matrix()
    target_pose = pin.SE3(rot_matrix, np.array(position))

    # Weight matrix
    W = np.diag([position_weight] * 3 + [orientation_weight] * 3)

    # CLIK iteration with nullspace
    for i in range(max_iters):
        # Forward kinematics
        pin.forwardKinematics(model, data, q)
        pin.updateFramePlacement(model, data, end_effector_frame_id)

        # Get current frame placement
        current_pose = data.oMf[end_effector_frame_id]

        # Position error
        pos_error = target_pose.translation - current_pose.translation

        # Orientation error
        rot_error = pin.log3(current_pose.rotation.T @ target_pose.rotation)

        # Combined error
        error = np.concatenate([pos_error, rot_error])

        # Check convergence
        pos_error_norm = np.linalg.norm(pos_error)
        rot_error_norm = np.linalg.norm(rot_error)

        if pos_error_norm < tol and rot_error_norm < tol:
            return q.tolist(), True

        # Compute frame Jacobian
        J = pin.computeFrameJacobian(
            model, data, q, end_effector_frame_id,
            pin.ReferenceFrame.LOCAL_WORLD_ALIGNED
        )

        # Weighted Jacobian
        J_weighted = W @ J
        error_weighted = W @ error

        # Damped pseudo-inverse
        JJT = J_weighted @ J_weighted.T
        damped_JJT = JJT + damping * np.eye(6)
        J_pinv = J_weighted.T @ np.linalg.inv(damped_JJT)

        # Primary task velocity
        v_task = J_pinv @ error_weighted

        # Nullspace projector: N = I - J^+ J
        N = np.eye(model.nv) - J_pinv @ J_weighted

        # Nullspace velocity to return toward reference configuration
        q_diff = pin.difference(model, q, q_ref)
        v_null = N @ (nullspace_gain * q_diff)

        # Combined velocity
        v = v_task + v_null

        # Integrate
        q = pin.integrate(model, q, v * dt)

        # Clip to joint limits
        q = np.clip(q, model.lowerPositionLimit, model.upperPositionLimit)

    return q.tolist(), False


@app.route('/solve_ik', methods=['POST'])
def handle_solve_ik():
    """Solve IK for a single pose."""
    data = request.json

    position = data.get('position')
    orientation = data.get('orientation')  # [qx, qy, qz, qw]
    q_init = data.get('q_init', None)
    use_nullspace = data.get('use_nullspace', False)

    if position is None or orientation is None:
        return jsonify({'error': 'Missing position or orientation', 'success': False}), 400

    try:
        if use_nullspace:
            q, success = solve_single_ik_nullspace(position, orientation, q_init)
        else:
            q, success = solve_single_ik(position, orientation, q_init)
        return jsonify({'q': q, 'success': success})
    except Exception as e:
        return jsonify({'error': str(e), 'success': False}), 500


@app.route('/solve_trajectory', methods=['POST'])
def handle_solve_trajectory():
    """Solve IK for a trajectory of poses."""
    req_data = request.json

    poses = req_data.get('poses')  # List of {"position": [...], "orientation": [...]}
    q_init = req_data.get('q_init', None)
    use_nullspace = req_data.get('use_nullspace', False)

    if poses is None or len(poses) == 0:
        return jsonify({'error': 'Missing or empty poses', 'success': False}), 400

    try:
        trajectory = []
        q_current = q_init
        all_success = True

        solve_fn = solve_single_ik_nullspace if use_nullspace else solve_single_ik

        for pose in poses:
            position = pose['position']
            orientation = pose['orientation']

            q, success = solve_fn(position, orientation, q_current)
            trajectory.append(q)
            q_current = q  # Use previous solution as initial guess for next

            if not success:
                all_success = False

        return jsonify({
            'trajectory': trajectory,
            'success': all_success
        })
    except Exception as e:
        return jsonify({'error': str(e), 'success': False}), 500


@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint."""
    return jsonify({'status': 'ok', 'solver': 'pinocchio-clik', 'end_effector': END_EFFECTOR_FRAME})


@app.route('/robot_info', methods=['GET'])
def robot_info():
    """Get robot information."""
    return jsonify({
        'nq': robot.model.nq,
        'nv': robot.model.nv,
        'end_effector': END_EFFECTOR_FRAME,
        'home_position': HOME_JOINT_POSITION.tolist(),
        'joint_names': [robot.model.names[i] for i in range(1, robot.model.njoints)],
        'lower_limits': robot.model.lowerPositionLimit.tolist(),
        'upper_limits': robot.model.upperPositionLimit.tolist(),
    })


def main():
    parser = argparse.ArgumentParser(description='Pinocchio IK Solver Server (CLIK)')
    parser.add_argument('--port', type=int, default=5003, help='Server port')
    parser.add_argument('--host', type=str, default='0.0.0.0', help='Server host')
    args = parser.parse_args()

    init_robot()

    print(f"\nStarting Pinocchio IK server on {args.host}:{args.port}")
    print("Endpoints:")
    print("  POST /solve_ik - Solve IK for single pose")
    print("  POST /solve_trajectory - Solve IK for trajectory")
    print("  GET  /health - Health check")
    print("  GET  /robot_info - Robot information")
    print("\nOptional parameters:")
    print("  use_nullspace: bool - Use nullspace optimization (default: false)")

    app.run(host=args.host, port=args.port, debug=False)


if __name__ == '__main__':
    main()
