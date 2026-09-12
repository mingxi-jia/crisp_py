"""Gripper geometry and fingertip<->end-effector transforms.

Pure numpy/scipy: importing this must NOT pull in torch or the external
diffusion_policy checkout. The same math used to live only in
the old diffusion_transforms module, which imported
``diffusion_policy.model.common.rotation_transformer`` at module scope and so
cannot be imported by a plain teleop script.

Frame convention
----------------
``fingertip`` is the frame the policy and the training data speak: offset
``FINGER_HAND_OFFSET`` along the gripper's +z and rotated by
``ROBOTIQ_ROTATION_OFFSET``. ``ee`` is what the robot driver takes.
Use :func:`ee_to_fingertip` on observations and
:func:`fingertip_to_ee` on actions.
"""

import copy
from typing import TYPE_CHECKING

import numpy as np
from scipy.spatial.transform import Rotation as R

if TYPE_CHECKING:                     # pragma: no cover
    from crisp_py.robot import Pose

# NOTE: crisp_py.robot imports rclpy, so it must NOT be imported at runtime
# here. control/pink_ik_server.py reaches this module (via diffusion_constants)
# from the ROS-free `pink_solver` conda env. Annotations are strings; the
# functions below only deepcopy the Pose they are handed, never construct one.

# Distance from finger tip to gripper base along the gripper z-axis [m].
FINGER_HAND_OFFSET = 0.06
# Yaw offset baked into the Robotiq mount.
ROBOTIQ_ROTATION_OFFSET = np.array([0.0, 0.0, np.pi / 4])
# Lowest z the end-effector may be commanded to [m]. Table safety.
MIN_Z = 0.02
# Normalization constant for the gripper value.
GRIPPER_NORM_CONST = 0.05

# Workspace limits applied to commanded positions.
EEF_BOUNDS = {
    "x": (0.3, 0.8),
    "y": (-0.35, 0.35),
    "z": (MIN_Z, 0.61),
}


def _apply_offset(pose: "Pose", xyz_offset: np.ndarray, euler_offset: np.ndarray) -> "Pose":
    """Right-multiply ``pose`` by the rigid transform (euler_offset, xyz_offset)."""
    mat = np.eye(4)
    mat[:3, :3] = pose.orientation.as_matrix()
    mat[:3, 3] = pose.position

    offset = np.eye(4)
    offset[:3, 3] = xyz_offset
    offset[:3, :3] = R.from_euler("XYZ", euler_offset).as_matrix()

    out = mat @ offset
    result = copy.deepcopy(pose)
    result.position = out[:3, 3]
    result.orientation = R.from_matrix(out[:3, :3])
    return result


def ee_to_fingertip(ee_pose: "Pose") -> "Pose":
    """Convert an end-effector pose (as the robot reports it) to fingertip frame."""
    return _apply_offset(
        ee_pose,
        np.array([0.0, 0.0, FINGER_HAND_OFFSET]),
        ROBOTIQ_ROTATION_OFFSET,
    )


def fingertip_to_ee(fingertip_pose: "Pose") -> "Pose":
    """Convert a fingertip-frame pose to the end-effector pose to command."""
    return _apply_offset(
        fingertip_pose,
        np.array([0.0, 0.0, -FINGER_HAND_OFFSET]),
        -1.0 * ROBOTIQ_ROTATION_OFFSET,
    )


def clamp_to_workspace(position: np.ndarray) -> np.ndarray:
    """Clip a position into :data:`EEF_BOUNDS`."""
    position = np.asarray(position, dtype=np.float64).copy()
    for axis, key in enumerate("xyz"):
        low, high = EEF_BOUNDS[key]
        position[axis] = np.clip(position[axis], low, high)
    return position
