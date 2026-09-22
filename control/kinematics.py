"""Gripper geometry and fingertip<->end-effector transforms.

Pure numpy/scipy: importing this must NOT pull in torch or the external
diffusion_policy checkout. The same math used to live only in
the old diffusion_transforms module, which imported
``diffusion_policy.model.common.rotation_transformer`` at module scope and so
cannot be imported by a plain teleop script.

Frame convention
----------------
``fingertip`` is the frame the policy and the training data speak: the
reported end-effector pose offset along the gripper's +z and rotated by the
mount's yaw. ``ee`` is what the robot driver takes. Use
:func:`ee_to_fingertip` on observations and :func:`fingertip_to_ee` on
actions.

The offset itself is tool geometry, not code: it lives in
``config/gripper_geometry.yaml`` so that swapping the gripper or its adapter
plate is a config edit rather than a patch. Point
``$CRISP_GRIPPER_GEOMETRY`` elsewhere to use a different file.
"""

import copy
import os
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from scipy.spatial.transform import Rotation as R

if TYPE_CHECKING:                     # pragma: no cover
    from crisp_py.robot import Pose

# NOTE: crisp_py.robot imports rclpy, so it must NOT be imported at runtime
# here. control/pink_ik_server.py reaches this module (via diffusion_constants)
# from the ROS-free `pink_solver` conda env. Annotations are strings; the
# functions below only deepcopy the Pose they are handed, never construct one.

GEOMETRY_FILE = Path(os.environ.get(
    "CRISP_GRIPPER_GEOMETRY",
    Path(__file__).resolve().parent.parent / "config" / "gripper_geometry.yaml"))


def load_fingertip_offset(path=None) -> tuple[np.ndarray, np.ndarray]:
    """Read the ee -> fingertip offset from YAML -> (xyz [m], rpy [rad]).

    Raises rather than defaulting: a silently assumed gripper offset would
    shift every observation and every action by centimetres, and the symptom
    would look like a badly behaved policy rather than a missing file.
    """
    import yaml

    path = Path(path or GEOMETRY_FILE)
    if not path.exists():
        raise FileNotFoundError(
            f"gripper geometry file not found: {path}. It holds the "
            f"ee -> fingertip offset that every observation and action passes "
            f"through; set $CRISP_GRIPPER_GEOMETRY if it lives elsewhere.")
    cfg = (yaml.safe_load(path.read_text()) or {}).get("fingertip") or {}
    try:
        xyz = np.asarray(cfg["offset_xyz"], dtype=np.float64)
        rpy = np.asarray(cfg["offset_rpy"], dtype=np.float64)
    except KeyError as e:
        raise KeyError(f"{path}: fingertip section needs {e} "
                       f"(offset_xyz and offset_rpy, both length 3)") from e
    if xyz.shape != (3,) or rpy.shape != (3,):
        raise ValueError(f"{path}: offset_xyz and offset_rpy must both be "
                         f"length 3, got {xyz.shape} and {rpy.shape}")
    return xyz, rpy


# Translation and rotation from the reported end-effector frame to the
# fingertip, as configured. Module-level so the cost is paid once.
FINGERTIP_OFFSET_XYZ, FINGERTIP_OFFSET_RPY = load_fingertip_offset()


def fingertip_offset_matrix(inverse: bool = False) -> np.ndarray:
    """The ee -> fingertip offset as a 4x4, for callers holding matrices
    rather than a crisp_py Pose (which needs rclpy to construct)."""
    mat = np.eye(4)
    mat[:3, :3] = R.from_euler("XYZ", FINGERTIP_OFFSET_RPY).as_matrix()
    mat[:3, 3] = FINGERTIP_OFFSET_XYZ
    return np.linalg.inv(mat) if inverse else mat


# Lowest z the FINGERTIP may be commanded to [m]. Table safety.
#
# Applied twice in SpacemousePolicy.predict_action -- directly as min_z, and
# again through EEF_BOUNDS below -- so this one constant sets both.
#
# It bounds the MODELLED fingertip, which only matches the real one while
# config/gripper_geometry.yaml is right. It was previously 0.02 against a
# tool modelled 46 mm short, so the physical tip actually reached ~ -26 mm
# and pressed into the table. Correcting the tool length to 209.5 mm made
# the clamp mean what it says and lifted the reachable floor by that 46 mm;
# 0.005 puts the fingertip 5 mm above z = 0 instead.
#
# Change the gripper geometry and this number changes meaning with it.
#
# NOTE this is enforced ONLY on the spacemouse teleop path. robot_server's
# servo/goto, control/takeover.py and the pi0.5 policy do not clamp z at all.
MIN_Z = 0.005
# Normalization constant for the gripper value.
GRIPPER_NORM_CONST = 0.05

# Workspace limits applied to commanded positions.
EEF_BOUNDS = {
    "x": (0.3, 0.8),
    "y": (-0.35, 0.35),
    "z": (MIN_Z, 0.61),
}


def _apply_offset(pose: "Pose", offset: np.ndarray) -> "Pose":
    """Right-multiply ``pose`` by the 4x4 rigid transform ``offset``."""
    mat = np.eye(4)
    mat[:3, :3] = pose.orientation.as_matrix()
    mat[:3, 3] = pose.position

    out = mat @ offset
    result = copy.deepcopy(pose)
    result.position = out[:3, 3]
    result.orientation = R.from_matrix(out[:3, :3])
    return result


def ee_to_fingertip(ee_pose: "Pose") -> "Pose":
    """Convert an end-effector pose (as the robot reports it) to fingertip frame."""
    return _apply_offset(ee_pose, fingertip_offset_matrix())


def fingertip_to_ee(fingertip_pose: "Pose") -> "Pose":
    """Convert a fingertip-frame pose to the end-effector pose to command.

    The true matrix inverse, not a negated offset: those agree only while the
    configured rotation is about a single axis, and the YAML is free to
    describe a mount that is not.
    """
    return _apply_offset(fingertip_pose, fingertip_offset_matrix(inverse=True))


def clamp_to_workspace(position: np.ndarray) -> np.ndarray:
    """Clip a position into :data:`EEF_BOUNDS`."""
    position = np.asarray(position, dtype=np.float64).copy()
    for axis, key in enumerate("xyz"):
        low, high = EEF_BOUNDS[key]
        position[axis] = np.clip(position[axis], low, high)
    return position
