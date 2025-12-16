from enum import Enum
from dataclasses import dataclass
import numpy as np


class ControlMode(Enum):
    """Control mode for teleoperation."""
    DELTA_POSE = "delta_pose"      # Cartesian space: dx, dy, dz, droll, dpitch, dyaw
    JOINT_DELTA = "joint_delta"    # Joint space: dq1, dq2, ..., dq7 (reserved for future)


@dataclass
class ControlCommand:
    """
    Unified control command for teleoperation.

    Attributes:
        delta_pose: Cartesian pose delta [dx, dy, dz, droll, dpitch, dyaw]
        delta_joints: Joint position delta [dq1, dq2, ..., dq7] (reserved for future)
        gripper_toggle: Toggle gripper open/close
        reset_requested: Request reset to home position
        mode: Control mode (DELTA_POSE or JOINT_DELTA)
    """
    delta_pose: np.ndarray = None
    delta_joints: np.ndarray = None
    gripper_toggle: bool = False
    reset_requested: bool = False
    mode: ControlMode = ControlMode.DELTA_POSE
