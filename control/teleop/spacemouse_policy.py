"""Spacemouse teleoperation as a policy.

The delta-integration, deadzone, scaling and gripper-toggle logic here used to
be copy-pasted across five controllers in the old diff_eval_utils package
(teleop, teleop_intv, test_teleop, intervention, intervention_party_host). This
is the single implementation, and it deliberately exposes the same interface a
policy client does::

    action = policy.predict_action(obs)

so a control loop can swap a human for a network without changing shape.

Poses are in *fingertip* frame throughout, matching the training data.
"""

import threading
import time
from dataclasses import dataclass, field

import numpy as np
from scipy.spatial.transform import Rotation as R

from control.kinematics import MIN_Z, clamp_to_workspace


@dataclass
class Action:
    """One command produced by a policy."""

    position: np.ndarray            # fingertip frame
    quat_xyzw: np.ndarray           # fingertip frame
    gripper: float | None = None    # 1.0 open, 0.0 closed; None = leave as is
    reset: bool = False             # caller should re-home before continuing


@dataclass
class SpacemouseConfig:
    """Tuning for :class:`SpacemousePolicy`."""

    # Per-axis gain applied to the normalised spacemouse reading.
    # [x, y, z, roll, pitch, yaw] — signs match the device's frame.
    action_size: float = 0.003
    action_scaling: np.ndarray = field(
        default_factory=lambda: np.array([1.0, 1.0, 1.0, 2.0, -1.0, -3.0])
    )
    deadzone: float = 0.1
    # Lowest fingertip z that may be commanded [m]. Table safety.
    min_z: float = MIN_Z
    # Clip commanded positions into control.kinematics.EEF_BOUNDS.
    clamp_workspace: bool = True
    # Which spacemouse button toggles the gripper.
    gripper_button: int = 0
    # Enable the 'r' key to request a reset.
    keyboard_reset: bool = True

    @property
    def scale(self) -> np.ndarray:
        return self.action_size * np.asarray(self.action_scaling, dtype=np.float64)


class SpacemousePolicy:
    """Turns spacemouse motion into absolute fingertip pose targets.

    Use as a context manager so the device thread and keyboard listener are
    cleaned up::

        with SpacemousePolicy() as policy:
            policy.reset(robot.get_state().fingertip_pose)
            action = policy.predict_action(state)
    """

    def __init__(self, config: SpacemouseConfig | None = None):
        self.config = config or SpacemouseConfig()

        self._spacemouse = None
        self._spacemouse_ctx = None
        self._keyboard_listener = None

        # Absolute target we integrate deltas onto. Set by reset().
        self._target_position = None
        self._target_rotation = None

        self._gripper = 1.0            # start open
        self._prev_button = False

        self._keys = {"r": False}
        self._key_lock = threading.Lock()

    # -- lifecycle -------------------------------------------------------
    def start(self):
        """Open the spacemouse and (optionally) the keyboard listener."""
        from crisp_py.spacemouse import Spacemouse

        self._spacemouse_ctx = Spacemouse(deadzone=self.config.deadzone)
        self._spacemouse = self._spacemouse_ctx.__enter__()
        if self.config.keyboard_reset:
            self._start_keyboard_listener()
        return self

    def stop(self):
        """Close the spacemouse and keyboard listener."""
        if self._keyboard_listener is not None:
            self._keyboard_listener.stop()
            self._keyboard_listener = None
        if self._spacemouse_ctx is not None:
            self._spacemouse_ctx.__exit__(None, None, None)
            self._spacemouse_ctx = None
            self._spacemouse = None

    def __enter__(self):
        return self.start()

    def __exit__(self, exc_type, exc, tb):
        self.stop()

    def _start_keyboard_listener(self):
        try:
            from pynput import keyboard
        except Exception as e:                       # headless / no X11
            print(f"[spacemouse] keyboard reset disabled: {e}")
            return

        def on_press(key):
            if getattr(key, "char", None) == "r":
                with self._key_lock:
                    self._keys["r"] = True

        self._keyboard_listener = keyboard.Listener(on_press=on_press)
        self._keyboard_listener.start()

    def _take_key(self, name: str) -> bool:
        """Read a key flag and clear it."""
        with self._key_lock:
            pressed = self._keys[name]
            self._keys[name] = False
        return pressed

    # -- policy ----------------------------------------------------------
    def reset(self, pose):
        """Anchor the integrated target on ``pose`` (a fingertip-frame Pose).

        Call this at startup and after every re-home, otherwise the first
        action will jump the robot back to a stale target.
        """
        self._target_position = np.asarray(pose.position, dtype=np.float64).copy()
        self._target_rotation = R.from_quat(pose.orientation.as_quat())
        self._prev_button = False

    def predict_action(self, obs) -> Action:
        """Read the device once and return the next absolute target.

        Args:
            obs: a RobotState/RobotObs. Only used to lazily anchor the target
                 if reset() was never called.

        Returns:
            An :class:`Action` in fingertip frame. ``gripper`` is set only on
            the frame where the button is pressed; ``reset`` is True on 'r'.
        """
        if self._spacemouse is None:
            raise RuntimeError("SpacemousePolicy.start() was not called")
        if self._target_position is None:
            self.reset(obs.fingertip_pose)

        if self._take_key("r"):
            return Action(position=self._target_position.copy(),
                          quat_xyzw=self._target_rotation.as_quat(),
                          reset=True)

        motion = self._spacemouse.get_motion_state_transformed()
        dx, dy, dz, droll, dpitch, dyaw = motion * self.config.scale

        # Gripper toggles on the button's rising edge only.
        pressed = self._spacemouse.is_button_pressed(self.config.gripper_button)
        gripper = None
        if pressed and not self._prev_button:
            self._gripper = 1.0 - self._gripper
            gripper = self._gripper
        self._prev_button = pressed

        # Integrate the deltas onto the running target. Integrating onto the
        # target (not the measured pose) keeps the command from being dragged
        # backwards by impedance-controller tracking error.
        position = self._target_position + np.array([dx, dy, dz])
        position[2] = max(position[2], self.config.min_z)
        if self.config.clamp_workspace:
            position = clamp_to_workspace(position)

        euler = self._target_rotation.as_euler("XYZ") + np.array([droll, dpitch, dyaw])
        rotation = R.from_euler("XYZ", euler)

        self._target_position = position
        self._target_rotation = rotation

        return Action(position=position.copy(),
                      quat_xyzw=rotation.as_quat(),
                      gripper=gripper)
