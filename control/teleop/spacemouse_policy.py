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
    gripper_latched: float = 1.0    # held command: 1.0 open, 0.0 closed
    motion: np.ndarray = field(default_factory=lambda: np.zeros(6, dtype=np.float64))
    delta: np.ndarray = field(default_factory=lambda: np.zeros(6, dtype=np.float64))
    button: bool = False


@dataclass
class SpacemouseConfig:
    """Tuning for :class:`SpacemousePolicy`."""

    # Per-axis gain applied to the normalised spacemouse reading.
    # [x, y, z, roll, pitch, yaw] — signs match the device's frame.
    # The shared default. control/takeover.py builds on it too, so raising
    # it here rescales the server-side panel takeover as well -- per-script
    # gains belong in the script (see deploy/deploy_spacemouse.py, which
    # resolves its own per-control-space value).
    action_size: float = 0.003
    # Rotation gain, decoupled from translation. None reuses action_size (the
    # old coupled behaviour). Both are multiplied by action_scaling, so yaw
    # keeps its 3x relative to roll regardless.
    rotation_size: float | None = None
    action_scaling: np.ndarray = field(
        default_factory=lambda: np.array([1.0, 1.0, 1.0, 2.0, -1.0, -3.0])
    )
    deadzone: float = 0.1
    # How far the integrated target may run ahead of the measured fingertip
    # [m] / [rad]. The impedance controller lags its target under motion, so
    # without a leash the target outruns the arm while the puck is held and
    # the arm coasts the whole accumulated gap after release -- which feels
    # like momentum and grows linearly with action_size. None disables it.
    # The joint-space path already does this in control.ik.DifferentialIK.
    #
    # The two caps are NOT interchangeable, because rotation both moves
    # faster and settles slower than translation:
    #
    #   natural lead = speed * d/k     (steady-state impedance tracking lag)
    #   translation  0.4 m/s   * 40/500 = 32 mm
    #   yaw          1.2 rad/s * 10/60  = 0.20 rad    <- 3x gain, 2.1x slower
    #
    # (yaw is 3x from action_scaling; d/k from
    # config/control/default_cartesian_impedance.yaml, at action_size 0.008.)
    #
    # A cap does nothing unless it is BELOW the natural lead -- 0.20 rad looks
    # like a sane rotational counterpart to 0.02 m and is in fact inert.
    # These are a SPEED CEILING as well as a coast bound. Once the target
    # saturates the leash it sits a constant `cap` ahead, so the arm settles
    # at a steady
    #
    #     ceiling = cap / (d/k)        and        coast = cap
    #
    # i.e. coast = ceiling * d/k. The two cannot be traded against each other
    # here: for a given impedance profile, stopping distance is fixed by the
    # speed you want. Raising gain above the ceiling buys nothing at all --
    # max_lead_rad=0.05 pins yaw at 17 deg/s whatever action_scaling says.
    # The only way to get fast AND crisp is a smaller d/k, i.e. stiffer
    # rotation in config/control/default_cartesian_impedance.yaml.
    # With brake_on_release these are a SAFETY backstop only -- they stop the
    # target running away if the arm is blocked -- so they are set well above
    # anything normal teleop commands and no longer throttle speed.
    max_lead_m: float | None = 0.10
    max_lead_rad: float | None = 0.50
    # Stop dead when the puck is released, instead of coasting out the lead.
    #
    # A lead clamp cannot do this: it bounds the gap, and the gap IS the
    # commanded speed, so ceiling = cap/(d/k) and coast = cap are the same
    # knob -- fast and crisp are mutually exclusive under it.
    #
    # Re-anchoring the target onto the measured pose on the release edge
    # breaks that tie. The remaining position error becomes zero, so the
    # controller has nothing left to drive the arm with and it stops within
    # one settling time, whatever speed it was doing. Speed is then free to
    # be as high as the gain and the controller allow.
    #
    # Edge-triggered, not continuous: re-anchoring every idle tick would let
    # the target follow the arm if someone pushed it, leaving it limp. One
    # snap on release, then the target holds.
    brake_on_release: bool = True
    # Consecutive idle ticks before the brake fires. 1 stops the instant the
    # puck centres, which is what you want; but a signal that jitters across
    # the deadzone then cancels, on every falling edge, the lead built on the
    # rising one, which weakens slow hesitant input (measured: 4x less travel
    # at the threshold). Raise to 2-3 to debounce, at the cost of the arm
    # closing part of its lead first -- roughly 40% of it per extra tick.
    brake_release_ticks: int = 1
    # Lowest fingertip z that may be commanded [m]. Table safety.
    min_z: float = MIN_Z
    # Clip commanded positions into control.kinematics.EEF_BOUNDS.
    clamp_workspace: bool = True
    # Which spacemouse button toggles the gripper.
    gripper_button: int = 0
    # Enable the 'r' key to request a reset.
    keyboard_reset: bool = True

    @property
    def rot_gain(self) -> float:
        """Rotation gain, falling back to action_size when unset."""
        return self.action_size if self.rotation_size is None else self.rotation_size

    @property
    def scale(self) -> np.ndarray:
        gains = np.array([self.action_size] * 3 + [self.rot_gain] * 3,
                         dtype=np.float64)
        return gains * np.asarray(self.action_scaling, dtype=np.float64)


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

        # True on any tick where the target had to be pulled back to the arm.
        self.lead_clamped = False
        # Per-group activity, for the release edge. Translation and rotation
        # brake independently so rotation cannot drift while you translate.
        self._was_translating = False
        self._was_rotating = False
        self._idle_trans = 0
        self._idle_rot = 0

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
        self._was_translating = False
        self._was_rotating = False
        self._idle_trans = 0
        self._idle_rot = 0

    @property
    def idle(self) -> bool:
        """True when neither group has had input since the brake last fired.

        Joint-space callers need this: their command is derived from the
        measured joint values, so an idle tick would otherwise re-target the
        arm onto wherever it has drifted to and never resist the drift.
        """
        return not (self._was_translating or self._was_rotating)

    def _clamp_lead(self, position, rotation, obs):
        """Pull ``position``/``rotation`` back to within the configured lead."""
        self.lead_clamped = False
        max_m = self.config.max_lead_m
        max_rad = self.config.max_lead_rad
        if max_m is None and max_rad is None:
            return position, rotation

        pose = obs.fingertip_pose
        if max_m is not None:
            actual = np.asarray(pose.position, dtype=np.float64)
            lead = position - actual
            dist = float(np.linalg.norm(lead))
            if dist > max_m:
                position = actual + lead * (max_m / dist)
                self.lead_clamped = True

        if max_rad is not None:
            current = R.from_quat(pose.orientation.as_quat())
            turn = (rotation * current.inv()).as_rotvec()
            angle = float(np.linalg.norm(turn))
            if angle > max_rad:
                rotation = R.from_rotvec(turn * (max_rad / angle)) * current
                self.lead_clamped = True

        return position, rotation

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
            pressed = self._spacemouse.is_button_pressed(self.config.gripper_button)
            return Action(position=self._target_position.copy(),
                          quat_xyzw=self._target_rotation.as_quat(),
                          reset=True,
                          gripper_latched=self._gripper,
                          motion=np.zeros(6, dtype=np.float64),
                          delta=np.zeros(6, dtype=np.float64),
                          button=pressed)

        motion = self._spacemouse.get_motion_state_transformed()
        delta = motion * self.config.scale
        dx, dy, dz, droll, dpitch, dyaw = delta

        # Gripper toggles on the button's rising edge only.
        pressed = self._spacemouse.is_button_pressed(self.config.gripper_button)
        gripper = None
        if pressed and not self._prev_button:
            self._gripper = 1.0 - self._gripper
            gripper = self._gripper
        self._prev_button = pressed

        # The deadzone has already zeroed anything resting near centre, so a
        # group is "active" exactly when it still has a non-zero component.
        translating = bool(np.any(motion[:3]))
        rotating = bool(np.any(motion[3:]))

        # Integrate the deltas onto the running target. Integrating onto the
        # target (not the measured pose) keeps the command from being dragged
        # backwards by impedance-controller tracking error.
        position = self._target_position + np.array([dx, dy, dz])
        position[2] = max(position[2], self.config.min_z)
        if self.config.clamp_workspace:
            position = clamp_to_workspace(position)

        euler = self._target_rotation.as_euler("XYZ") + np.array([droll, dpitch, dyaw])
        rotation = R.from_euler("XYZ", euler)

        # Brake on the release edge: collapse the lead onto the arm so there
        # is no error left to coast out. Only on the edge -- while idle the
        # target is left alone so the arm holds its pose.
        self._idle_trans = 0 if translating else self._idle_trans + 1
        self._idle_rot = 0 if rotating else self._idle_rot + 1
        if self.config.brake_on_release:
            n = max(1, self.config.brake_release_ticks)
            pose = obs.fingertip_pose
            if self._was_translating and self._idle_trans == n:
                position = np.asarray(pose.position, dtype=np.float64).copy()
                self._was_translating = False
            if self._was_rotating and self._idle_rot == n:
                rotation = R.from_quat(pose.orientation.as_quat())
                self._was_rotating = False
        if translating:
            self._was_translating = True
        if rotating:
            self._was_rotating = True

        # Safety backstop: stop the target running away if the arm is
        # blocked. Stopping on release is the brake's job, not this one, so
        # the caps sit well above normal teleop and rarely fire.
        position, rotation = self._clamp_lead(position, rotation, obs)

        self._target_position = position
        self._target_rotation = rotation

        return Action(position=position.copy(),
                      quat_xyzw=rotation.as_quat(),
                      gripper=gripper,
                      gripper_latched=self._gripper,
                      motion=motion,
                      delta=delta,
                      button=pressed)
