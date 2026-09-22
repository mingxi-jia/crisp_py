"""One shared tick of spacemouse teleoperation.

The policy owns how device motion feels; this driver owns how that target is
sent to either server control space.  Keeping this boundary shared is
important: standalone teleop and recording must not acquire subtly different
hold or IK behaviour.
"""

from dataclasses import dataclass

import numpy as np

from control.ik import DifferentialIK


@dataclass
class TeleopStep:
    """The command and server state produced by one teleoperation tick."""

    state: object
    action: object
    result: dict
    q: np.ndarray | None
    held: bool
    events: list[str]


class TeleopDriver:
    """Send a :class:`SpacemousePolicy` target to a ``RobotClient``."""

    def __init__(self, robot, policy, ctrl_space, ik=None):
        if ctrl_space not in ("cartesian", "joint"):
            raise ValueError("ctrl_space must be 'cartesian' or 'joint'")
        self.robot = robot
        self.policy = policy
        self.ctrl_space = ctrl_space
        self.ik = ik if ctrl_space == "joint" else None
        if ctrl_space == "joint" and self.ik is None:
            self.ik = DifferentialIK(frame="fingertip")
        self._was_held = False
        self._was_lagging = False
        self._was_at_joint_limit = False
        # Last joint command, latched while the operator is idle. See step().
        self._q_hold = None

    @property
    def lagging(self):
        return bool(self.ik and self.ik.lagging)

    @property
    def at_joint_limit(self):
        return bool(self.ik and self.ik.at_joint_limit)

    def __enter__(self):
        self.policy.start()
        self.reset_anchor()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.policy.stop()

    def reset_anchor(self):
        """Anchor the policy target at the robot's current fingertip pose."""
        self.policy.reset(self.robot.get_state().fingertip_pose)
        self._q_hold = None

    def step(self) -> TeleopStep:
        """Run one non-blocking control tick without printing anything."""
        state = self.robot.get_state()
        action = self.policy.predict_action(state)
        events = []

        # A reset is intentionally only a request.  The caller homes the arm,
        # then calls reset_anchor(), preserving deploy_spacemouse's behaviour.
        if action.reset:
            return TeleopStep(state, action, {}, None, self._was_held, events)

        q = None
        if self.ik is None:
            result = self.robot.servo(action.position, action.quat_xyzw)
        else:
            # DifferentialIK builds its command as `measured_q + dq`. Once the
            # operator lets go dq goes to zero, so the command collapses onto
            # the measurement -- an impedance controller with target == state
            # applies no restoring torque, the arm free-floats, and the
            # command chases it wherever it drifts. Latching the last command
            # instead gives the controller a fixed target with real error to
            # push against, which is what actually stops the arm.
            idle = getattr(self.policy, "idle", False)
            if idle and self._q_hold is not None:
                q = self._q_hold
            else:
                q = self.ik.solve(state.joint_values, action.position,
                                  action.quat_xyzw)
                self._q_hold = q
            result = self.robot.servo_joint(q)
            if self.ik.lagging and not self._was_lagging:
                events.append("lagging")
            if self.ik.at_joint_limit and not self._was_at_joint_limit:
                events.append("at_joint_limit")
            self._was_lagging = self.ik.lagging
            self._was_at_joint_limit = self.ik.at_joint_limit
        if action.gripper is not None:
            self.robot.set_gripper(action.gripper)

        held = bool(result.get("paused") or result.get("takeover"))
        if held and not self._was_held:
            events.append("held")
        elif self._was_held and not held:
            self.reset_anchor()
            events.append("resumed")
        self._was_held = held
        return TeleopStep(state, action, result, q, held, events)
