"""The small joint-space IK step used by both ways of holding the stick.

This is deliberately one damped-least-squares step, not a full IK solve. It
runs every tick against a target only one tick of motion away (3 mm at the
default gain), so the linearisation the Jacobian assumes is exact enough. It
also stays on the current IK branch by construction; a full solve is free to
return a different elbow for neighbouring targets, which reads as the arm
snapping while the stick is barely moving.

Both server takeover and deploy_spacemouse need this conversion. Sharing it
means the arm does not feel different depending on which one is holding the
stick.
"""

import numpy as np
from scipy.spatial.transform import Rotation

from control.projection import Chain, damped_least_squares, frame_offset

# How far the Cartesian target is allowed to run ahead of the arm. The target
# advances at the commanded rate whether or not the joints keep up, which is
# what makes it feel direct; without a ceiling on the gap, a few seconds of
# pushing against a limit would leave the arm chasing a target metres away
# after the stick was released.
MAX_LEAD_M = 0.05
MAX_LEAD_RAD = 0.35


class DifferentialIK:
    """One bounded DLS step from a Cartesian target to joint positions."""

    def __init__(self, chain: Chain | None = None, frame: str = "fingertip",
                 damping: float = 0.05, max_step_rad: float = 0.05,
                 max_lead_m: float = MAX_LEAD_M,
                 max_lead_rad: float = MAX_LEAD_RAD,
                 scale_step: bool = True):
        self.chain = chain or Chain.from_urdf()
        self.offset = frame_offset(frame)
        self.frame = frame
        self.damping = float(damping)
        self.max_step_rad = float(max_step_rad)
        self.max_lead_m = float(max_lead_m)
        self.max_lead_rad = float(max_lead_rad)
        self.scale_step = bool(scale_step)
        self.lagging = False
        self.at_joint_limit = False

    def solve(self, q, position, quat_xyzw) -> np.ndarray:
        """Joint positions + a Cartesian target -> the joint target to send.

        A single damped-least-squares step towards the target rather than a
        full IK solve: this runs every tick, and the target is only ever a
        tick's motion away once it is being followed.
        """
        q = np.asarray(q, dtype=np.float64).ravel()[:self.chain.n_joints]
        current = self.chain.fk(q) @ self.offset

        error = np.empty(6)
        error[:3] = np.asarray(position, dtype=np.float64) - current[:3, 3]
        error[3:] = Rotation.from_matrix(
            Rotation.from_quat(quat_xyzw).as_matrix() @ current[:3, :3].T).as_rotvec()

        # Clamp the lead, not the target: the target keeps its own pace, but
        # the arm is never asked to make up more than this in one go.
        lead = np.linalg.norm(error[:3])
        if lead > self.max_lead_m:
            error[:3] *= self.max_lead_m / lead
        turn = np.linalg.norm(error[3:])
        if turn > self.max_lead_rad:
            error[3:] *= self.max_lead_rad / turn
        self.lagging = lead > self.max_lead_m or turn > self.max_lead_rad

        dq = damped_least_squares(self.chain.jacobian(q, self.offset),
                                  error, self.damping)

        # Bound the step. Clipping each joint independently changes the
        # DIRECTION of the resulting Cartesian motion, not just its size: the
        # saturated joints stop while the others keep their full share, so the
        # tip veers off the commanded line. Scaling the whole vector keeps the
        # joint-space direction, and so approximately the Cartesian one.
        #
        # Measured over 800 configurations inside EEF_BOUNDS, median / p95
        # angle between commanded and achieved motion:
        #
        #   action_size 0.008   clip  2.0 / 32 deg    scale  2.0 / 32 deg
        #   action_size 0.015   clip  2.7 / 52 deg    scale  2.5 / 41 deg
        #   action_size 0.030   clip 17.2 / 79 deg    scale  2.7 / 35 deg
        #
        # It costs speed where it bites (every joint slows, not just the
        # saturated one), which is the right trade for teleoperation: going
        # somewhere slowly beats going somewhere else quickly.
        if self.scale_step:
            biggest = float(np.max(np.abs(dq))) if dq.size else 0.0
            if biggest > self.max_step_rad:
                dq = dq * (self.max_step_rad / biggest)
        else:
            dq = np.clip(dq, -self.max_step_rad, self.max_step_rad)

        limits = self.chain.joint_limits
        target = np.clip(q + dq, limits[:, 0], limits[:, 1])
        self.at_joint_limit = not np.allclose(target, q + dq)
        return target
