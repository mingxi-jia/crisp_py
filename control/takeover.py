"""Taking the arm off a running policy with the spacemouse.

A rollout that is going wrong has to be interruptible without being stopped:
the policy should keep watching and planning while a human moves the arm, so
the rollout can be handed back rather than restarted.

The feel is not reimplemented here
----------------------------------
The deltas, the axis scaling, the deadzone, the running target and the gripper
toggle all come from control.teleop.SpacemousePolicy -- the same object
deploy/deploy_spacemouse.py drives. Writing that again produced an arm that
was sluggish (integrating onto the *measured* pose lets impedance tracking lag
drag every step backwards -- the reference integrates onto the target for
exactly this reason) and two rotation axes that felt wrong (increments to the
target's intrinsic XYZ euler angles are not a base-frame rotation vector).
Driving the reference removes both by construction.

The Cartesian target becomes *joint* targets through the shared
control.ik.DifferentialIK solver, which deploy_spacemouse also uses when it
talks to a joint-space server.

    SpacemousePolicy -> fingertip pose target -> dq = J^+ error -> set_target_joint

pi0.5 needs the server in joint space, where the Cartesian controller is
inactive and pose targets are dropped (robot_server.servo). Converting here
means the same controller stays active through an intervention, so handing
back is just: stop sending. The policy's joint deltas are applied to the
measured position, so it resumes from wherever the arm ended up.
"""

import threading
import time

import numpy as np
from control.ik import DifferentialIK
from control.kinematics import ee_to_fingertip
from control.projection import Chain
from control.teleop import SpacemousePolicy, resolve_spacemouse_config


class SpacemouseTakeover:
    """A thread that jogs the arm from the spacemouse, in joint space.

    Owns the device only while it is running, so an idle server does not hold
    it away from deploy_spacemouse.
    """

    def __init__(self, robot, gripper=None, chain: Chain | None = None,
                 frame: str = "fingertip", rate: float = 50.0,
                 control_space: str = "joint", speed: float = 1.0,
                 action_size: float | None = None,
                 max_step_rad: float = 0.05, damping: float = 0.05,
                 deadzone: float | None = None, gripper_button: int = 0):
        self.robot = robot
        self.gripper = gripper
        self.chain = chain or Chain.from_urdf()
        self.frame = frame
        self.rate = float(rate)
        self.control_space = control_space
        self.speed = float(speed)
        # A hard ceiling on one tick's joint motion. At 50 Hz, 0.05 rad is
        # 2.5 rad/s: enough to keep up with the target, slow enough that a
        # stuck stick or a singularity cannot fling the arm.
        self.max_step_rad = float(max_step_rad)
        self.damping = float(damping)
        self._ik = DifferentialIK(chain=self.chain, frame=frame,
                                  damping=self.damping,
                                  max_step_rad=self.max_step_rad)
        self.offset = self._ik.offset
        self.policy_config = resolve_spacemouse_config(
            control_space, ik=self._ik, action_size=action_size,
            deadzone=deadzone, keyboard_reset=False, speed=speed)
        self.policy_config.gripper_button = gripper_button
        self.action_size = self.policy_config.action_size
        self.deadzone = self.policy_config.deadzone
        self.gripper_button = gripper_button

        self._policy = None
        self._thread = None
        self._stop = threading.Event()
        self._error = None
        self._ticks = 0
        self._started_at = None

    # -- lifecycle -------------------------------------------------------
    @property
    def active(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start(self) -> dict:
        """Open the spacemouse and begin jogging. Idempotent."""
        if self.active:
            return self.status()
        self._stop.clear()
        self._error = None
        self._ticks = 0
        try:
            self._policy = SpacemousePolicy(self.policy_config)
            self._policy.start()
            # Anchor the running target on where the arm is now, or the first
            # tick would drive it to wherever the last intervention ended.
            self._policy.reset(self._fingertip_pose())
        except Exception as e:
            self._policy = None
            self._error = (f"could not start the spacemouse ({e}). Another "
                           f"process -- deploy_spacemouse, or a calibration "
                           f"run -- may be holding the device.")
            return self.status()

        self._started_at = time.time()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self.status()

    def stop(self) -> dict:
        """Stop jogging and release the device.

        Sends nothing on the way out: the impedance controller keeps the last
        joint target, so the arm holds where the human left it until the
        policy's next command moves it from there.
        """
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        if self._policy is not None:
            try:
                self._policy.stop()
            finally:
                self._policy = None
        return self.status()

    def status(self) -> dict:
        return {
            "active": self.active,
            "error": self._error,
            "frame": self.frame,
            "speed": self.speed,
            "ticks": self._ticks,
            "seconds": (None if self._started_at is None or not self.active
                        else round(time.time() - self._started_at, 1)),
            # The arm is not keeping up with the stick -- a joint limit, a
            # singularity, or simply asking for more than max_step_rad.
            "lagging": self._ik.lagging,
            "at_joint_limit": self._ik.at_joint_limit,
            "max_step_rad": self.max_step_rad,
        }

    # -- kinematics ------------------------------------------------------
    def _fingertip_pose(self):
        """The arm's current fingertip pose, in the frame teleop speaks."""
        return ee_to_fingertip(self.robot.end_effector_pose)

    def solve(self, q, position, quat_xyzw) -> np.ndarray:
        """Delegate one Cartesian-target step to the shared joint solver."""
        return self._ik.solve(q, position, quat_xyzw)

    # -- the loop --------------------------------------------------------
    def _run(self):
        period = 1.0 / self.rate
        while not self._stop.is_set():
            tick = time.perf_counter()
            try:
                q = self.robot.joint_values
                if q is not None and len(q) >= self.chain.n_joints:
                    action = self._policy.predict_action(_Obs(self._fingertip_pose()))
                    self.robot.set_target_joint(
                        self.solve(q, action.position, action.quat_xyzw))
                    if action.gripper is not None and self.gripper is not None:
                        self.gripper.set_target(float(action.gripper))
                    self._ticks += 1
            except Exception as e:
                self._error = f"{type(e).__name__}: {e}"
                break
            time.sleep(max(0.0, period - (time.perf_counter() - tick)))


class _Obs:
    """The one attribute SpacemousePolicy reads off an observation."""

    def __init__(self, fingertip_pose):
        self.fingertip_pose = fingertip_pose
