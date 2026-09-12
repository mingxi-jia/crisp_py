"""Client for robot_server.py.

The whole robot surface a control loop needs, as plain function calls:

    robot = RobotClient("http://localhost:7000")
    state = robot.get_state()          # cheap, poll this in the loop
    robot.servo(position, quat_xyzw)   # non-blocking target
    robot.set_gripper(0.0)
    robot.home()

Poses cross this boundary in *fingertip* frame — the frame the policy and the
training data speak. The client applies the gripper offset before it hits the
wire, so the server stays a plain end-effector pose executor. Pass
``fingertip=False`` to bypass the conversion and command raw EE poses.
"""

from dataclasses import dataclass, field

import numpy as np
import requests
from scipy.spatial.transform import Rotation as R

from control.kinematics import ee_to_fingertip, fingertip_to_ee
from control.robot_io import decode_array
from crisp_py.robot import Pose


@dataclass
class RobotState:
    """Proprioception returned by :meth:`RobotClient.get_state`."""

    timestamp: float
    ee_position: np.ndarray          # end-effector frame
    ee_quat_xyzw: np.ndarray
    joint_values: np.ndarray
    gripper_value: float | None
    gripper_state: np.ndarray | None = None
    ft_wrench: np.ndarray | None = None

    @property
    def ee_pose(self) -> Pose:
        """End-effector pose as reported by the robot."""
        return Pose(position=np.asarray(self.ee_position, dtype=np.float64),
                    orientation=R.from_quat(self.ee_quat_xyzw))

    @property
    def fingertip_pose(self) -> Pose:
        """End-effector pose converted into fingertip frame."""
        return ee_to_fingertip(self.ee_pose)


@dataclass
class CameraFrame:
    """One camera's latest RGB-D pair. Either field may be None."""

    rgb: np.ndarray | None = None
    depth: np.ndarray | None = None      # uint16, millimetres
    age_s: float | None = None           # seconds since the frame arrived
    stale: bool = False                  # server judged it too old to use


@dataclass
class RobotObs(RobotState):
    """:class:`RobotState` plus camera frames, from :meth:`RobotClient.get_obs`.

    ``cameras`` is keyed by whatever the server is actually watching, so a
    2-camera launch yields two entries rather than a fixed-size list.
    """

    cameras: dict[str, CameraFrame] = field(default_factory=dict)
    camera_names: list[str] = field(default_factory=list)
    inhand_camera: str | None = None

    def rgb(self, name: str) -> np.ndarray | None:
        frame = self.cameras.get(name)
        return None if frame is None else frame.rgb

    def depth(self, name: str) -> np.ndarray | None:
        frame = self.cameras.get(name)
        return None if frame is None else frame.depth

    @property
    def inhand_rgb(self) -> np.ndarray | None:
        return self.rgb(self.inhand_camera) if self.inhand_camera else None

    @property
    def inhand_depth(self) -> np.ndarray | None:
        return self.depth(self.inhand_camera) if self.inhand_camera else None

    @property
    def ready_cameras(self) -> list[str]:
        """Cameras that delivered a fresh image."""
        return [n for n, f in self.cameras.items() if f.rgb is not None]

    @property
    def stale_cameras(self) -> list[str]:
        """Cameras whose newest frame was too old to use."""
        return [n for n, f in self.cameras.items() if f.stale]

    def frame_age(self, name: str) -> float | None:
        f = self.cameras.get(name)
        return None if f is None else f.age_s


class RobotClient:
    """Thin HTTP client. One method per robot_server endpoint."""

    def __init__(self, url: str = "http://localhost:7000", timeout: float = 30.0):
        self.url = url.rstrip("/")
        self.timeout = timeout
        self._session = requests.Session()
        try:
            self._session.get(f"{self.url}/health", timeout=5).raise_for_status()
        except requests.exceptions.ConnectionError as e:
            raise ConnectionError(
                f"No robot server at {self.url}.\n"
                f"Start it first, from the repository root:\n"
                f"    python -m control.robot_server "
                f"--camera-config config/camera_info.yaml --port "
                f"{self.url.rsplit(':', 1)[-1] or '7000'}"
            ) from e

    # -- internals -------------------------------------------------------
    def _post(self, route: str, payload: dict | None = None) -> dict:
        r = self._session.post(f"{self.url}/{route}", json=payload or {},
                               timeout=self.timeout)
        # Read the body BEFORE raise_for_status(): the server puts a specific
        # explanation in {"error": ...} on a 500, and raising first replaces it
        # with a bare "500 Server Error" that says nothing about the cause.
        try:
            body = r.json()
        except ValueError:
            body = None
        if body is not None and body.get("ok") is False:
            raise RuntimeError(f"{route} failed: {body.get('error')}")
        r.raise_for_status()
        return body if body is not None else {}

    @staticmethod
    def _state_fields(raw: dict) -> dict:
        return {
            "timestamp": raw["timestamp"],
            "ee_position": decode_array(raw["ee_position"]),
            "ee_quat_xyzw": decode_array(raw["ee_quat_xyzw"]),
            "joint_values": decode_array(raw["joint_values"]),
            "gripper_value": raw["gripper_value"],
            "gripper_state": decode_array(raw.get("gripper_state")),
            "ft_wrench": decode_array(raw.get("ft_wrench")),
        }

    def _resolve(self, position, quat_xyzw, fingertip: bool):
        """Convert a commanded pose to the end-effector frame the server wants."""
        position = np.asarray(position, dtype=np.float64)
        if not fingertip:
            return position, (None if quat_xyzw is None
                              else np.asarray(quat_xyzw, dtype=np.float64))
        if quat_xyzw is None:
            raise ValueError(
                "quat_xyzw is required in fingertip frame: the offset is applied "
                "along the gripper's own axes, so the orientation must be known. "
                "Pass fingertip=False to command a raw EE position."
            )
        ee = fingertip_to_ee(Pose(position=position,
                                  orientation=R.from_quat(quat_xyzw)))
        return ee.position, ee.orientation.as_quat()

    # -- reads -----------------------------------------------------------
    def get_state(self) -> RobotState:
        """Proprioception only. Cheap enough to poll in a control loop."""
        return RobotState(**self._state_fields(self._post("get_state")))

    def get_obs(self, include_depth: bool = True) -> RobotObs:
        """Full observation including camera frames. Several MB per call.

        include_depth=False halves the payload when the consumer only needs
        RGB, as the pi0.5 policy does.
        """
        raw = self._post("get_obs", {"include_depth": bool(include_depth)})
        cameras = {
            name: CameraFrame(rgb=decode_array(payload.get("rgb")),
                              depth=decode_array(payload.get("depth")),
                              age_s=payload.get("age_s"),
                              stale=bool(payload.get("stale", False)))
            for name, payload in (raw.get("cameras") or {}).items()
        }
        return RobotObs(
            **self._state_fields(raw),
            cameras=cameras,
            camera_names=raw.get("camera_names", list(cameras)),
            inhand_camera=raw.get("inhand_camera"),
        )

    # -- writes ----------------------------------------------------------
    def servo(self, position, quat_xyzw=None, fingertip: bool = True) -> dict:
        """Set one target pose and return immediately. Use this in a loop."""
        position, quat_xyzw = self._resolve(position, quat_xyzw, fingertip)
        payload = {"position": position.tolist()}
        if quat_xyzw is not None:
            payload["quat_xyzw"] = np.asarray(quat_xyzw).tolist()
        return self._post("servo", payload)

    def goto(self, position, quat_xyzw=None, use_move_to: bool = False,
             speed: float = 0.05, gripper: float | None = None,
             fingertip: bool = True) -> dict:
        """Blocking interpolated move. For scripted setup moves, not loops."""
        position, quat_xyzw = self._resolve(position, quat_xyzw, fingertip)
        payload = {
            "position": position.tolist(),
            "use_move_to": bool(use_move_to),
            "speed": float(speed),
        }
        if quat_xyzw is not None:
            payload["quat_xyzw"] = np.asarray(quat_xyzw).tolist()
        if gripper is not None:
            payload["gripper"] = float(gripper)
        return self._post("goto", payload)

    def servo_joint(self, q) -> dict:
        """Set one joint-position target and return immediately.

        Needs a server started with --ctrl-space joint.
        """
        q = np.asarray(q, dtype=np.float64)
        return self._post("servo_joint", {"q": q.tolist()})

    def set_gripper(self, value: float) -> dict:
        """1.0 = open, 0.0 = closed."""
        return self._post("set_gripper", {"value": float(value)})

    def home(self) -> dict:
        """Home the robot and restore the impedance controller. Blocking."""
        return self._post("home")

    def pause(self) -> dict:
        """Hold: the server accepts motion commands but does not apply them."""
        return self._post("pause")

    def resume(self) -> dict:
        """Apply motion commands again."""
        return self._post("resume")

    @property
    def is_paused(self) -> bool:
        return bool(self.health().get("paused", False))

    def cameras_status(self) -> dict:
        """Per-camera health, including which cameras are stale."""
        return self._post("cameras/status")

    def restart_cameras(self) -> dict:
        """Stop and relaunch the camera nodes. Blocking on the stop."""
        self._post("cameras/stop")
        return self._post("cameras/start")

    def health(self) -> dict:
        """Server health, including which control space it is running in."""
        r = self._session.get(f"{self.url}/health", timeout=5)
        r.raise_for_status()
        return r.json()

    @property
    def ctrl_space(self) -> str | None:
        """'joint' or 'cartesian'; None if the server predates this field."""
        return self.health().get("ctrl_space")
