"""The single source of observations. Everything that needs obs imports this.

Before this, each consumer built its own camera subscriptions and its own
proprioception plumbing -- robot_server, the recorder, the live point-cloud
scripts, the old controllers. That meant four places to fix when a camera
convention changed, and no shared notion of "this frame is too old to use".

    from control.observer import Observer

    obs_source = Observer(backend="ros", cameras=["cam1", "cam4"])
    obs_source.wait_until_ready()

    state = obs_source.get_state()        # proprioception only, cheap
    obs   = obs_source.get_obs()          # + camera frames
    rgb   = obs.rgb("cam1")               # None if that camera is stale

Backends, swapped with one argument:

    "ros"        subscribe to /camN/... topics. Shares cameras with any other
                 ROS consumer (the recorder, realsense-viewer, rviz).
    "realsense"  open the devices through the SDK. One layer instead of four,
                 and a stalled device can be hardware_reset(), but this
                 process then OWNS the cameras -- nothing else can read them.
    "client"     read from a running robot_server over HTTP. For code on
                 another machine, or outside the ROS environment.

Camera health is also checkable from here:

    python -m control.observer --require cam1,cam4     # exit 1 if unhealthy
    python -m control.observer --watch

Staleness is enforced in every backend: a frame older than max_age is reported
as absent rather than returned, because a RealSense can drop off the USB bus
while its last image sits in memory looking perfectly valid.
"""

import dataclasses
import threading
import time

import numpy as np

DEFAULT_MAX_AGE = 1.0
BACKENDS = ("ros", "realsense", "client")


@dataclasses.dataclass
class CameraFrame:
    rgb: np.ndarray | None = None
    depth: np.ndarray | None = None       # uint16, millimetres
    age_s: float | None = None
    stale: bool = False


@dataclasses.dataclass
class Observation:
    """One timestep. Camera fields are empty for get_state()."""

    timestamp: float
    cameras: dict = dataclasses.field(default_factory=dict)
    joint_values: np.ndarray | None = None
    ee_position: np.ndarray | None = None
    ee_quat_xyzw: np.ndarray | None = None
    gripper_value: float | None = None          # crisp convention: 1.0 open
    gripper_state: np.ndarray | None = None
    ft_wrench: np.ndarray | None = None         # fx fy fz tx ty tz
    inhand_camera: str | None = None

    # -- cameras ---------------------------------------------------------
    def rgb(self, name: str):
        f = self.cameras.get(name)
        return None if f is None else f.rgb

    def depth(self, name: str):
        f = self.cameras.get(name)
        return None if f is None else f.depth

    def frame_age(self, name: str):
        f = self.cameras.get(name)
        return None if f is None else f.age_s

    @property
    def ready_cameras(self) -> list[str]:
        return [n for n, f in self.cameras.items() if f.rgb is not None]

    @property
    def stale_cameras(self) -> list[str]:
        return [n for n, f in self.cameras.items() if f.stale]

    @property
    def inhand_rgb(self):
        return self.rgb(self.inhand_camera) if self.inhand_camera else None

    @property
    def inhand_depth(self):
        return self.depth(self.inhand_camera) if self.inhand_camera else None

    # -- robot -----------------------------------------------------------
    @property
    def ee_pose(self):
        from crisp_py.robot import Pose
        from scipy.spatial.transform import Rotation as R
        if self.ee_position is None or self.ee_quat_xyzw is None:
            return None
        return Pose(position=np.asarray(self.ee_position, dtype=np.float64),
                    orientation=R.from_quat(self.ee_quat_xyzw))

    @property
    def fingertip_pose(self):
        from control.kinematics import ee_to_fingertip
        pose = self.ee_pose
        return None if pose is None else ee_to_fingertip(pose)


# ---------------------------------------------------------------------------
# Proprioception sources
# ---------------------------------------------------------------------------
class RosProprio:
    """Joint states, gripper and FT read straight off ROS topics.

    Standalone: it owns its node, so a script needs nothing but this. When a
    crisp_py Robot/Gripper already exists in the process, pass them instead
    (see RobotProprio) rather than duplicating the subscriptions.
    """

    FRANKA_JOINTS = [f"fr3_joint{i}" for i in range(1, 8)]
    GRIPPER_JOINTS = ["gripper_joint"]

    def __init__(self, node=None, ft_sensor: bool = True, spin: bool = True):
        import rclpy
        from geometry_msgs.msg import PoseStamped
        from sensor_msgs.msg import JointState

        self._owns_node = node is None
        if self._owns_node:
            if not rclpy.ok():
                rclpy.init()
            node = rclpy.create_node("crisp_observer_proprio")
        self._node = node
        self._lock = threading.Lock()
        self.joint_values = None
        self.gripper_state = None
        self.ft_wrench = None
        self.ee_position = None
        self.ee_quat_xyzw = None
        self.ft_sensor = ft_sensor

        node.create_subscription(JointState, "/joint_states", self._on_joints, 10)
        node.create_subscription(JointState, "/gripper/gripper_state",
                                 self._on_gripper, 10)
        node.create_subscription(PoseStamped, "/current_pose", self._on_pose, 10)
        if ft_sensor:
            from geometry_msgs.msg import WrenchStamped
            node.create_subscription(
                WrenchStamped,
                "/ft/robotiq_force_torque_sensor_broadcaster/wrench",
                self._on_wrench, 10)

        # Only spin when nobody else will. Two concurrent rclpy.spin() calls
        # in one process fight over the executor and raise
        # "generator already executing", which silently kills the callbacks.
        self._spin = None
        if self._owns_node and spin:
            self._spin = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
            self._spin.start()

    def _on_joints(self, msg):
        by_name = dict(zip(msg.name, msg.position))
        if all(j in by_name for j in self.FRANKA_JOINTS):
            with self._lock:
                self.joint_values = np.array(
                    [by_name[j] for j in self.FRANKA_JOINTS], dtype=np.float32)

    def _on_gripper(self, msg):
        vals = [1.0 if p >= 0.95 else 0.0
                for n, p in zip(msg.name, msg.position) if n in self.GRIPPER_JOINTS]
        if vals:
            with self._lock:
                self.gripper_state = np.asarray(vals, dtype=np.float32)

    def _on_pose(self, msg):
        p, o = msg.pose.position, msg.pose.orientation
        with self._lock:
            self.ee_position = np.array([p.x, p.y, p.z], dtype=np.float32)
            self.ee_quat_xyzw = np.array([o.x, o.y, o.z, o.w], dtype=np.float32)

    def _on_wrench(self, msg):
        w = msg.wrench
        with self._lock:
            self.ft_wrench = np.array(
                [w.force.x, w.force.y, w.force.z,
                 w.torque.x, w.torque.y, w.torque.z], dtype=np.float32)

    @property
    def node(self):
        return self._node

    @property
    def is_ready(self) -> bool:
        with self._lock:
            return self.joint_values is not None and self.ee_position is not None

    def read(self) -> dict:
        with self._lock:
            gripper = (None if self.gripper_state is None
                       else float(self.gripper_state[0]))
            return {"joint_values": self.joint_values,
                    "ee_position": self.ee_position,
                    "ee_quat_xyzw": self.ee_quat_xyzw,
                    "gripper_value": gripper,
                    "gripper_state": self.gripper_state,
                    "ft_wrench": self.ft_wrench}

    def close(self):
        if self._owns_node:
            try:
                self._node.destroy_node()
            except Exception:
                pass


class RobotProprio:
    """Proprioception from an existing crisp_py Robot + Gripper.

    Used by robot_server, which already owns both; creating a second set of
    subscriptions for the same data would be waste and a source of skew.
    """

    def __init__(self, robot, gripper=None, joint_state_sub=None):
        self.robot = robot
        self.gripper = gripper
        self.joint_state_sub = joint_state_sub

    @property
    def is_ready(self) -> bool:
        return self.robot is not None

    def read(self) -> dict:
        pose = self.robot.end_effector_pose
        try:
            gripper_value = None if self.gripper is None else float(self.gripper.value)
        except Exception:
            gripper_value = None
        sub = self.joint_state_sub
        return {
            "joint_values": np.asarray(self.robot.joint_values, dtype=np.float32),
            "ee_position": np.asarray(pose.position, dtype=np.float32),
            "ee_quat_xyzw": pose.orientation.as_quat().astype(np.float32),
            "gripper_value": gripper_value,
            "gripper_state": getattr(sub, "gripper_joint_array", None),
            "ft_wrench": getattr(sub, "gripper_torque_array", None),
        }

    def close(self):
        pass


# ---------------------------------------------------------------------------
# Observer
# ---------------------------------------------------------------------------
class Observer:
    """Assembles Observations from a camera backend and a proprio source."""

    def __init__(self,
                 backend: str = "ros",
                 cameras=None,
                 serials: dict | None = None,
                 url: str = "http://localhost:7000",
                 inhand_camera: str | None = None,
                 max_age: float = DEFAULT_MAX_AGE,
                 ft_sensor: bool = True,
                 robot=None, gripper=None, joint_state_sub=None,
                 node=None):
        if backend not in BACKENDS:
            raise ValueError(f"backend must be one of {BACKENDS}, got {backend!r}")
        self.backend = backend
        self.inhand_camera = inhand_camera or "cam4"
        self.max_age = max_age
        self._client = None
        self._cameras = None
        self._proprio = None

        if backend == "client":
            from control.robot_client import RobotClient
            self._client = RobotClient(url)
            return

        import rclpy
        if not rclpy.ok():
            rclpy.init()
        self._nodes = []
        self._executor = None

        if backend == "realsense":
            from control.realsense_direct import CameraSpec, RealsenseCameras
            if serials:
                specs = [CameraSpec(name, **cfg) for name, cfg in serials.items()]
            else:
                from control.realsense_direct import load_camera_config
                specs, configured_inhand = load_camera_config()
                if not specs:
                    raise ValueError(
                        "no cameras configured: add them to config/cameras.yaml "
                        "or pass serials={...}")
                if inhand_camera is None:
                    self.inhand_camera = configured_inhand
            # Say what was opened and what is unreachable. Without this the
            # server starts "successfully" and only reveals the problem later
            # as 503s from /camera/<cam>.jpg.
            from control.realsense_direct import missing_devices
            missing = missing_devices([sp.serial for sp in specs])
            for sp in specs:
                state = "MISSING from the SDK" if sp.serial in missing else "ok"
                print(f"  camera {sp.name}: {sp.serial} "
                      f"{sp.width}x{sp.height}@{sp.fps}"
                      f"{' +depth' if sp.depth else ''}  {state}")
            if missing:
                print(f"WARNING: {len(missing)} configured camera(s) are enumerated "
                      f"by the kernel but cannot be opened by the RealSense SDK. "
                      f"That is a wedged USB device -- replug it or run "
                      f"`sudo usbreset`. Streams will keep retrying.")
            self._cameras = RealsenseCameras(specs)
        else:
            from control.cameras import CameraHub, discover_cameras
            names = list(cameras) if cameras else discover_cameras(timeout=5.0)
            self._cameras = CameraHub(names)
            self._nodes.append(self._cameras)

        if robot is not None:
            self._proprio = RobotProprio(robot, gripper, joint_state_sub)
        else:
            # spin=False: this Observer drives every node it owns from a single
            # MultiThreadedExecutor below.
            self._proprio = RosProprio(node, ft_sensor, spin=False)
            if self._proprio.node is not None:
                self._nodes.append(self._proprio.node)

        if self._nodes:
            from rclpy.executors import MultiThreadedExecutor
            self._executor = MultiThreadedExecutor()
            for n in self._nodes:
                self._executor.add_node(n)
            self._spin = threading.Thread(target=self._executor.spin, daemon=True)
            self._spin.start()

    # -- introspection ---------------------------------------------------
    @property
    def cameras(self):
        """The camera backend (CameraHub or RealsenseCameras); None for 'client'."""
        return self._cameras

    @property
    def proprio(self):
        """The proprioception source; None for 'client'."""
        return self._proprio

    @property
    def camera_names(self) -> list[str]:
        if self._client is not None:
            return self._client.cameras_status().get("names", [])
        return list(self._cameras.camera_names)

    def camera_status(self) -> dict:
        if self._client is not None:
            return self._client.cameras_status().get("frames", {})
        return self._cameras.status(self.max_age)

    def wait_until_ready(self, timeout: float = 30.0,
                         require_cameras: bool = True) -> bool:
        deadline = time.time() + timeout
        while time.time() < deadline:
            if self._client is not None:
                try:
                    st = self._client.cameras_status()
                    if not require_cameras or st.get("healthy"):
                        return True
                except Exception:
                    pass
            else:
                proprio_ok = self._proprio is None or self._proprio.is_ready
                cams_ok = (not require_cameras
                           or not self._cameras.stale_cameras(self.max_age))
                if proprio_ok and cams_ok:
                    return True
            time.sleep(0.1)
        return False

    # -- reads -----------------------------------------------------------
    def get_state(self) -> Observation:
        """Proprioception only. Cheap enough for a control loop."""
        if self._client is not None:
            s = self._client.get_state()
            return Observation(
                timestamp=s.timestamp, joint_values=s.joint_values,
                ee_position=s.ee_position, ee_quat_xyzw=s.ee_quat_xyzw,
                gripper_value=s.gripper_value, gripper_state=s.gripper_state,
                ft_wrench=s.ft_wrench, inhand_camera=self.inhand_camera)
        return Observation(timestamp=time.time(),
                           inhand_camera=self.inhand_camera,
                           **self._proprio.read())

    def get_obs(self) -> Observation:
        """Proprioception plus the current camera frames."""
        if self._client is not None:
            o = self._client.get_obs()
            return Observation(
                timestamp=o.timestamp,
                cameras={n: CameraFrame(f.rgb, f.depth, f.age_s, f.stale)
                         for n, f in o.cameras.items()},
                joint_values=o.joint_values, ee_position=o.ee_position,
                ee_quat_xyzw=o.ee_quat_xyzw, gripper_value=o.gripper_value,
                gripper_state=o.gripper_state, ft_wrench=o.ft_wrench,
                inhand_camera=o.inhand_camera or self.inhand_camera)

        obs = self.get_state()
        status = self._cameras.status(self.max_age)
        for name in self._cameras.camera_names:
            info = status.get(name, {})
            stale = bool(info.get("stale", True))
            obs.cameras[name] = CameraFrame(
                rgb=None if stale else self._cameras.rgb(name, self.max_age),
                depth=None if stale else self._cameras.depth(name, max_age=self.max_age),
                age_s=info.get("age_s"),
                stale=stale)
        return obs

    def close(self):
        if getattr(self, "_executor", None) is not None:
            try:
                self._executor.shutdown(timeout_sec=2.0)
            except Exception:
                pass
            self._executor = None
        for obj in (self._cameras, self._proprio):
            for meth in ("shutdown", "destroy_node", "close"):
                fn = getattr(obj, meth, None)
                if callable(fn):
                    try:
                        fn()
                    except Exception:
                        pass
                    break

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()



# ---------------------------------------------------------------------------
# CLI: camera health
#
# Node liveness is not camera liveness: a RealSense can fall off the USB bus
# while its ROS node keeps running, after which the last frame sits in memory
# forever. This checks the only thing that matters -- how old each frame is --
# and exits non-zero when a required camera is not delivering, so it gates a
# rollout:
#
#   python -m control.observer --require cam1,cam4 && python -m deploy.deploy_pi05 ...
# ---------------------------------------------------------------------------
_G, _Y, _R, _D, _N = "\033[32m", "\033[33m", "\033[31m", "\033[2m", "\033[0m"


def parse_args():
    import argparse
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--url", default="http://localhost:7000")
    p.add_argument("--require", default=None,
                   help="comma-separated cameras that must be fresh "
                        "(default: all the server watches)")
    p.add_argument("--max-age", type=float, default=1.0,
                   help="a frame older than this counts as stale [s] "
                        "(used only if the server does not say)")
    p.add_argument("--watch", action="store_true", help="poll until interrupted")
    p.add_argument("--interval", type=float, default=1.0)
    return p.parse_args()


def fetch(url: str) -> dict:
    import requests
    r = requests.post(f"{url.rstrip('/')}/cameras/status", timeout=5)
    r.raise_for_status()
    return r.json()


def report(status: dict, required: list[str], color: bool = True,
           max_age: float = 1.0) -> bool:
    g, y, r, d, n = (_G, _Y, _R, _D, _N) if color else ("",) * 5
    frames = status.get("frames", {})
    names = status.get("names", [])
    required = required or names

    print(f"  nodes running: {status.get('n_nodes', 0)}   "
          f"{d}(process count — not proof of streaming){n}")
    ok = True
    for name in names:
        f = frames.get(name, {})
        age = f.get("age_s")
        # Older servers do not send `stale`; derive it from the age so this
        # tool works against both. Defaulting the flag to True would report a
        # perfectly fresh camera as stale.
        stale = f.get("stale")
        if stale is None:
            stale = age is None or age > f.get("max_age_s", max_age)
        needed = name in required
        if age is None:
            mark, col, detail = "x", r, "no frame ever received"
        elif stale:
            mark, col, detail = "x", r, f"STALE — last frame {age:.1f}s ago"
        else:
            mark, col, detail = "o", g, f"fresh ({age * 1000:.0f} ms)"
        shape = f.get("shape")
        shape_s = f"{shape[1]}x{shape[0]}" if shape else "—"
        req_s = "" if needed else f" {d}(not required){n}"
        print(f"  [{col}{mark}{n}] {name:<6s} {shape_s:>9s}  {col}{detail}{n}{req_s}")
        if needed and stale:
            ok = False

    missing = [c for c in required if c not in names]
    for name in missing:
        print(f"  [{r}x{n}] {name:<6s} {'—':>9s}  {r}not watched by the server{n}")
        ok = False

    print(f"\n  {(g + 'HEALTHY' + n) if ok else (r + 'UNHEALTHY' + n)}"
          f"  required: {', '.join(required) or 'none'}")
    if not ok:
        print(f"  {d}restart cameras:  ./scripts/start_all.sh --stop && "
              f"./scripts/start_all.sh{n}")
        print(f"  {d}or from the panel: the Reconnect cameras button{n}")
    return ok


def main():
    import sys
    args = parse_args()
    required = ([c.strip() for c in args.require.split(",") if c.strip()]
                if args.require else [])
    color = sys.stdout.isatty()

    try:
        if not args.watch:
            sys.exit(0 if report(fetch(args.url), required, color, args.max_age) else 1)
        while True:
            print(f"\n=== {time.strftime('%H:%M:%S')} ===")
            try:
                report(fetch(args.url), required, color, args.max_age)
            except Exception as e:
                print(f"  cannot reach {args.url}: {e}")
            time.sleep(args.interval)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"cannot reach the robot server at {args.url}: {e}")
        sys.exit(2)


if __name__ == "__main__":
    main()
