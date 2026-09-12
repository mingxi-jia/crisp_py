"""Independent per-camera RGB-D subscriber.

Deliberately NOT crisp_py.camera.pointcloud.PointCloudManager, which feeds
cam1+cam2+cam3+cam4 into a single ApproximateTimeSynchronizer: with any of
those cameras absent the callback never fires and *every* camera goes dark.
That makes it unusable with a launch file that brings up a subset.

Here each camera has its own subscriptions and its own latest frame, so a
missing or dead camera costs you that camera only. There is no cross-camera
time synchronisation -- frames carry their own stamp; anything that needs them
aligned should compare stamps itself.
"""

import re
import threading
import time

import numpy as np
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from rclpy.qos import (
    DurabilityPolicy,
    HistoryPolicy,
    QoSProfile,
    ReliabilityPolicy,
)
from sensor_msgs.msg import Image

# Colour topic naming is not consistent across this rig: the external cameras
# publish image_raw, the in-hand one publishes image_rect_raw. Subscribe to
# both and take whichever arrives.
COLOR_TOPICS = ("color/image_raw", "color/image_rect_raw")
DEPTH_TOPIC = "aligned_depth_to_color/image_raw"

# A frame older than this is treated as absent. RealSense cameras can drop off
# the USB bus while their ROS node stays alive and healthy-looking, so node
# liveness is NOT evidence that images are arriving -- only the frame's own age
# is. Without this check a policy happily consumes a minutes-old image.
DEFAULT_MAX_AGE = 1.0        # seconds


class CameraFrame:
    """Latest RGB and depth for one camera."""

    __slots__ = ("rgb", "depth", "rgb_stamp", "depth_stamp")

    def __init__(self):
        self.rgb = None
        self.depth = None
        self.rgb_stamp = None
        self.depth_stamp = None

    @property
    def has_rgb(self) -> bool:
        return self.rgb is not None

    def age(self) -> float | None:
        """Seconds since the last RGB frame, or None if none has arrived."""
        return None if self.rgb_stamp is None else time.time() - self.rgb_stamp


class CameraHub(Node):
    """ROS node holding the latest frame from each requested camera."""

    def __init__(self, camera_names, node_name: str = "crisp_camera_hub",
                 depth_scale: float = 1000.0):
        super().__init__(node_name)
        self.camera_names = list(camera_names)
        self.depth_scale = depth_scale
        self.bridge = CvBridge()
        self._frames = {name: CameraFrame() for name in self.camera_names}
        self._lock = threading.Lock()

        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )

        self._subs = []
        for name in self.camera_names:
            for topic in COLOR_TOPICS:
                self._subs.append(self.create_subscription(
                    Image, f"/{name}/{topic}",
                    self._make_rgb_cb(name), qos))
            self._subs.append(self.create_subscription(
                Image, f"/{name}/{DEPTH_TOPIC}",
                self._make_depth_cb(name), qos))

        self.get_logger().info(
            f"CameraHub watching {len(self.camera_names)} camera(s): "
            f"{', '.join(self.camera_names)}")

    # -- callbacks -------------------------------------------------------
    def _make_rgb_cb(self, name):
        def cb(msg):
            try:
                img = self.bridge.imgmsg_to_cv2(msg, "rgb8")
            except Exception as e:
                self.get_logger().warn(f"{name} rgb decode failed: {e}")
                return
            with self._lock:
                f = self._frames[name]
                f.rgb = img
                f.rgb_stamp = time.time()
        return cb

    def _make_depth_cb(self, name):
        def cb(msg):
            try:
                img = self.bridge.imgmsg_to_cv2(msg, "16UC1")
            except Exception as e:
                self.get_logger().warn(f"{name} depth decode failed: {e}")
                return
            with self._lock:
                f = self._frames[name]
                f.depth = img
                f.depth_stamp = time.time()
        return cb

    # -- accessors -------------------------------------------------------
    def frame(self, name: str) -> CameraFrame | None:
        with self._lock:
            return self._frames.get(name)

    def is_stale(self, name: str, max_age: float | None = None) -> bool:
        """True if this camera has no frame, or its newest is too old."""
        max_age = DEFAULT_MAX_AGE if max_age is None else max_age
        f = self.frame(name)
        if f is None or f.rgb_stamp is None:
            return True
        return (time.time() - f.rgb_stamp) > max_age

    def rgb(self, name: str, max_age: float | None = None):
        """Latest RGB, or None if the camera is stale.

        Pass max_age=float('inf') to get the last frame regardless of age --
        useful for a UI that wants to show it with a "stale" badge, never for
        anything that acts on it.
        """
        f = self.frame(name)
        if f is None or self.is_stale(name, max_age):
            return None
        return f.rgb

    def depth(self, name: str, meters: bool = False, max_age: float | None = None):
        f = self.frame(name)
        if f is None or f.depth is None or self.is_stale(name, max_age):
            return None
        return f.depth.astype(np.float32) / self.depth_scale if meters else f.depth

    def ready_cameras(self) -> list[str]:
        """Cameras delivering fresh frames right now."""
        return [n for n in self.camera_names if not self.is_stale(n)]

    def seen_cameras(self) -> list[str]:
        """Cameras that have delivered a frame at some point, fresh or not."""
        with self._lock:
            return [n for n, f in self._frames.items() if f.rgb is not None]

    def stale_cameras(self, max_age: float | None = None) -> list[str]:
        return [n for n in self.camera_names if self.is_stale(n, max_age)]

    def status(self, max_age: float | None = None) -> dict:
        max_age = DEFAULT_MAX_AGE if max_age is None else max_age
        now = time.time()
        with self._lock:
            out = {}
            for name, f in self._frames.items():
                age = None if f.rgb_stamp is None else now - f.rgb_stamp
                out[name] = {
                    "has_rgb": f.rgb is not None,
                    "has_depth": f.depth is not None,
                    "age_s": None if age is None else round(age, 3),
                    "stale": age is None or age > max_age,
                    "max_age_s": max_age,
                    "shape": list(f.rgb.shape) if f.rgb is not None else None,
                }
            return out

    def wait_for_all(self, timeout: float = 20.0,
                     max_age: float | None = None) -> bool:
        """Block until every camera is delivering fresh frames."""
        deadline = time.time() + timeout
        while time.time() < deadline:
            if not self.stale_cameras(max_age):
                return True
            time.sleep(0.05)
        return False

    def wait_for_any(self, timeout: float = 10.0) -> bool:
        """Block until at least one camera has a frame. False on timeout."""
        deadline = time.time() + timeout
        while time.time() < deadline:
            if self.ready_cameras():
                return True
            time.sleep(0.05)
        return False

    def clear_cache(self):
        with self._lock:
            for f in self._frames.values():
                f.rgb = f.depth = f.rgb_stamp = f.depth_stamp = None


# Only names of this shape are auto-discovered. The ROS domain is shared with
# other machines on this network -- a bare "publishes a colour topic" match
# also picks up their cameras (seen in the wild: "dave", "stuart"), and this
# server would then happily serve someone else's video. Pass an explicit
# camera list to use anything outside this convention.
CAMERA_NAME_RE = re.compile(r"^cam\d+$")


def discover_cameras(node: Node | None = None, timeout: float = 5.0,
                     pattern: re.Pattern | None = None) -> list[str]:
    """Camera names on this ROS graph publishing colour, e.g. ['cam1', 'cam4'].

    Lets the server adapt to whichever launch file is running instead of
    hardcoding a camera list. Restricted to :data:`CAMERA_NAME_RE` unless
    ``pattern`` says otherwise.
    """
    pattern = pattern or CAMERA_NAME_RE
    owns_node = node is None
    if owns_node:
        node = rclpy.create_node("crisp_camera_discovery")
    try:
        deadline = time.time() + timeout
        found: set[str] = set()
        while time.time() < deadline:
            for topic, _types in node.get_topic_names_and_types():
                parts = topic.strip("/").split("/")
                if (len(parts) >= 3 and parts[1] == "color"
                        and parts[2] in ("image_raw", "image_rect_raw")
                        and pattern.match(parts[0])):
                    found.add(parts[0])
            if found:
                break
            time.sleep(0.25)
        return sorted(found)
    finally:
        if owns_node:
            node.destroy_node()
