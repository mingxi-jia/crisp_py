"""Read RealSense cameras directly through the SDK, bypassing ROS.

Drop-in alternative to control/cameras.py (CameraHub): same accessor surface,
so robot_server can use either. What this buys over the ROS path:

  * One layer instead of four (device -> SDK -> node -> DDS -> subscriber).
  * Explicit per-device health. A wedged camera is visible here as a device
    the SDK cannot enumerate, rather than a topic that silently stops.
  * hardware_reset() on stall. The ROS node cannot recover a stalled device;
    this can, because it holds the handle.

What it does NOT buy: immunity to a device falling off the USB bus. When that
happens the SDK cannot see it either (verified: kernel lsusb showed 4 devices
while the SDK enumerated 2). The fix there is a USB reset, not a software
layer -- see reset_usb_device() below.

Only one process may own a RealSense device, so the ROS camera nodes must be
stopped before using this.
"""

import threading
import time

import numpy as np

DEFAULT_MAX_AGE = 1.0          # seconds; matches control/cameras.py


class CameraSpec:
    """How to open one camera."""

    def __init__(self, name: str, serial: str, width: int = 848, height: int = 480,
                 fps: int = 15, depth: bool = False,
                 exposure_us: float | None = None, gain: float | None = None,
                 auto_exposure: bool | None = None):
        """depth defaults OFF, matching control/launch_cameras.launch.py.

        Nothing in the pi0.5 pipeline reads depth -- its observation is two RGB
        images plus joint positions -- and the stream costs ~40% of a camera's
        USB bandwidth plus a depth-to-colour alignment pass. Enable it per
        camera when something actually consumes it."""
        self.name = name
        self.serial = str(serial)
        self.width = int(width)
        self.height = int(height)
        self.fps = int(fps)
        self.depth = bool(depth)
        # Motion blur is exposure time, not frame rate: a moving gripper smears
        # by (its speed x exposure). Auto-exposure optimises for brightness and
        # will happily pick 33 ms, which is half a frame at 15 fps. Set
        # exposure_us to pin it short and raise gain to compensate.
        self.exposure_us = exposure_us
        self.gain = gain
        # None = leave as-is; False = manual (required for exposure_us to hold).
        self.auto_exposure = (False if (auto_exposure is None and exposure_us is not None)
                              else auto_exposure)

    def __repr__(self):
        return (f"CameraSpec({self.name!r}, {self.serial!r}, "
                f"{self.width}x{self.height}@{self.fps}"
                f"{'+depth' if self.depth else ''}"
                f"{'' if self.exposure_us is None else f', {self.exposure_us:.0f}us'})")


class _Stream:
    """One camera's pipeline, polled by its own thread."""

    def __init__(self, spec: CameraSpec, stall_timeout: float, auto_reset: bool):
        self.spec = spec
        self.stall_timeout = stall_timeout
        self.auto_reset = auto_reset
        self.rgb = None
        self.depth = None
        self.rgb_stamp = None
        self.depth_stamp = None
        self.error = None
        self.resets = 0
        self.frames = 0
        self.gave_up = False
        self._pipe = None
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = None

    # -- lifecycle -------------------------------------------------------
    def start(self):
        self._thread = threading.Thread(target=self._run, daemon=True,
                                        name=f"rs-{self.spec.name}")
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
        self._close()

    def _close(self):
        if self._pipe is not None:
            try:
                self._pipe.stop()
            except Exception:
                pass
            self._pipe = None

    def _open(self):
        import pyrealsense2 as rs

        cfg = rs.config()
        cfg.enable_device(self.spec.serial)
        cfg.enable_stream(rs.stream.color, self.spec.width, self.spec.height,
                          rs.format.rgb8, self.spec.fps)
        if self.spec.depth:
            cfg.enable_stream(rs.stream.depth, self.spec.width, self.spec.height,
                              rs.format.z16, self.spec.fps)
        pipe = rs.pipeline()
        profile = pipe.start(cfg)
        self._pipe = pipe
        self._apply_options(profile)
        # Align depth to colour so a depth pixel matches its colour pixel.
        self._align = rs.align(rs.stream.color) if self.spec.depth else None
        return profile

    def _apply_options(self, profile):
        """Push exposure/gain onto every sensor that accepts them.

        The D405 has no separate RGB sensor -- its colour comes from the depth
        module -- so the option has to be applied per sensor rather than to a
        named 'RGB Camera'.
        """
        import pyrealsense2 as rs

        spec = self.spec
        if spec.auto_exposure is None and spec.exposure_us is None and spec.gain is None:
            return
        for sensor in profile.get_device().query_sensors():
            name = sensor.get_info(rs.camera_info.name)
            try:
                if spec.auto_exposure is not None and sensor.supports(
                        rs.option.enable_auto_exposure):
                    sensor.set_option(rs.option.enable_auto_exposure,
                                      1.0 if spec.auto_exposure else 0.0)
                if spec.exposure_us is not None and sensor.supports(rs.option.exposure):
                    rng = sensor.get_option_range(rs.option.exposure)
                    # The RGB sensor counts exposure in 100 us units while the
                    # depth module counts microseconds; detect from the range
                    # rather than hardcoding per model.
                    value = (spec.exposure_us / 100.0 if rng.max <= 10000
                             else spec.exposure_us)
                    value = max(rng.min, min(rng.max, value))
                    sensor.set_option(rs.option.exposure, value)
                if spec.gain is not None and sensor.supports(rs.option.gain):
                    rng = sensor.get_option_range(rs.option.gain)
                    sensor.set_option(rs.option.gain,
                                      max(rng.min, min(rng.max, spec.gain)))
            except Exception as e:
                print(f"[realsense] {spec.name} [{name}]: could not set options: {e}")

    # -- worker ----------------------------------------------------------
    # A device error that never self-heals (a wedged D4xx needing a USB reset)
    # must not be retried forever: each attempt grabs the handle again, which
    # is the opposite of helpful for a device trying to re-enumerate.
    MAX_CONSECUTIVE_FAILURES = 10

    def _run(self):
        backoff = 1.0
        failures = 0
        while not self._stop.is_set():
            try:
                self._open()
                with self._lock:
                    self.error = None
                backoff, failures = 1.0, 0
                self._pump()
            except Exception as e:
                failures += 1
                with self._lock:
                    self.error = repr(e)
                self._close()
                if failures >= self.MAX_CONSECUTIVE_FAILURES:
                    msg = (f"{self.spec.name}: giving up after {failures} failed "
                           f"attempts ({e}). The device needs a USB reset -- "
                           f"replug it, then restart. Not holding the handle.")
                    print(f"[realsense] {msg}")
                    with self._lock:
                        self.error = msg
                        self.gave_up = True
                    return
                if self._stop.wait(backoff):
                    return
                backoff = min(backoff * 2, 10.0)

    def _pump(self):
        last = time.time()
        while not self._stop.is_set():
            frames = self._pipe.poll_for_frames()
            if not frames:
                # No frame for stall_timeout: the device has gone quiet while
                # still held open. This is the failure the ROS node could only
                # report, never fix.
                if time.time() - last > self.stall_timeout:
                    with self._lock:
                        self.error = f"no frames for {time.time() - last:.1f}s"
                    if self.auto_reset:
                        self._hardware_reset()
                    self._close()
                    return
                time.sleep(0.002)
                continue

            if self._align is not None:
                frames = self._align.process(frames)
            color = frames.get_color_frame()
            if not color:
                continue
            rgb = np.asanyarray(color.get_data())
            depth_frame = frames.get_depth_frame() if self.spec.depth else None
            depth = np.asanyarray(depth_frame.get_data()) if depth_frame else None

            now = time.time()
            last = now
            with self._lock:
                self.rgb = rgb
                self.rgb_stamp = now
                if depth is not None:
                    self.depth = depth
                    self.depth_stamp = now
                self.frames += 1

    def _hardware_reset(self):
        """Ask the device to reset itself. Only possible while we hold it."""
        try:
            profile = self._pipe.get_active_profile()
            profile.get_device().hardware_reset()
            with self._lock:
                self.resets += 1
            time.sleep(3.0)          # the device re-enumerates
        except Exception as e:
            with self._lock:
                self.error = f"reset failed: {e!r}"

    # -- accessors -------------------------------------------------------
    def snapshot(self) -> dict:
        with self._lock:
            return {"rgb": self.rgb, "depth": self.depth,
                    "rgb_stamp": self.rgb_stamp, "error": self.error,
                    "resets": self.resets, "frames": self.frames,
                    "gave_up": self.gave_up}


class RealsenseCameras:
    """CameraHub-compatible reader that talks to the SDK directly."""

    def __init__(self, specs, stall_timeout: float = 3.0, auto_reset: bool = True,
                 depth_scale: float = 1000.0):
        self.specs = {s.name: s for s in specs}
        self.camera_names = [s.name for s in specs]
        self.depth_scale = depth_scale
        self._streams = {s.name: _Stream(s, stall_timeout, auto_reset) for s in specs}
        for st in self._streams.values():
            st.start()

    # -- the CameraHub surface -------------------------------------------
    def is_stale(self, name: str, max_age: float | None = None) -> bool:
        max_age = DEFAULT_MAX_AGE if max_age is None else max_age
        st = self._streams.get(name)
        if st is None:
            return True
        snap = st.snapshot()
        if snap["rgb_stamp"] is None:
            return True
        return (time.time() - snap["rgb_stamp"]) > max_age

    def rgb(self, name: str, max_age: float | None = None):
        st = self._streams.get(name)
        if st is None or self.is_stale(name, max_age):
            return None
        return st.snapshot()["rgb"]

    def depth(self, name: str, meters: bool = False, max_age: float | None = None):
        st = self._streams.get(name)
        if st is None or self.is_stale(name, max_age):
            return None
        d = st.snapshot()["depth"]
        if d is None:
            return None
        return d.astype(np.float32) / self.depth_scale if meters else d

    def ready_cameras(self) -> list[str]:
        return [n for n in self.camera_names if not self.is_stale(n)]

    def stale_cameras(self, max_age: float | None = None) -> list[str]:
        return [n for n in self.camera_names if self.is_stale(n, max_age)]

    def status(self, max_age: float | None = None) -> dict:
        max_age = DEFAULT_MAX_AGE if max_age is None else max_age
        now = time.time()
        out = {}
        for name, st in self._streams.items():
            snap = st.snapshot()
            age = None if snap["rgb_stamp"] is None else now - snap["rgb_stamp"]
            out[name] = {
                "has_rgb": snap["rgb"] is not None,
                "has_depth": snap["depth"] is not None,
                "age_s": None if age is None else round(age, 3),
                "stale": age is None or age > max_age,
                "max_age_s": max_age,
                "shape": list(snap["rgb"].shape) if snap["rgb"] is not None else None,
                "error": snap["error"],
                "resets": snap["resets"],
                "frames": snap["frames"],
                "gave_up": snap.get("gave_up", False),
                "serial": self.specs[name].serial,
            }
        return out

    def wait_for_any(self, timeout: float = 10.0) -> bool:
        deadline = time.time() + timeout
        while time.time() < deadline:
            if self.ready_cameras():
                return True
            time.sleep(0.05)
        return False

    def wait_for_all(self, timeout: float = 20.0, max_age: float | None = None) -> bool:
        deadline = time.time() + timeout
        while time.time() < deadline:
            if not self.stale_cameras(max_age):
                return True
            time.sleep(0.05)
        return False

    def clear_cache(self):
        for st in self._streams.values():
            with st._lock:
                st.rgb = st.depth = st.rgb_stamp = st.depth_stamp = None

    def destroy_node(self):
        """Named for CameraHub compatibility."""
        self.shutdown()

    def shutdown(self):
        for st in self._streams.values():
            st.stop()


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------
class EnumerationError(RuntimeError):
    """The SDK could not enumerate devices at all.

    Not "no cameras attached" -- the query itself failed, e.g.
    "RS4XX: RGB modules inconsistency - 2 found", which a wedged D4xx raises
    and which aborts the whole query, hiding every other camera too. It does
    not self-heal; the device needs a USB reset.
    """


def enumerate_devices(raise_on_error: bool = False) -> list[dict]:
    """Serials the SDK can actually open right now.

    Returns [] when enumeration fails, rather than propagating. This is a
    diagnostic helper: callers use it to decide what to report, and a helper
    that crashes its caller is worse than one that reports nothing.
    """
    import pyrealsense2 as rs
    try:
        devices = list(rs.context().query_devices())
    except Exception as e:
        if raise_on_error:
            raise EnumerationError(str(e)) from e
        print(f"[realsense] device enumeration failed: {e}")
        return []
    out = []
    for d in devices:
        try:
            out.append({
                "name": d.get_info(rs.camera_info.name),
                "serial": d.get_info(rs.camera_info.serial_number),
                "usb": d.get_info(rs.camera_info.usb_type_descriptor),
                "firmware": d.get_info(rs.camera_info.firmware_version),
            })
        except Exception:
            continue          # one unreadable device must not hide the rest
    return out


def enumeration_error() -> str | None:
    """The enumeration failure message, or None if enumeration works."""
    try:
        enumerate_devices(raise_on_error=True)
        return None
    except EnumerationError as e:
        return str(e)


def missing_devices(serials) -> list[str]:
    """Which of `serials` the SDK cannot see.

    A serial listed by `lsusb` but missing here is wedged: enumerated by the
    kernel but unopenable. No amount of retrying in software fixes that; the
    USB device needs resetting.
    """
    have = {d["serial"] for d in enumerate_devices()}
    return [s for s in serials if s not in have]


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
CAMERAS_FILE = None            # set lazily to avoid a pathlib import at module load


def config_path():
    from pathlib import Path
    return Path(__file__).resolve().parent.parent / "config" / "cameras.yaml"


def load_camera_config(path=None) -> tuple[list, str]:
    """Read config/cameras.yaml -> ([CameraSpec, ...], inhand_camera).

    Returns ([], default) if the file is absent, so a caller can fall back
    rather than crash.
    """
    import yaml

    path = path or config_path()
    if not path.exists():
        return [], "cam4"
    data = yaml.safe_load(path.read_text()) or {}
    specs = []
    for name, cfg in (data.get("cameras") or {}).items():
        cfg = dict(cfg)
        serial = cfg.pop("serial", None)
        if serial is None:
            raise ValueError(f"{path.name}: camera {name!r} has no serial")
        specs.append(CameraSpec(name, str(serial), **cfg))
    return specs, data.get("inhand_camera", "cam4")
