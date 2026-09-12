#!/usr/bin/env python3
"""Pure robot/camera HTTP server — no diffusion-policy dependency.

Endpoints:

  POST /get_obs      -> full observation (camera RGB/D + robot state). Several
                        MB per call; use /get_state in a control loop.
  POST /get_state    -> proprioception only (pose, joints, gripper, FT wrench)
  POST /servo        -> set one target pose, non-blocking. For control loops.
  POST /servo_joint  -> set one joint target, non-blocking (--ctrl-space joint)
  POST /goto         -> interpolated/blocking move. For scripted setup moves.
  POST /set_gripper  -> open/close the gripper
  POST /pause        -> hold: motion commands are accepted but not applied
  POST /resume       -> apply motion commands again
  POST /home         -> home the robot and restore the impedance controller
  GET  /panel        -> live HTML panel (camera feeds, EE pose, gripper, reset)
  GET  /camera/<cam>.jpg -> latest JPEG frame from cam1..cam4
  POST /cameras/stop -> stop the realsense nodes, freeing the USB devices
  POST /cameras/start-> relaunch them
  GET  /cameras/status
  GET  /health

Payloads use base64-encoded numpy arrays (matches the convention of the other
crisp_py Flask servers like pink_ik_server.py).

Example:
  python -m control.robot_server --cameras cam1,cam4 --port 7000
"""

import argparse
import os
import threading
import time
from pathlib import Path

import numpy as np
import rclpy
import yaml
from flask import Flask, Response, jsonify, request
from scipy.spatial.transform import Rotation as R, Slerp

from control.home_pose import load_home_config, load_time_to_home
from control.robot_io import as_vector, encode_array
from crisp_py.gripper.gripper import Gripper, GripperConfig
from crisp_py.robot import Pose, Robot
from crisp_py.robot_config import FrankaConfig



def load_robot_config() -> FrankaConfig:
    """FrankaConfig with the home pose from config/home_pose.yaml."""
    config = FrankaConfig()
    config.home_config = load_home_config()
    config.time_to_home = load_time_to_home(config.time_to_home)
    pretty = ", ".join(f"{v:+.4f}" for v in config.home_config)
    print(f"Home pose: [{pretty}]  (config/home_pose.yaml)")
    return config


# ---------------------------------------------------------------------------
# RobotServer — wraps Robot + Gripper + CameraHub and exposes the API.
# ---------------------------------------------------------------------------
class RobotServer:

    def __init__(self,
                 cameras: list[str] | None = None,
                 ctrl_space: str = "cartesian",
                 ft_sensor: bool = False,
                 do_home: bool = False,
                 gripper_stroke_mm: float = 85.0,
                 inhand_camera: str | None = None,
                 compliance: str = "normal",
                 camera_backend: str = "realsense",
                 camera_serials: dict | None = None):
        self.ctrl_space = ctrl_space
        if (ctrl_space, compliance) not in self.IMPEDANCE_CONFIGS:
            raise ValueError(f"no impedance profile for {ctrl_space}/{compliance}")
        self.compliance = compliance
        # Nominal full-open stroke, used only to turn the normalised 0..1
        # gripper value into a millimetre reading for the panel. Robotiq 2F-85
        # is 85 mm. This is derived, not measured.
        self.gripper_stroke_mm = gripper_stroke_mm
        self.inhand_camera = inhand_camera
        # Paused = motion commands are accepted but not applied, so the arm
        # holds its last target while a running policy keeps looping. Refusing
        # them instead would crash the caller mid-rollout.
        self.paused = False
        self.paused_at = None

        # Gripper
        print("Initializing gripper...")
        gripper_cfg = GripperConfig.from_yaml("./config/gripper_robotiq.yaml")
        self.gripper = Gripper(gripper_config=gripper_cfg)
        self.gripper.wait_until_ready()
        self.gripper.set_target(1.0)

        # Robot
        print("Initializing robot...")
        self.robot = Robot(namespace="", robot_config=load_robot_config())
        self.robot.wait_until_ready()

        # Cameras are set up BEFORE any motion: a camera problem must not
        # be discovered after the arm has already homed.
        # Cameras: one independent subscription set per camera, no
        # cross-camera synchroniser, so a subset launch works fine.
        # Observations come from the shared Observer, so the server is one
        # consumer of it rather than a parallel implementation. Swapping
        # camera backends is the same one argument here as anywhere else.
        from control.observer import Observer
        self.observer = Observer(
            backend=camera_backend,
            cameras=cameras,
            serials=camera_serials,
            inhand_camera=inhand_camera,
            ft_sensor=ft_sensor,
            robot=self.robot, gripper=self.gripper,
        )
        # Resolve now that the Observer has read config/cameras.yaml.
        self.inhand_camera = self.inhand_camera or self.observer.inhand_camera
        self.cameras = self.observer.cameras      # panel + JPEG endpoints

        self.joint_state_sub = self.observer.proprio

        if do_home:
            print("Going to home position — THE ARM WILL MOVE...")
            self.robot.home()
        else:
            print("Skipping home (pass --home to move the arm to its home pose)")

        self._switch_to_impedance_controller()


        print("Waiting for joint/gripper streams...")
        self.observer.wait_until_ready(timeout=20.0, require_cameras=False)
        if self.cameras.camera_names:
            if self.cameras.wait_for_any(timeout=15.0):
                print(f"Cameras streaming: {', '.join(self.cameras.ready_cameras())}")
            else:
                print("WARNING: no camera frames yet; /get_obs will report them missing")

        # Tunable interpolation count for goto()
        self.n_interpolation = 10
        self._goto_lock = threading.Lock()
        print("RobotServer ready.")

    # Impedance profiles. "compliant" caps the force the controller applies
    # against an obstruction (F = k_pos x error_clip) below a Franka's
    # collision reflex, so a slow collision stops the arm instead of faulting
    # it. See config/control/compliant_*.yaml for the arithmetic.
    IMPEDANCE_CONFIGS = {
        ("cartesian", "normal"):   "config/control/default_cartesian_impedance.yaml",
        ("cartesian", "compliant"): "config/control/compliant_cartesian_impedance.yaml",
        ("joint", "normal"):       "config/control/joint_impedance_controller.yaml",
        ("joint", "compliant"):    "config/control/compliant_joint_impedance.yaml",
    }

    def _switch_to_impedance_controller(self):
        """Activate the impedance controller matching self.ctrl_space."""
        profile = getattr(self, "compliance", "normal")
        config = self.IMPEDANCE_CONFIGS[(self.ctrl_space, profile)]
        print(f"Impedance: {self.ctrl_space}/{profile}  ({config})")
        if self.ctrl_space == "joint":
            self.robot.controller_switcher_client.switch_controller(
                "joint_impedance_controller")
            self.robot.joint_controller_parameters_client.load_param_config(
                file_path=config)
        else:
            self.robot.controller_switcher_client.switch_controller(
                "cartesian_impedance_controller")
            self.robot.cartesian_controller_parameters_client.load_param_config(
                file_path=config)

    # -- state (cheap) ---------------------------------------------------
    def get_state(self) -> dict:
        """Robot proprioception only — no camera frames.

        /get_obs base64-encodes 3 RGB + 3 depth + in-hand RGB/D on every call,
        which is a few MB per request. A control loop that only needs the pose
        should poll this instead.
        """
        state = self.observer.get_state()
        ee = self.robot.end_effector_pose
        gripper_value = state.gripper_value
        return {
            "timestamp": time.time(),
            "paused": self.paused,
            "ee_position": encode_array(np.asarray(ee.position, dtype=np.float32)),
            "ee_quat_xyzw": encode_array(ee.orientation.as_quat().astype(np.float32)),
            "joint_values": encode_array(
                np.asarray(self.robot.joint_values, dtype=np.float32)),
            "gripper_value": gripper_value,
            "gripper_width_mm": (None if gripper_value is None
                                 else gripper_value * self.gripper_stroke_mm),
            "gripper_state": encode_array(state.gripper_state),
            "ft_wrench": encode_array(state.ft_wrench),
        }

    # -- observation -----------------------------------------------------
    def get_obs(self, include_depth: bool = True) -> dict:
        """Return the latest observation as a JSON-serialisable dict."""
        # Whatever each camera has right now. No blocking wait: a camera that
        # is down reports null instead of wedging the request forever.
        # Uses only the accessors BOTH camera backends share -- rgb(), depth(),
        # status(). CameraHub also has frame(), RealsenseCameras does not, and
        # depending on it silently bound this endpoint to the ROS backend.
        status = self.cameras.status()
        camera_payload = {}
        for name in self.cameras.camera_names:
            info = status.get(name, {})
            stale = bool(info.get("stale", True))
            # A stale frame is sent as null: a camera that fell off the USB bus
            # leaves its last image in memory forever, and handing that to a
            # policy is worse than handing it nothing.
            camera_payload[name] = {
                "rgb": encode_array(None if stale else self.cameras.rgb(name)),
                "depth": encode_array(None if (stale or not include_depth)
                                      else self.cameras.depth(name)),
                "age_s": info.get("age_s"),
                "stale": stale,
            }

        # Proprioception comes from get_state() rather than being rebuilt here,
        # so the two endpoints cannot drift apart.
        payload = self.get_state()
        payload.update({
            "cameras": camera_payload,
            "camera_names": list(self.cameras.camera_names),
            "inhand_camera": self.inhand_camera,
        })
        return payload

    # -- panel -----------------------------------------------------------
    @property
    def camera_names(self) -> list[str]:
        return list(self.cameras.camera_names)

    def camera_jpeg(self, cam_name: str, quality: int = 70) -> bytes | None:
        """Latest frame from one camera as JPEG, or None if it has none yet.

        Non-blocking by construction: CameraHub just hands back whatever it
        last received, so a camera that is down costs a 503 rather than
        wedging a Flask worker.
        """
        import cv2

        # max_age=inf: the panel wants to show the last frame with a STALE
        # badge rather than a blank tile. Only the UI may do this.
        rgb = self.cameras.rgb(cam_name, max_age=float("inf"))
        if rgb is None:
            return None
        bgr = cv2.cvtColor(np.asarray(rgb), cv2.COLOR_RGB2BGR)
        ok, buf = cv2.imencode(".jpg", bgr,
                               [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
        return buf.tobytes() if ok else None

    # -- camera process control ------------------------------------------
    # Overridable with --camera-launch; the default matches start_all.sh.
    CAMERA_LAUNCH = "control/launch_cameras.launch.py"

    @staticmethod
    def camera_node_pids() -> list[int]:
        """PIDs of running realsense2_camera_node processes.

        Matched by executable via /proc/<pid>/exe. Matching by name fails --
        Linux truncates /proc/<pid>/comm to 15 chars ("realsense2_came") -- and
        matching the command line would also catch unrelated processes that
        merely mention the node.
        """
        pids = []
        for entry in os.scandir("/proc"):
            if not entry.name.isdigit():
                continue
            try:
                exe = os.readlink(f"/proc/{entry.name}/exe")
            except OSError:
                continue
            if os.path.basename(exe) == "realsense2_camera_node":
                pids.append(int(entry.name))
        return sorted(pids)

    def _launch_pids(self) -> list[int]:
        """PIDs of the `ros2 launch <camera launch file>` parents.

        Matched on the launch file's basename so this keeps working whichever
        launch file is in use (launch_cameras, launch_three_cameras, ...).
        """
        stem = os.path.basename(self.CAMERA_LAUNCH).replace(".launch.py", "")
        pids = []
        for entry in os.scandir("/proc"):
            if not entry.name.isdigit():
                continue
            try:
                with open(f"/proc/{entry.name}/cmdline", "rb") as fh:
                    cmdline = fh.read().decode("utf-8", "replace")
            except OSError:
                continue
            if stem in cmdline and "ros2" in cmdline:
                pids.append(int(entry.name))
        return sorted(pids)

    def cameras_status(self) -> dict:
        pids = self.camera_node_pids()
        return {"ok": True, "running": bool(pids), "node_pids": pids,
                "n_nodes": len(pids), "launch_pids": self._launch_pids()}

    def stop_cameras(self, timeout: float = 10.0) -> dict:
        """Stop the realsense nodes so the USB devices become claimable.

        The nodes -- not this server -- hold the cameras: realsense USB claims
        are exclusive, so realsense-viewer cannot open a device while they run.
        This server only subscribes to their ROS topics.
        """
        import signal

        before = self.camera_node_pids()
        targets = self._launch_pids() + before
        if not targets:
            return {"ok": True, "stopped": 0, "message": "no camera nodes were running"}

        for pid in targets:
            try:
                os.kill(pid, signal.SIGTERM)
            except OSError:
                pass

        deadline = time.time() + timeout
        while time.time() < deadline and self.camera_node_pids():
            time.sleep(0.2)

        remaining = self.camera_node_pids()
        if remaining:
            for pid in remaining:
                try:
                    os.kill(pid, signal.SIGKILL)
                except OSError:
                    pass
            time.sleep(1.0)
            remaining = self.camera_node_pids()

        # Drop cached frames so the panel shows "no frame" instead of a stale
        # image from before the shutdown.
        try:
            self.cameras.clear_cache()
        except Exception:
            pass

        return {
            "ok": not remaining,
            "stopped": len(before) - len(remaining),
            "still_running": remaining,
            "message": ("cameras released" if not remaining
                        else f"{len(remaining)} node(s) would not exit"),
        }

    def start_cameras(self) -> dict:
        """Relaunch the camera nodes (the inverse of stop_cameras)."""
        import subprocess

        if self.camera_node_pids():
            return {"ok": True, "message": "cameras already running",
                    "n_nodes": len(self.camera_node_pids())}
        if not os.path.exists(self.CAMERA_LAUNCH):
            return {"ok": False, "error": f"{self.CAMERA_LAUNCH} not found "
                                          f"(run the server from the repo root)"}
        log = open("/tmp/crisp-camera-relaunch.log", "ab", buffering=0)
        subprocess.Popen(["ros2", "launch", self.CAMERA_LAUNCH],
                         stdout=log, stderr=log, start_new_session=True)
        return {"ok": True, "message": "launch started; cameras take ~20-40 s "
                                       "to come up", "log": "/tmp/crisp-camera-relaunch.log"}

    # -- pause -----------------------------------------------------------
    def set_paused(self, paused: bool) -> dict:
        was = self.paused
        self.paused = bool(paused)
        self.paused_at = time.time() if self.paused else None
        if was != self.paused:
            print("PAUSED — motion commands will be ignored" if self.paused
                  else "RESUMED — motion commands apply again")
        return self.pause_status()

    def pause_status(self) -> dict:
        return {
            "ok": True,
            "paused": self.paused,
            "paused_for_s": (None if not self.paused or self.paused_at is None
                             else round(time.time() - self.paused_at, 1)),
        }

    # -- control ---------------------------------------------------------
    def goto(self,
             position: np.ndarray,
             quat_xyzw: np.ndarray | None = None,
             use_move_to: bool = False,
             speed: float = 0.05,
             gripper: float | None = None) -> dict:
        """Move EE to target Cartesian pose.

        Args:
            position:   [x, y, z] in robot base frame.
            quat_xyzw:  optional orientation quaternion. If None, holds current.
            use_move_to:if True, uses robot.move_to (linear blocking move).
                        Otherwise, sets target via interpolated set_target stream.
            speed:      m/s (only used when use_move_to=True).
            gripper:    optional 0..1 gripper command (1=open, 0=close).
        """
        if self.ctrl_space != "cartesian":
            raise RuntimeError(
                "goto requires --ctrl-space cartesian; this server is running "
                f"in '{self.ctrl_space}' space and pose targets are ignored")
        if self.paused:
            return {"ok": True, "paused": True, "applied": False}
        with self._goto_lock:
            current = self.robot.end_effector_pose
            current_pos = np.asarray(current.position, dtype=np.float64)
            current_orient = current.orientation

            target_pos = np.asarray(position, dtype=np.float64)
            target_orient = (R.from_quat(np.asarray(quat_xyzw, dtype=np.float64))
                             if quat_xyzw is not None else current_orient)

            if use_move_to:
                self.robot.move_to(
                    pose=Pose(position=target_pos, orientation=target_orient),
                    speed=speed,
                )
            else:
                # SLERP orientation, linear position; small delay between steps
                key_rots = R.concatenate([current_orient, target_orient])
                slerp = Slerp([0, 1], key_rots)
                steps = max(int(self.n_interpolation), 1)
                for s in range(1, steps + 1):
                    a = s / steps
                    pos_i = (1 - a) * current_pos + a * target_pos
                    orient_i = slerp(a)
                    self.robot.set_target(pose=Pose(position=pos_i,
                                                   orientation=orient_i))
                    time.sleep(0.01)

            if gripper is not None:
                self.gripper.set_target(float(np.clip(gripper, 0.0, 1.0)))

            new_pose = self.robot.end_effector_pose
            return {
                "ok": True,
                "ee_position": new_pose.position.tolist(),
                "ee_quat_xyzw": new_pose.orientation.as_quat().tolist(),
            }

    def servo(self,
              position: np.ndarray,
              quat_xyzw: np.ndarray | None = None) -> dict:
        """Publish a single target pose and return immediately.

        This is the endpoint a teleop or policy control loop should use. Unlike
        goto(), it takes no lock and runs no interpolation loop, so it does not
        block the caller for ~n_interpolation * 10 ms. Smoothing is the
        impedance controller's job.
        """
        # Symmetric with servo_joint: in joint space the Cartesian controller
        # is inactive and /target_pose has no consumer, so the command would be
        # silently dropped -- the arm just sits there while the gripper still
        # responds, which reads as "the spacemouse is broken".
        if self.ctrl_space != "cartesian":
            raise RuntimeError(
                "servo requires --ctrl-space cartesian; this server is running "
                f"in '{self.ctrl_space}' space, where the cartesian controller "
                "is inactive and pose targets are ignored. Use servo_joint, or "
                "restart the server with --ctrl-space cartesian")
        target_orient = (R.from_quat(np.asarray(quat_xyzw, dtype=np.float64))
                         if quat_xyzw is not None
                         else self.robot.end_effector_pose.orientation)
        if self.paused:
            # Do not publish a new target: the impedance controller keeps the
            # last one, so the arm holds rather than drifting or jumping.
            return {"ok": True, "paused": True, "applied": False}
        self.robot.set_target(pose=Pose(
            position=np.asarray(position, dtype=np.float64),
            orientation=target_orient,
        ))
        return {"ok": True, "applied": True}

    def servo_joint(self, q: np.ndarray) -> dict:
        """Publish a joint-position target and return immediately.

        The joint-space counterpart of servo(). Requires the joint impedance
        controller, i.e. the server started with --ctrl-space joint; the
        Cartesian controller ignores joint targets.
        """
        q = np.asarray(q, dtype=np.float64)
        if q.shape != (7,):
            raise ValueError(f"expected 7 joint values, got shape {q.shape}")
        if self.ctrl_space != "joint":
            raise RuntimeError(
                "servo_joint requires --ctrl-space joint; this server is "
                f"running in '{self.ctrl_space}' space and the target would "
                "be ignored by the active controller"
            )
        if self.paused:
            return {"ok": True, "paused": True, "applied": False}
        self.robot.set_target_joint(q)
        return {"ok": True, "applied": True}

    def set_gripper(self, value: float) -> dict:
        """Command the gripper. 1.0 = open, 0.0 = closed.

        Also held while paused: a gripper that closes on a paused arm is as
        much of a surprise as one that moves.
        """
        if self.paused:
            return {"ok": True, "paused": True, "applied": False}
        self.gripper.set_target(float(np.clip(value, 0.0, 1.0)))
        return {"ok": True, "gripper": float(np.clip(value, 0.0, 1.0))}

    def home(self) -> dict:
        """Home the robot, then restore the impedance controller.

        robot.home() leaves the joint_trajectory_controller active, so the
        impedance controller has to be switched back on before the next
        servo()/goto() call will do anything.
        """
        with self._goto_lock:
            self.gripper.set_target(1.0)
            self.robot.home()
            self._switch_to_impedance_controller()
            self.robot.wait_until_ready()
            pose = self.robot.end_effector_pose
            return {
                "ok": True,
                "ee_position": pose.position.tolist(),
                "ee_quat_xyzw": pose.orientation.as_quat().tolist(),
            }

    def shutdown(self):
        # Release the cameras first: an un-stopped RealSense pipeline is what
        # leaves a device unopenable until it is physically replugged.
        try:
            self.observer.close()
        except Exception:
            pass
        try:
            self.cameras.destroy_node()
        except Exception:
            pass
        try:
            self.robot.shutdown()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Flask app
# ---------------------------------------------------------------------------
def build_app(server: RobotServer) -> Flask:
    app = Flask(__name__)

    @app.route("/health", methods=["GET"])
    def health():
        return jsonify({
            "ok": True,
            "paused": getattr(server, "paused", False),
            "ctrl_space": getattr(server, "ctrl_space", None),
            "cameras": getattr(server, "camera_names", []),
            "inhand_camera": getattr(server, "inhand_camera", None),
        })

    def _fail(exc):
        return jsonify({"ok": False, "error": repr(exc)}), 500

    @app.route("/get_obs", methods=["GET", "POST"])
    def get_obs():
        try:
            body = request.get_json(silent=True) or {}
            return jsonify(server.get_obs(
                include_depth=bool(body.get("include_depth", True))))
        except Exception as e:
            return _fail(e)

    @app.route("/get_state", methods=["GET", "POST"])
    def get_state():
        try:
            return jsonify(server.get_state())
        except Exception as e:
            return _fail(e)

    @app.route("/goto", methods=["POST"])
    def goto():
        body = request.get_json(force=True)
        try:
            return jsonify(server.goto(
                position=as_vector(body["position"]),
                quat_xyzw=as_vector(body.get("quat_xyzw")),
                use_move_to=bool(body.get("use_move_to", False)),
                speed=float(body.get("speed", 0.05)),
                gripper=body.get("gripper"),
            ))
        except Exception as e:
            return _fail(e)

    @app.route("/servo", methods=["POST"])
    def servo():
        body = request.get_json(force=True)
        try:
            return jsonify(server.servo(
                position=as_vector(body["position"]),
                quat_xyzw=as_vector(body.get("quat_xyzw")),
            ))
        except Exception as e:
            return _fail(e)

    @app.route("/servo_joint", methods=["POST"])
    def servo_joint():
        body = request.get_json(force=True)
        try:
            return jsonify(server.servo_joint(as_vector(body["q"])))
        except Exception as e:
            return _fail(e)

    @app.route("/set_gripper", methods=["POST"])
    def set_gripper():
        body = request.get_json(force=True)
        try:
            return jsonify(server.set_gripper(float(body["value"])))
        except Exception as e:
            return _fail(e)

    @app.route("/panel", methods=["GET"])
    def panel():
        html = Path(__file__).with_name("panel.html").read_text()
        return Response(html, mimetype="text/html")

    @app.route("/camera/<cam>.jpg", methods=["GET"])
    def camera(cam):
        if cam not in server.camera_names:
            return jsonify({"ok": False,
                            "error": f"unknown camera {cam!r}",
                            "available": server.camera_names}), 404
        try:
            quality = int(request.args.get("q", 70))
            jpeg = server.camera_jpeg(cam, quality=quality)
        except Exception as e:
            return _fail(e)
        if jpeg is None:
            return jsonify({"ok": False, "error": f"{cam} has no frame yet"}), 503
        return Response(jpeg, mimetype="image/jpeg",
                        headers={"Cache-Control": "no-store"})

    @app.route("/cameras/status", methods=["GET", "POST"])
    def cameras_status():
        try:
            status = server.cameras_status()
            status["names"] = server.camera_names
            status["inhand_camera"] = server.inhand_camera
            status["frames"] = server.cameras.status()
            status["fresh"] = server.cameras.ready_cameras()
            status["stale"] = server.cameras.stale_cameras()
            # `running` counts processes; a node can be alive while its camera
            # has silently dropped off the bus, so report both.
            status["healthy"] = (bool(status["names"])
                                 and not status["stale"])
            return jsonify(status)
        except Exception as e:
            return _fail(e)

    @app.route("/cameras/stop", methods=["POST"])
    def cameras_stop():
        try:
            return jsonify(server.stop_cameras())
        except Exception as e:
            return _fail(e)

    @app.route("/cameras/start", methods=["POST"])
    def cameras_start():
        try:
            return jsonify(server.start_cameras())
        except Exception as e:
            return _fail(e)

    @app.route("/pause", methods=["POST"])
    def pause():
        try:
            return jsonify(server.set_paused(True))
        except Exception as e:
            return _fail(e)

    @app.route("/resume", methods=["POST"])
    def resume():
        try:
            return jsonify(server.set_paused(False))
        except Exception as e:
            return _fail(e)

    @app.route("/pause", methods=["GET"])
    def pause_state():
        try:
            return jsonify(server.pause_status())
        except Exception as e:
            return _fail(e)

    @app.route("/home", methods=["POST"])
    def home():
        try:
            return jsonify(server.home())
        except Exception as e:
            return _fail(e)

    return app


def parse_camera_serials(text: str | None) -> dict:
    """'cam1=SERIAL@848x480[+depth],...' -> {name: {serial, width, height, depth}}.

    Depth is off unless '+depth' is given, which matches the launch file's
    default for the ROS backend. Without this the two backends disagreed: the
    launch file could turn depth off, the direct backend always turned it on.
    """
    if not text:
        return {}
    out = {}
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        if "=" not in part:
            raise ValueError(f"--camera-serials: expected name=serial[@WxH], got {part!r}")
        name, rest = part.split("=", 1)
        cfg = {"depth": False}
        if rest.endswith("+depth"):
            cfg["depth"] = True
            rest = rest[: -len("+depth")]
        if "@" in rest:
            rest, res = rest.split("@", 1)
            if "x" not in res:
                raise ValueError(f"--camera-serials: expected WxH after '@', got {res!r}")
            w, h = res.split("x", 1)
            cfg["width"], cfg["height"] = int(w), int(h)
        cfg["serial"] = rest
        out[name.strip()] = cfg
    return out


def parse_args():
    p = argparse.ArgumentParser(description="Pure robot+camera HTTP server")
    p.add_argument("--cameras", default=None,
                   help="Comma-separated camera names, e.g. cam1,cam4. "
                        "Default: auto-discover from the ROS graph.")
    p.add_argument("--camera-backend", choices=["realsense", "ros"], default="realsense",
                   help="'realsense' (default) opens the devices directly via "
                        "the SDK, configured by config/cameras.yaml: no camera "
                        "launch needed, and a stalled device can be reset. "
                        "'ros' subscribes to camera topics instead, which is "
                        "what other ROS consumers such as the recorder need. "
                        "Only one process may own a device, so the two are "
                        "mutually exclusive.")
    p.add_argument("--camera-serials", default=None,
                   help="for --camera-backend realsense: "
                        "name=serial@WxH[+depth][,...]  e.g. "
                        "cam1=239222303046@848x480,cam4=218722271574@480x270. "
                        "Depth is off unless +depth, matching the launch file.")
    p.add_argument("--inhand-camera", default=None,
                   help="Which camera is the in-hand one "
                        "(default: inhand_camera from config/cameras.yaml)")
    p.add_argument("--compliance", choices=["normal", "compliant"], default="normal",
                   help="'compliant' caps the force the controller applies "
                        "against an obstruction (20 N instead of 100 N in "
                        "cartesian, 3 N.m instead of 5 in joint), so a slow "
                        "collision stops the arm rather than faulting it. "
                        "Costs tracking accuracy.")
    p.add_argument("--ctrl-space", default="cartesian",
                   choices=["cartesian", "joint"])
    p.add_argument("--ft-sensor-on", action="store_true",
                   help="Require FT sensor msgs before becoming ready")
    p.add_argument("--home", action="store_true",
                   help="Move the arm to its home pose at startup. Off by "
                        "default: starting a server should not move hardware.")
    p.add_argument("--no-home", action="store_true",
                   help=argparse.SUPPRESS)  # accepted for compatibility; now the default
    p.add_argument("--camera-launch", default=None,
                   help="ros2 launch file used by /cameras/start "
                        "(default: control/launch_cameras.launch.py)")
    p.add_argument("--gripper-stroke-mm", type=float, default=85.0,
                   help="Full-open gripper stroke, for the panel's mm readout")
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=7000)
    return p.parse_args()


def _install_signal_handlers(get_server):
    """Exit cleanly on SIGTERM/SIGINT.

    Without this the process relies on the default disposition, and a
    supervisor is left escalating to SIGKILL. That matters beyond tidiness:
    SIGKILL gives the RealSense pipelines no chance to stop, which is a
    reliable way to leave a camera wedged until it is replugged.
    """
    import signal

    state = {"stopping": False}

    def handler(signum, _frame):
        name = signal.Signals(signum).name
        if state["stopping"]:
            print(f"\n{name} again — exiting immediately")
            os._exit(1)
        state["stopping"] = True
        print(f"\n{name} received; releasing cameras and shutting down...")
        server = get_server()
        if server is not None:
            try:
                server.shutdown()
            except Exception as e:
                print(f"  shutdown error (continuing): {e!r}")
        try:
            rclpy.shutdown()
        except Exception:
            pass
        os._exit(0)

    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, handler)


def main():
    args = parse_args()
    rclpy.init()
    server = None
    _install_signal_handlers(lambda: server)
    try:
        server = RobotServer(
            cameras=([c.strip() for c in args.cameras.split(",") if c.strip()]
                     if args.cameras else None),
            inhand_camera=args.inhand_camera,
            camera_backend=args.camera_backend,
            camera_serials=parse_camera_serials(args.camera_serials),
            ctrl_space=args.ctrl_space,
            compliance=args.compliance,
            ft_sensor=args.ft_sensor_on,
            do_home=args.home and not args.no_home,
            gripper_stroke_mm=args.gripper_stroke_mm,
        )
        if args.camera_launch:
            server.CAMERA_LAUNCH = args.camera_launch
        app = build_app(server)
        shown = "localhost" if args.host in ("0.0.0.0", "") else args.host
        print(f"Listening on http://{args.host}:{args.port}")
        print(f"Panel:        http://{shown}:{args.port}/panel")
        app.run(host=args.host, port=args.port,
                debug=False, threaded=True, use_reloader=False)
    finally:
        if server is not None:
            server.shutdown()
        try:
            rclpy.shutdown()
        except Exception:
            pass


if __name__ == "__main__":
    main()
