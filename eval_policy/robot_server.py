#!/usr/bin/env python3
"""Pure robot/camera HTTP server — no diffusion-policy dependency.

Exposes two endpoints used by external clients:

  POST /get_obs   -> latest observation (camera RGB/D + robot state)
  POST /goto      -> command robot end-effector to a Cartesian pose

Plus a /health endpoint.

Payloads use base64-encoded numpy arrays (matches the convention of the other
crisp_py Flask servers like pink_ik_server.py).

Example:
  python eval_policy/robot_server.py \
      --camera-config /home/mingxi/mingxi_ws/handpi/diffusion_policy/robotool/robot_configs/camera_info.yaml \
      --port 7000
"""

import argparse
import base64
import io
import threading
import time

import numpy as np
import rclpy
from rclpy.executors import MultiThreadedExecutor
from flask import Flask, jsonify, request
from scipy.spatial.transform import Rotation as R, Slerp

from crisp_py.camera.pointcloud import ObservationManager
from crisp_py.gripper.gripper import Gripper, GripperConfig
from crisp_py.robot import Pose, Robot
from crisp_py.robot_config import FrankaConfig


# ---------------------------------------------------------------------------
# Encoding helpers (same convention used by diffusion_clients.py)
# ---------------------------------------------------------------------------
def encode_array(arr: np.ndarray | None) -> dict | None:
    if arr is None:
        return None
    arr = np.ascontiguousarray(arr)
    buf = io.BytesIO()
    np.save(buf, arr, allow_pickle=False)
    return {
        "data": base64.b64encode(buf.getvalue()).decode("ascii"),
        "dtype": str(arr.dtype),
        "shape": list(arr.shape),
    }


def decode_array(payload: dict) -> np.ndarray:
    raw = base64.b64decode(payload["data"])
    return np.load(io.BytesIO(raw), allow_pickle=False)


# ---------------------------------------------------------------------------
# RobotServer — wraps Robot + Gripper + ObservationManager and exposes the API.
# ---------------------------------------------------------------------------
class RobotServer:

    def __init__(self,
                 camera_config: str,
                 ctrl_space: str = "cartesian",
                 ft_sensor_on: bool = False,
                 do_home: bool = True):
        self.ctrl_space = ctrl_space

        # Gripper
        print("Initializing gripper...")
        gripper_cfg = GripperConfig.from_yaml("./config/gripper_robotiq.yaml")
        self.gripper = Gripper(gripper_config=gripper_cfg)
        self.gripper.wait_until_ready()
        self.gripper.set_target(1.0)

        # Robot
        print("Initializing robot...")
        self.robot = Robot(namespace="", robot_config=FrankaConfig())
        self.robot.wait_until_ready()

        if do_home:
            print("Going to home position...")
            self.robot.home()

        # Switch to the requested controller
        if ctrl_space == "joint":
            self.robot.controller_switcher_client.switch_controller(
                "joint_impedance_controller")
            self.robot.joint_controller_parameters_client.load_param_config(
                file_path="config/control/joint_impedance_controller.yaml")
        else:
            self.robot.controller_switcher_client.switch_controller(
                "cartesian_impedance_controller")
            self.robot.cartesian_controller_parameters_client.load_param_config(
                file_path="config/control/default_cartesian_impedance.yaml")

        # Unified observation manager (cameras + joint states + gripper + FT)
        print(f"Loading ObservationManager from {camera_config}")
        self.obs_manager = ObservationManager(
            camera_config,
            robot=self.robot,
            gripper=self.gripper,
            ft_sensor_on=ft_sensor_on,
        )

        self._executor = MultiThreadedExecutor(num_threads=4)
        self._executor.add_node(self.obs_manager)
        self._spin_thread = threading.Thread(
            target=self._executor.spin, daemon=True)
        self._spin_thread.start()

        print("Waiting for sensor streams...")
        while not self.obs_manager.is_ready:
            time.sleep(0.1)

        # Tunable interpolation count for goto()
        self.n_interpolation = 10
        self._goto_lock = threading.Lock()
        print("RobotServer ready.")

    # -- observation -----------------------------------------------------
    def get_obs(self) -> dict:
        """Return the latest observation as a JSON-serialisable dict."""
        obs = self.obs_manager.get_obs()
        return {
            "timestamp": obs["timestamp"],
            "rgb": [encode_array(x) for x in obs["rgb"]],
            "depth": [encode_array(x) for x in obs["depth"]],
            "inhand_rgb": encode_array(obs["inhand_rgb"]),
            "inhand_depth": encode_array(obs["inhand_depth"]),
            "joint_values": encode_array(obs["joint_values"]),
            "ee_position": encode_array(obs["ee_position"]),
            "ee_quat_xyzw": encode_array(obs["ee_quat_xyzw"]),
            "gripper_value": obs["gripper_value"],
            "gripper_state": encode_array(obs["gripper_state"]),
            "ft_wrench": encode_array(obs["ft_wrench"]),
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

    def shutdown(self):
        try:
            self.obs_manager.destroy_node()
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
        return jsonify({"ok": True})

    @app.route("/get_obs", methods=["GET", "POST"])
    def get_obs():
        return jsonify(server.get_obs())

    @app.route("/goto", methods=["POST"])
    def goto():
        body = request.get_json(force=True)
        position = decode_array(body["position"]) if isinstance(body.get("position"), dict) \
            else np.asarray(body["position"], dtype=np.float64)
        quat = body.get("quat_xyzw")
        if isinstance(quat, dict):
            quat = decode_array(quat)
        elif quat is not None:
            quat = np.asarray(quat, dtype=np.float64)
        try:
            result = server.goto(
                position=position,
                quat_xyzw=quat,
                use_move_to=bool(body.get("use_move_to", False)),
                speed=float(body.get("speed", 0.05)),
                gripper=body.get("gripper"),
            )
            return jsonify(result)
        except Exception as e:
            return jsonify({"ok": False, "error": repr(e)}), 500

    return app


def parse_args():
    p = argparse.ArgumentParser(description="Pure robot+camera HTTP server")
    p.add_argument("--camera-config", default='camera_info.yaml',
                   help="Path to camera_info.yaml (ObservationManager config)")
    p.add_argument("--ctrl-space", default="cartesian",
                   choices=["cartesian", "joint"])
    p.add_argument("--ft-sensor-on", action="store_true",
                   help="Require FT sensor msgs before becoming ready")
    p.add_argument("--no-home", action="store_true",
                   help="Skip robot.home() at startup")
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=7000)
    return p.parse_args()


def main():
    args = parse_args()
    rclpy.init()
    server = None
    try:
        server = RobotServer(
            camera_config=args.camera_config,
            ctrl_space=args.ctrl_space,
            ft_sensor_on=args.ft_sensor_on,
            do_home=not args.no_home,
        )
        app = build_app(server)
        print(f"Listening on http://{args.host}:{args.port}")
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
