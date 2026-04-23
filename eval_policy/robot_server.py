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
from flask import Flask, jsonify, request
from scipy.spatial.transform import Rotation as R, Slerp

from crisp_py.camera.pointcloud import PointCloudManager
from crisp_py.gripper.gripper import Gripper, GripperConfig
from crisp_py.robot import Pose, Robot
from crisp_py.robot_config import FrankaConfig

from sensor_msgs.msg import JointState
from geometry_msgs.msg import WrenchStamped


# ---------------------------------------------------------------------------
# Local joint-state subscriber (mirrors diff_eval_utils.ros_utils.JointStateSubscriber
# but stripped of diffusion-policy imports).
# ---------------------------------------------------------------------------
class JointStateSubscriber:
    """Subscribes to /joint_states + gripper state + optional FT sensor."""

    FRANKA_JOINTS = [
        "fr3_joint1", "fr3_joint2", "fr3_joint3", "fr3_joint4",
        "fr3_joint5", "fr3_joint6", "fr3_joint7",
    ]
    GRIPPER_JOINTS = ["gripper_joint"]

    def __init__(self, node, ft_sensor_on: bool = False):
        self._node = node
        self.ft_sensor_on = ft_sensor_on

        self._franka_received = False
        self._gripper_received = False
        self._ft_received = False

        self.franka_joint_array = None
        self.gripper_joint_array = None
        self.gripper_torque_array = None

        node.create_subscription(JointState, "/joint_states",
                                 self._franka_cb, 10)
        node.create_subscription(JointState, "/gripper/gripper_state",
                                 self._gripper_cb, 10)
        node.create_subscription(WrenchStamped,
                                 "/ft/robotiq_force_torque_sensor_broadcaster/wrench",
                                 self._ft_cb, 10)

    def _franka_cb(self, msg: JointState):
        arr = [p for n, p in zip(msg.name, msg.position) if n in self.FRANKA_JOINTS]
        if arr:
            self.franka_joint_array = np.asarray(arr, dtype=np.float32)
            self._franka_received = True

    def _gripper_cb(self, msg: JointState):
        arr = []
        for n, p in zip(msg.name, msg.position):
            if n in self.GRIPPER_JOINTS:
                arr.append(1.0 if p >= 0.95 else 0.0)
        if arr:
            self.gripper_joint_array = np.asarray(arr, dtype=np.float32)
            self._gripper_received = True

    def _ft_cb(self, msg: WrenchStamped):
        w = msg.wrench
        self.gripper_torque_array = np.asarray(
            [w.force.x, w.force.y, w.force.z,
             w.torque.x, w.torque.y, w.torque.z], dtype=np.float32)
        self._ft_received = True

    @property
    def is_ready(self) -> bool:
        if self.ft_sensor_on:
            return self._franka_received and self._gripper_received and self._ft_received
        return self._franka_received and self._gripper_received


# ---------------------------------------------------------------------------
# Encoding helpers (same convention used by diffusion_clients.py)
# ---------------------------------------------------------------------------
def encode_array(arr: np.ndarray) -> dict:
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
# RobotServer — wraps Robot + Gripper + PointCloudManager and exposes the API.
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

        # Cameras + extra joint state sub
        print(f"Loading PointCloudManager from {camera_config}")
        self.pcd_manager = PointCloudManager(camera_config)
        self.joint_state_sub = JointStateSubscriber(self.pcd_manager,
                                                    ft_sensor_on=ft_sensor_on)

        self._spin_thread = threading.Thread(
            target=rclpy.spin, args=(self.pcd_manager,), daemon=True)
        self._spin_thread.start()

        print("Waiting for sensor streams...")
        while not self.joint_state_sub.is_ready:
            time.sleep(0.1)

        # Tunable interpolation count for goto()
        self.n_interpolation = 10
        self._goto_lock = threading.Lock()
        print("RobotServer ready.")

    # -- observation -----------------------------------------------------
    def get_obs(self) -> dict:
        """Return the latest observation as a JSON-serialisable dict."""
        # Wait for at least one synchronized frame
        while not self.pcd_manager.rgb_images or not self.pcd_manager.depth_images:
            time.sleep(0.005)
        while self.pcd_manager.inhand_image is None:
            time.sleep(0.005)

        rgb = [np.asarray(img) for img in self.pcd_manager.rgb_images]
        depth = [np.asarray(img) for img in self.pcd_manager.depth_images]
        inhand_rgb = np.asarray(self.pcd_manager.inhand_image)
        inhand_depth = np.asarray(self.pcd_manager.inhand_depth)

        ee = self.robot.end_effector_pose
        ee_position = np.asarray(ee.position, dtype=np.float32)
        ee_quat_xyzw = ee.orientation.as_quat().astype(np.float32)

        joint_values = np.asarray(self.robot.joint_values, dtype=np.float32)

        gripper_value = float(self.gripper.value) if self.gripper.value is not None else None

        return {
            "timestamp": time.time(),
            "rgb": [encode_array(x) for x in rgb],            # 3 external cams
            "depth": [encode_array(x) for x in depth],        # 3 external cams (uint16, mm)
            "inhand_rgb": encode_array(inhand_rgb),
            "inhand_depth": encode_array(inhand_depth),       # meters (float)
            "joint_values": encode_array(joint_values),
            "ee_position": encode_array(ee_position),
            "ee_quat_xyzw": encode_array(ee_quat_xyzw),
            "gripper_value": gripper_value,
            "gripper_state": encode_array(self.joint_state_sub.gripper_joint_array)
                if self.joint_state_sub.gripper_joint_array is not None else None,
            "ft_wrench": encode_array(self.joint_state_sub.gripper_torque_array)
                if self.joint_state_sub.gripper_torque_array is not None else None,
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
            self.pcd_manager.destroy_node()
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
    p.add_argument("--camera-config", required=True,
                   help="Path to camera_info.yaml (PointCloudManager config)")
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
