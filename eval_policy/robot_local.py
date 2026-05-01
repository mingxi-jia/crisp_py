#!/usr/bin/env python3
"""Direct robot + camera controller — no HTTP server.

ObsManager runs cameras + joint_states + gripper on a single rclpy node.
The camera ApproximateTimeSynchronizer fires `_sync_cb`, which snapshots
the latest joint/gripper/FT values together with the synced RGB-D frames
into a single dict under one lock — so a caller of get_obs() always sees
images and joints from the same instant, on the same thread.

RobotLocal owns Robot + Gripper + ObsManager and exposes get_obs / goto
in-process (no Flask, no requests). Use it as a library or run this file
as a smoke test.
"""

import argparse
import threading
import time

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from cv_bridge import CvBridge
from message_filters import ApproximateTimeSynchronizer, Subscriber
from sensor_msgs.msg import Image, JointState
from geometry_msgs.msg import WrenchStamped
from scipy.spatial.transform import Rotation as R, Slerp

from crisp_py.gripper.gripper import Gripper, GripperConfig
from crisp_py.robot import Pose, Robot
from crisp_py.robot_config import FrankaConfig


FRANKA_JOINTS = [
    "fr3_joint1", "fr3_joint2", "fr3_joint3", "fr3_joint4",
    "fr3_joint5", "fr3_joint6", "fr3_joint7",
]
GRIPPER_JOINTS = ["gripper_joint"]


class ObsManager(Node):
    """Single-node observation manager.

    Cameras (cam3 RGB+depth, cam4 inhand RGB+depth) are synchronized via
    ApproximateTimeSynchronizer. Joint state, gripper state, and (optionally)
    FT wrench are plain subscriptions; their latest values are sampled
    inside the camera sync callback, so every published snapshot is taken
    atomically on the spin thread.
    """

    def __init__(self, ft_sensor_on: bool = False):
        super().__init__("obs_manager")
        self.bridge = CvBridge()
        self.ft_sensor_on = ft_sensor_on

        self._lock = threading.Lock()
        self._latest = None
        self.callback_count = 0

        # latest values from plain subs (sampled inside sync_cb)
        self.franka_joints = None
        self.gripper_state = None
        self.ft_wrench = None

        self.create_subscription(JointState, "/joint_states", self._joints_cb, 10)
        self.create_subscription(JointState, "/gripper/gripper_state", self._gripper_cb, 10)
        if ft_sensor_on:
            self.create_subscription(
                WrenchStamped,
                "/ft/robotiq_force_torque_sensor_broadcaster/wrench",
                self._ft_cb, 10,
            )

        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )
        rgb3 = Subscriber(self, Image, "/cam3/cam3/color/image_raw", qos_profile=qos)
        d3 = Subscriber(self, Image, "/cam3/cam3/aligned_depth_to_color/image_raw", qos_profile=qos)
        rgb4 = Subscriber(self, Image, "/cam4/cam4/color/image_rect_raw", qos_profile=qos)
        d4 = Subscriber(self, Image, "/cam4/cam4/aligned_depth_to_color/image_raw", qos_profile=qos)
        self._sync = ApproximateTimeSynchronizer(
            [rgb3, d3, rgb4, d4], queue_size=100, slop=0.5,
        )
        self._sync.registerCallback(self._sync_cb)

        self.get_logger().info("ObsManager initialized")

    def _joints_cb(self, msg: JointState):
        arr = [p for n, p in zip(msg.name, msg.position) if n in FRANKA_JOINTS]
        if arr:
            self.franka_joints = np.asarray(arr, dtype=np.float32)

    def _gripper_cb(self, msg: JointState):
        arr = []
        for n, p in zip(msg.name, msg.position):
            if n in GRIPPER_JOINTS:
                arr.append(1.0 if p >= 0.95 else 0.0)
        if arr:
            self.gripper_state = np.asarray(arr, dtype=np.float32)

    def _ft_cb(self, msg: WrenchStamped):
        w = msg.wrench
        self.ft_wrench = np.asarray(
            [w.force.x, w.force.y, w.force.z,
             w.torque.x, w.torque.y, w.torque.z], dtype=np.float32)

    def _sync_cb(self, rgb3, d3, rgb4, d4):
        try:
            snap = {
                "timestamp": time.time(),
                "rgb": [self.bridge.imgmsg_to_cv2(rgb3, "rgb8")],
                "depth": [self.bridge.imgmsg_to_cv2(d3, "16UC1")],
                "inhand_rgb": self.bridge.imgmsg_to_cv2(rgb4, "rgb8"),
                "inhand_depth": self.bridge.imgmsg_to_cv2(d4, "16UC1") / 1000.0,
                "franka_joints": None if self.franka_joints is None else self.franka_joints.copy(),
                "gripper_state": None if self.gripper_state is None else self.gripper_state.copy(),
                "ft_wrench": None if self.ft_wrench is None else self.ft_wrench.copy(),
            }
            with self._lock:
                self._latest = snap
                self.callback_count += 1
        except Exception as e:
            self.get_logger().error(f"_sync_cb error: {e!r}")

    def get_obs(self, timeout: float = 30.0) -> dict:
        deadline = time.time() + timeout
        while True:
            with self._lock:
                if self._latest is not None:
                    return self._latest
            if time.time() > deadline:
                raise TimeoutError("ObsManager: no observation within timeout")
            time.sleep(0.005)

    def is_ready(self) -> bool:
        ready = self.franka_joints is not None and self.gripper_state is not None
        if self.ft_sensor_on:
            ready = ready and self.ft_wrench is not None
        return ready and self._latest is not None


class RobotLocal:
    """Owns Robot + Gripper + ObsManager. Provides in-process get_obs / goto."""

    def __init__(self, ft_sensor_on: bool = False, do_home: bool = True):
        print("Initializing gripper...")
        gcfg = GripperConfig.from_yaml("./config/gripper_robotiq.yaml")
        self.gripper = Gripper(gripper_config=gcfg)
        self.gripper.wait_until_ready()
        self.gripper.set_target(1.0)

        print("Initializing robot...")
        self.robot = Robot(namespace="", robot_config=FrankaConfig())
        self.robot.wait_until_ready()
        if do_home:
            print("Homing...")
            self.robot.home()

        print("Starting ObsManager...")
        self.obs = ObsManager(ft_sensor_on=ft_sensor_on)
        self._spin_thread = threading.Thread(
            target=rclpy.spin, args=(self.obs,), daemon=True)
        self._spin_thread.start()

        print("Waiting for first observation...")
        while not self.obs.is_ready():
            time.sleep(0.05)
        print("RobotLocal ready.")

        self.n_interpolation = 10
        self._goto_lock = threading.Lock()

    def get_obs(self) -> dict:
        snap = self.obs.get_obs()
        ee = self.robot.end_effector_pose
        snap["ee_position"] = np.asarray(ee.position, dtype=np.float32)
        snap["ee_quat_xyzw"] = ee.orientation.as_quat().astype(np.float32)
        snap["joint_values"] = np.asarray(self.robot.joint_values, dtype=np.float32)
        snap["gripper_value"] = (
            float(self.gripper.value) if self.gripper.value is not None else None
        )
        return snap

    def goto(self, position, quat_xyzw=None, use_move_to=False, speed=0.05, gripper=None):
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
                    speed=speed)
            else:
                key_rots = R.concatenate([current_orient, target_orient])
                slerp = Slerp([0, 1], key_rots)
                steps = max(int(self.n_interpolation), 1)
                for s in range(1, steps + 1):
                    a = s / steps
                    pos_i = (1 - a) * current_pos + a * target_pos
                    orient_i = slerp(a)
                    self.robot.set_target(
                        pose=Pose(position=pos_i, orientation=orient_i))
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
            self.obs.destroy_node()
        except Exception:
            pass
        try:
            self.robot.shutdown()
        except Exception:
            pass


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--no-home", action="store_true")
    p.add_argument("--ft-sensor-on", action="store_true")
    p.add_argument("--dx", type=float, default=0.0)
    p.add_argument("--dy", type=float, default=0.0)
    p.add_argument("--dz", type=float, default=0.05)
    p.add_argument("--gripper", type=float, default=None)
    p.add_argument("--no-move", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    rclpy.init()
    bot = None
    try:
        bot = RobotLocal(ft_sensor_on=args.ft_sensor_on, do_home=not args.no_home)

        print("\n--- get_obs() ---")
        obs = bot.get_obs()
        print(f"timestamp     : {obs['timestamp']:.3f}")
        print(f"joint_values  : {obs['joint_values']}")
        print(f"ee_position   : {obs['ee_position']}")
        print(f"ee_quat_xyzw  : {obs['ee_quat_xyzw']}")
        print(f"gripper_value : {obs['gripper_value']}")
        for i, (rgb, d) in enumerate(zip(obs["rgb"], obs["depth"]), 1):
            print(f"cam{i}: rgb={rgb.shape}{rgb.dtype}, depth={d.shape}{d.dtype}")
        print(f"inhand_rgb    : {obs['inhand_rgb'].shape}{obs['inhand_rgb'].dtype}")
        print(f"inhand_depth  : {obs['inhand_depth'].shape}{obs['inhand_depth'].dtype}")

        if args.no_move:
            return

        target = obs["ee_position"] + np.array(
            [args.dx, args.dy, args.dz], dtype=np.float32)
        print(f"\n--- goto({target.tolist()}) ---")
        resp = bot.goto(
            position=target,
            quat_xyzw=obs["ee_quat_xyzw"],
            use_move_to=False,
            gripper=args.gripper,
        )
        print(f"response: {resp}")
    finally:
        if bot is not None:
            bot.shutdown()
        try:
            rclpy.shutdown()
        except Exception:
            pass


if __name__ == "__main__":
    main()
