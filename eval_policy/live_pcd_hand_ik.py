"""Live point cloud with WiLoR hand-orientation-driven fake gripper + Pink IK check.

Like live_pcd_hand.py but also solves IK in a background thread using the Pink IK
server. The sphere around the gripper turns RED when IK does not converge and
ORANGE when it does. When IK converges, the Franka arm is rendered at the solved
joint configuration.
"""

import os
import time
import threading
import sys
import numpy as np
import rclpy
import open3d as o3d
import torch
import cv2
import yaml
from scipy.spatial.transform import Rotation as R
from pynput import keyboard

from crisp_py.camera.pointcloud import PointCloudManager

TOOLBOX_PATH = '/home/mingxi/mingxi_ws/handpi/diffusion_policy/robotool'
sys.path.append('/home/mingxi/mingxi_ws/handpi/diffusion_policy')
sys.path.append(TOOLBOX_PATH)
from hand_tool.wilor_wrapper import WilorDetector  # type: ignore[import]
from hand_tool.trajectory_loader import ObservationProcessor  # type: ignore[import]
from hand_tool.config import T_hand_to_gripper  # type: ignore[import]
from robot_filter.arm_segmentor import RobotArmSegmentation  # type: ignore[import]

sys.path.insert(0, os.path.dirname(__file__))
from diff_eval_utils.diffusion_clients import PinkIKClient
from diff_eval_utils.diffusion_constants import DPEvalConfig  # type: ignore[import]
from diff_eval_utils.diffusion_transforms import convert_action_from_fingertip_to_gripper  # type: ignore[import]
from crisp_py.robot import Pose

# ── Config ────────────────────────────────────────────────────────────────────
CONFIG_PATH  = "/home/mingxi/mingxi_ws/handpi/diffusion_policy/robotool/robot_configs/camera_info.yaml"
CAM_FOR_HAND = "cam3"
CAM_IDX      = 3                                          # 1-indexed, for WilorDetector
GRIPPER_POS  = np.array([0.5, 0.0, 0.32])
DEFAULT_QUAT = np.array([0.0, np.pi / 12, 0.0, 0.0])   # fallback before first detection
_DATASET_PATH  = "/tmp/wilor_streaming"
SPHERE_RADIUS  = 0.2
SPHERE_N_PTS   = 300
# Pixel rectangle to exclude from hand detection (x1, y1, x2, y2).
# Set to None to disable exclusion.
EXCLUDE_RECT: tuple[int, int, int, int] | None = None

PINK_IK_SERVER_URL   = "http://localhost:5002"
HOME_JOINT_POSITION  = DPEvalConfig().home_joint_position
ROBOT_URDF_PATH      = TOOLBOX_PATH + "/robot_filter/panda_description/urdf/panda_arm_robotiq.urdf"

# Sphere colors
COLOR_IK_OK   = np.array([1.0, 0.5, 0.0])  # orange — IK converged
COLOR_IK_FAIL = np.array([1.0, 0.0, 0.0])  # red    — IK did not converge
COLOR_IK_INIT = np.array([0.5, 0.5, 0.5])  # grey   — not yet evaluated

# Robot arm render color (steel blue)
COLOR_ROBOT = np.array([0.4, 0.6, 0.8])


def _fibonacci_sphere(n: int) -> np.ndarray:
    """Return (n, 3) unit-sphere points via golden-ratio spiral (evenly spaced)."""
    golden = (1 + 5 ** 0.5) / 2
    i      = np.arange(n)
    theta  = np.arccos(1 - 2 * (i + 0.5) / n)
    phi    = 2 * np.pi * i / golden
    return np.stack([np.sin(theta) * np.cos(phi),
                     np.sin(theta) * np.sin(phi),
                     np.cos(theta)], axis=1)


# ── Background IK worker ───────────────────────────────────────────────────────
class IKWorker:
    """Solves IK in a background thread via the Pink IK server.

    Submit a (position, quaternion) target with `submit()`.  The worker keeps
    the last converged joint configuration to warm-start subsequent solves.

    Public attributes (updated after each solve):
        ik_success : bool | None  — True / False once at least one solve has run
        q_solution : np.ndarray | None  — last joint configuration (7 DOF)
    """

    def __init__(self, ik_client: PinkIKClient):
        self._client    = ik_client
        self._pending: tuple[np.ndarray, np.ndarray] | None = None
        self._lock      = threading.Lock()
        self.ik_success: bool | None = None   # None = not yet evaluated
        self.q_solution: np.ndarray | None = None
        threading.Thread(target=self._run, daemon=True).start()

    def submit(self, position: np.ndarray, quat: np.ndarray) -> None:
        """Submit a new target pose for IK solving (non-blocking, drops stale frames)."""
        with self._lock:
            self._pending = (position.copy(), quat.copy())

    def _run(self) -> None:
        while True:
            with self._lock:
                pending, self._pending = self._pending, None

            if pending is None:
                time.sleep(0.005)
                continue

            pos, quat = pending
            try:
                result = self._client.solve_ik(pos, quat, q_init=self.q_solution)  # type: ignore[arg-type]
                q_sol: np.ndarray = result[0]
                success: bool = result[1]
                if success:
                    self.q_solution = q_sol
                self.ik_success = success
            except Exception as e:
                print(f"[IK] server error: {e}")
                # Keep previous ik_success on transient errors


# ── Background hand tracker ────────────────────────────────────────────────────
class HandTracker:
    """Runs WiLoR in a background thread via WilorDetector.

    Public attributes (updated each inference pass):
        orientation : [4] quaternion [qx,qy,qz,qw] in world frame, or None
        overlay     : BGR uint8 image with hand mask composited, or None
    """

    def __init__(self, detector, cam_R, cam_t_world, frame_hw):
        self._detector    = detector
        self._cam_R       = cam_R
        self._cam_t_world = cam_t_world
        self._cam_T       = np.eye(4)
        self._cam_T[:3, :3] = self._cam_R
        self._cam_T[:3, 3]  = self._cam_t_world
        self._frame_hw    = frame_hw     # (height, width) — needed for renderer pre-init
        self._pending: tuple | None = None
        self._lock        = threading.Lock()
        self._renderer_ready = threading.Event()
        self.position    = GRIPPER_POS.copy()
        self.orientation: np.ndarray | None = None
        self.overlay: np.ndarray | None = None

        self.T_hand_to_gripper = T_hand_to_gripper

        threading.Thread(target=self._run, daemon=True).start()
        # Block until the background thread has created its EGL context.
        # This must happen before Open3D creates its own OpenGL context.
        self._renderer_ready.wait(timeout=15.0)

    def submit(self, frame_bgr: np.ndarray, depth_m: np.ndarray) -> None:
        """Submit a new BGR frame + metric depth map for processing."""
        if EXCLUDE_RECT is not None:
            x1, y1, x2, y2 = EXCLUDE_RECT
            frame_bgr = frame_bgr.copy()
            frame_bgr[y1:y2, x1:x2] = 0
            depth_m = depth_m.copy()
            depth_m[y1:y2, x1:x2] = 0
        with self._lock:
            self._pending = (frame_bgr, depth_m)

    def _run(self) -> None:
        # Pre-initialize pyrender OffscreenRenderer in this thread so its EGL
        # context is created before Open3D's OpenGL context (avoids EGL conflict).
        h, w = self._frame_hw
        try:
            import pyrender
            self._detector._pyrender_renderer = pyrender.OffscreenRenderer(
                viewport_width=w, viewport_height=h
            )
            self._detector._pyrender_renderer_size = (h, w)
        except Exception as e:
            print(f"[WiLoR] Renderer pre-init failed: {e}")
        finally:
            self._renderer_ready.set()

        while True:
            with self._lock:
                pending, self._pending = self._pending, None

            if pending is None:
                time.sleep(0.005)
                continue

            frame_bgr, depth_m = pending
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

            try:
                with torch.no_grad():
                    result = self._detector._detect_hands_one_frame(
                        frame_rgb, depth_m, cam_id=CAM_IDX, enable_refinement=True
                    )
            except Exception as e:
                print(f"[WiLoR] {e}")
                continue

            display     = frame_rgb.copy()
            orientation = None

            if result is not None:
                T_hand_cam = np.eye(4)
                rot_hand_cam       = result["global_orient"]
                T_hand_cam[:3, :3] = rot_hand_cam
                T_hand_cam[:3, 3]  = result["cam_t"]
                world_T_hand = self._cam_T @ T_hand_cam @ self.T_hand_to_gripper
                self.position = world_T_hand[:3, 3]
                orientation = R.from_matrix(world_T_hand[:3, :3]).as_quat()

                is_right = result["is_right"]
                if result.get("mask") is not None:
                    mask = result["mask"]
                else:
                    mask = self._detector._render_wilor_mask(
                        result["verts"], result["cam_t"],
                        result["img_size"], result["focal_length"], is_right,
                    )
                overlay_color = (0, 255, 0) if is_right > 0.5 else (0, 165, 255)
                colored_mask = np.zeros_like(display)
                colored_mask[mask > 0] = overlay_color
                display = cv2.addWeighted(display, 1.0, colored_mask, 0.4, 0)

                aligned_verts = result.get("aligned_verts")
                if aligned_verts is not None:
                    pos_3d = aligned_verts.mean(axis=0)
                    hand_str = "R" if is_right > 0.5 else "L"
                    cv2.putText(
                        display,
                        f"{hand_str}: ({pos_3d[0]:.2f}, {pos_3d[1]:.2f}, {pos_3d[2]:.2f}) m",
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2,
                    )

            self.orientation = orientation
            self.overlay     = display


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    rclpy.init()

    with open(CONFIG_PATH) as f:
        cam_params = yaml.safe_load(f)
    cam_R       = R.from_quat(cam_params[CAM_FOR_HAND]["q"]).as_matrix()
    cam_t_world = np.array(cam_params[CAM_FOR_HAND]["t"])

    os.makedirs(_DATASET_PATH, exist_ok=True)

    frame_hw = (cam_params[CAM_FOR_HAND]["height"], cam_params[CAM_FOR_HAND]["width"])

    manager       = PointCloudManager(CONFIG_PATH)
    obs_processor = ObservationProcessor()
    detector      = WilorDetector(_DATASET_PATH, main_cam_idx=CAM_IDX)

    # Robot arm segmentor — used to render arm at IK-solved joint config.
    robot_seg = RobotArmSegmentation()
    robot_seg.load_urdf(ROBOT_URDF_PATH)

    # HandTracker pre-initializes pyrender EGL in its background thread and
    # blocks here until done — Open3D window must be created after this point.
    tracker = HandTracker(detector, cam_R, cam_t_world, frame_hw)

    # Connect to Pink IK server and start background IK worker
    print(f"Connecting to Pink IK server at {PINK_IK_SERVER_URL} ...")
    ik_client = PinkIKClient(server_url=PINK_IK_SERVER_URL)
    ik_worker = IKWorker(ik_client)

    threading.Thread(target=rclpy.spin, args=(manager,), daemon=True).start()

    vis = o3d.visualization.Visualizer()  # type: ignore[attr-defined]
    vis.create_window("Live PCD + Hand Pose + IK", width=800, height=600)
    vis.get_render_option().point_size = 15.0
    vis.get_render_option().background_color = np.array([0.5, 0.5, 0.5])

    # Scene point cloud
    pcd_o3d = o3d.geometry.PointCloud()
    vis.add_geometry(pcd_o3d)

    # Sphere around gripper center (color encodes IK status)
    _unit_sphere = _fibonacci_sphere(SPHERE_N_PTS)
    sphere_o3d   = o3d.geometry.PointCloud()
    sphere_o3d.colors = o3d.utility.Vector3dVector(
        np.tile(COLOR_IK_INIT, (SPHERE_N_PTS, 1))
    )
    vis.add_geometry(sphere_o3d)

    # Robot arm point cloud (updated when IK has a solution)
    robot_o3d = o3d.geometry.PointCloud()
    vis.add_geometry(robot_o3d)

    first_frame = True

    # Gripper state: 1 = open, 0 = closed (hold space to close)
    gripper_closed = threading.Event()

    def _on_press(key):
        if key == keyboard.Key.space:
            gripper_closed.set()
        elif hasattr(key, 'char') and key.char == 'r':
            ik_worker.q_solution = HOME_JOINT_POSITION.copy()
            print("\n[reset] IK warm-start reset to home joint position")

    def _on_release(key):
        if key == keyboard.Key.space:
            gripper_closed.clear()

    kb_listener = keyboard.Listener(on_press=_on_press, on_release=_on_release)
    kb_listener.start()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device}. Press Q (cv2 window) or close Open3D to exit.")
    print("Hold SPACE to close gripper.  Press R to reset IK to home.")
    print("Sphere: GREY=no IK yet  ORANGE=IK converged  RED=IK failed")

    try:
        while True:
            time.sleep(0.01)
            pcd = manager.get_latest_pointcloud()

            # Feed latest RGB+depth frame to WiLoR background thread
            cam3_rgb, cam3_depth = manager.get_latest_rgbd('cam3')
            tracker.submit(cam3_rgb, cam3_depth / 1000.)

            # Show WiLoR hand detection overlay
            if tracker.overlay is not None:
                cv2.imshow("WiLoR Hand Detection", tracker.overlay)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            # Hand-driven position and orientation
            pos  = tracker.position
            quat = tracker.orientation if tracker.orientation is not None else DEFAULT_QUAT
            pose = np.concatenate([pos, quat])
            euler = R.from_quat(quat).as_euler('XYZ', degrees=True)

            # Convert fingertip pose → gripper frame for IK (applies FINGER_HAND_OFFSET + ROBOTIQ_ROTATION_OFFSET)
            fingertip_pose = Pose(position=pos.copy(), orientation=R.from_quat(quat))
            pos_ik, rot_ik = convert_action_from_fingertip_to_gripper(fingertip_pose)
            ik_quat = rot_ik.as_quat()

            # Submit current pose to IK worker (non-blocking)
            ik_worker.submit(pos_ik, ik_quat)

            # ── Sphere: color encodes IK status ───────────────────────────────
            ik_ok = ik_worker.ik_success
            if ik_ok is None:
                sphere_color = COLOR_IK_INIT
            elif ik_ok:
                sphere_color = COLOR_IK_OK
            else:
                sphere_color = COLOR_IK_FAIL

            ik_str = "?" if ik_ok is None else ("OK  " if ik_ok else "FAIL")
            print(f"\rpos: [{pos[0]:+.3f} {pos[1]:+.3f} {pos[2]:+.3f}]  "
                  f"euler (XYZ°): [{euler[0]:+7.2f} {euler[1]:+7.2f} {euler[2]:+7.2f}]  "
                  f"IK: {ik_str}",
                  end="", flush=True)

            # ── Scene point cloud ─────────────────────────────────────────────
            gripper_state = 0 if gripper_closed.is_set() else 1
            pcd_filtered = obs_processor.filter_pcd_by_workspace(pcd)
            obs_processor.robot_filter._init_pre_samples()
            render_pcd = obs_processor.get_render_pcd(
                pcd_filtered, pose, gripper_state, render_type='gripper'
            )
            pcd_o3d.points = o3d.utility.Vector3dVector(render_pcd[:, :3])
            pcd_o3d.colors = o3d.utility.Vector3dVector(render_pcd[:, 3:])
            vis.update_geometry(pcd_o3d)

            # ── Sphere ────────────────────────────────────────────────────────
            sphere_o3d.points = o3d.utility.Vector3dVector(
                pos + SPHERE_RADIUS * _unit_sphere
            )
            sphere_o3d.colors = o3d.utility.Vector3dVector(
                np.tile(sphere_color, (SPHERE_N_PTS, 1))
            )
            vis.update_geometry(sphere_o3d)

            # ── Robot arm ─────────────────────────────────────────────────────
            q_joints = ik_worker.q_solution
            if q_joints is not None:
                robot_seg._init_pre_samples()
                arm_pts = robot_seg.get_robot_pcd(q_joints)   # (N, 3)
                robot_o3d.points = o3d.utility.Vector3dVector(arm_pts)
                robot_o3d.colors = o3d.utility.Vector3dVector(
                    np.tile(COLOR_ROBOT, (arm_pts.shape[0], 1))
                )
            else:
                robot_o3d.points = o3d.utility.Vector3dVector(np.empty((0, 3)))
                robot_o3d.colors = o3d.utility.Vector3dVector(np.empty((0, 3)))
            vis.update_geometry(robot_o3d)

            if first_frame:
                vis.reset_view_point(True)
                first_frame = False

            if not vis.poll_events():
                break
            vis.update_renderer()

    except KeyboardInterrupt:
        print("\nStopping…")
    except Exception:
        import traceback
        traceback.print_exc()
    finally:
        kb_listener.stop()
        cv2.destroyAllWindows()
        try: vis.destroy_window()
        except: pass
        try: manager.destroy_node()
        except: pass
        try:
            if rclpy.ok(): rclpy.shutdown()
        except: pass


if __name__ == "__main__":
    main()
