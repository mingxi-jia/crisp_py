"""Live point cloud with WiLoR hand-orientation-driven fake gripper."""

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

sys.path.append('/home/mingxi/mingxi_ws/handpi/diffusion_policy')
sys.path.append('/home/mingxi/mingxi_ws/handpi/diffusion_policy/robotool')
from hand_tool.wilor_wrapper import WilorDetector
from hand_tool.trajectory_loader import ObservationProcessor
from hand_tool.config import T_hand_to_gripper

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
EXCLUDE_RECT   = None   # e.g. (0, 400, 640, 720) to blank out the bottom strip


def _fibonacci_sphere(n: int) -> np.ndarray:
    """Return (n, 3) unit-sphere points via golden-ratio spiral (evenly spaced)."""
    golden = (1 + 5 ** 0.5) / 2
    i      = np.arange(n)
    theta  = np.arccos(1 - 2 * (i + 0.5) / n)
    phi    = 2 * np.pi * i / golden
    return np.stack([np.sin(theta) * np.cos(phi),
                     np.sin(theta) * np.sin(phi),
                     np.cos(theta)], axis=1)


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
        self._cam_t_world = cam_t_world  # camera origin in world frame (from extrinsics)
        self._cam_T       = np.eye(4)
        self._cam_T[:3, :3] = self._cam_R
        self._cam_T[:3, 3]  = self._cam_t_world
        self._frame_hw    = frame_hw     # (height, width) — needed for renderer pre-init
        self._pending     = None
        self._lock        = threading.Lock()
        self._renderer_ready = threading.Event()
        self.position    = GRIPPER_POS.copy()  # fallback until first detection
        self.orientation = None
        self.overlay     = None

        self.T_hand_to_gripper = T_hand_to_gripper
 
        # self.T_hand_to_gripper = np.array([
        #     [-0.005553190748803581, 0.62385488157325, 0.7815204724188192, 0.07920393734349347],
        #     [0.7164299691783929, 0.54771640902958, -0.4321282616802441, -0.05415943453928068],
        #     [-0.6976369122513192, 0.5575049973022381, -0.4499899072729043, -0.07488318876539202],
        #     [0.0, 0.0, 0.0, 1.0],
        # ]) # Esther

        # self.T_hand_to_gripper = np.array([
        #     [-0.2770034335328179, -0.6325336090917845, -0.7233051438918097, -0.005781714379761632],
        #     [0.11355091324762713, -0.7690359216205283, 0.6290389028970261, 0.02637953173660007],
        #     [-0.9541358854542992, 0.09211397628257183, 0.28485035977115114, 0.039404214040995256],
        #     [0.0, 0.0, 0.0, 1.0],
        # ]) # thumb index 


        threading.Thread(target=self._run, daemon=True).start()
        # Block until the background thread has created its EGL context.
        # This must happen before Open3D creates its own OpenGL context.
        self._renderer_ready.wait(timeout=15.0)

    def submit(self, frame_bgr, depth_m):
        """Submit a new BGR frame + metric depth map for processing."""
        if EXCLUDE_RECT is not None:
            x1, y1, x2, y2 = EXCLUDE_RECT
            frame_bgr = frame_bgr.copy()
            frame_bgr[y1:y2, x1:x2] = 0
            depth_m = depth_m.copy()
            depth_m[y1:y2, x1:x2] = 0
        with self._lock:
            self._pending = (frame_bgr, depth_m)

    def _run(self):
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
                # Position: transform hand root from camera frame to world frame
                # self.position = self._cam_R @ result["cam_t"] + self._cam_t_world
                T_hand_cam = np.eye(4)
                rot_hand_cam     = result["global_orient"]
                T_hand_cam[:3, :3] = rot_hand_cam
                T_hand_cam[:3, 3] = result["cam_t"]
                # print(f"output from detector {result['cam_t']}, rot {rot_hand_cam}")
                self._world_T_hand = self._cam_T @ T_hand_cam
                self._world_T_hand = self._world_T_hand @ self.T_hand_to_gripper
                self.position = self._world_T_hand[:3, 3]
                orientation = R.from_matrix(self._world_T_hand[:3, :3]).as_quat()

                # Build mask overlay — prefer ICP-refined mask when available
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

                # Show ICP-refined 3-D position when available
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

    frame_hw      = (cam_params[CAM_FOR_HAND]["height"], cam_params[CAM_FOR_HAND]["width"])

    manager       = PointCloudManager(CONFIG_PATH)
    obs_processor = ObservationProcessor()
    detector      = WilorDetector(_DATASET_PATH, main_cam_idx=CAM_IDX)
    # HandTracker pre-initializes pyrender EGL in its background thread and
    # blocks here until done — Open3D window must be created after this point.
    tracker       = HandTracker(detector, cam_R, cam_t_world, frame_hw)

    threading.Thread(target=rclpy.spin, args=(manager,), daemon=True).start()

    vis = o3d.visualization.Visualizer()
    vis.create_window("Live PCD + Hand Pose", width=800, height=600)
    vis.get_render_option().point_size = 15.0
    vis.get_render_option().background_color = np.array([0.5, 0.5, 0.5])
    pcd_o3d = o3d.geometry.PointCloud()
    vis.add_geometry(pcd_o3d)

    # Sparse sphere around gripper center
    _unit_sphere  = _fibonacci_sphere(SPHERE_N_PTS)
    sphere_o3d    = o3d.geometry.PointCloud()
    sphere_o3d.colors = o3d.utility.Vector3dVector(
        np.tile([1.0, 0.5, 0.0], (SPHERE_N_PTS, 1))   # orange
    )
    vis.add_geometry(sphere_o3d)

    first_frame = True

    # Gripper state: 1 = open, 0 = closed (hold space to close)
    gripper_closed = threading.Event()

    def _on_press(key):
        if key == keyboard.Key.space:
            gripper_closed.set()

    def _on_release(key):
        if key == keyboard.Key.space:
            gripper_closed.clear()

    kb_listener = keyboard.Listener(on_press=_on_press, on_release=_on_release)
    kb_listener.start()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device}. Press Q (cv2 window) or close Open3D to exit. Hold SPACE to close gripper.")

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
            pos  = tracker.position   # updated by background thread; init = GRIPPER_POS
            quat = tracker.orientation if tracker.orientation is not None else DEFAULT_QUAT
            pose = np.concatenate([pos, quat])
            euler = R.from_quat(quat).as_euler('XYZ', degrees=True)
            print(f"\rpos: [{pos[0]:+.3f} {pos[1]:+.3f} {pos[2]:+.3f}]  "
                  f"euler (XYZ°): [{euler[0]:+7.2f} {euler[1]:+7.2f} {euler[2]:+7.2f}]",
                  end="", flush=True)

            gripper_state = 0 if gripper_closed.is_set() else 1

            pcd_filtered = obs_processor.filter_pcd_by_workspace(pcd)
            obs_processor.robot_filter._init_pre_samples()
            render_pcd = obs_processor.get_render_pcd(pcd_filtered, pose, gripper_state, render_type='gripper')

            pcd_o3d.points = o3d.utility.Vector3dVector(render_pcd[:, :3])
            pcd_o3d.colors = o3d.utility.Vector3dVector(render_pcd[:, 3:])
            vis.update_geometry(pcd_o3d)

            sphere_o3d.points = o3d.utility.Vector3dVector(
                pos + SPHERE_RADIUS * _unit_sphere
            )
            vis.update_geometry(sphere_o3d)

            if first_frame:
                vis.reset_view_point(True)
                first_frame = False

            if not vis.poll_events():
                break
            vis.update_renderer()

    except KeyboardInterrupt:
        print("\nStopping…")
    except Exception as e:
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
