#!/usr/bin/env python3
"""
Interactive viewer for raw dataset RGB images from multiple cameras and episodes.
Navigate through episodes and frames using arrow keys.

Supports two data sources:
  - Directory of episode_* folders (default)
  - HDF5 file (use --hdf5 PATH)
"""

import argparse
from pathlib import Path
import cv2
import numpy as np
import shutil


def auto_clean_empty_episodes(episodes_path, use_segmented=False, check_cam=3):
    """
    Automatically delete episode folders that have no images in the specified camera folder.

    Args:
        episodes_path: Path to episodes directory
        use_segmented: If True, check segmented_rgb instead of rgb
        check_cam: Camera index to check (default: 3)

    Returns:
        int: Number of episodes deleted
    """
    episodes_path = Path(episodes_path)
    trash_folder = episodes_path / ".trash"
    trash_folder.mkdir(exist_ok=True)

    episodes = sorted([d for d in episodes_path.iterdir()
                      if d.is_dir() and d.name.startswith("episode_")])

    if not episodes:
        return 0

    deleted_count = 0
    img_subdir = "segmented_rgb" if use_segmented else "rgb"

    for episode_dir in episodes:
        cam_dir = episode_dir / f"cam{check_cam}" / img_subdir

        # Check if cam folder exists and has images
        should_delete = False
        reason = ""
        if not cam_dir.exists():
            should_delete = True
            reason = f"cam{check_cam}/{img_subdir} folder does not exist"
        else:
            images = list(cam_dir.glob("*.png"))
            print(len(images))
            if len(images) == 0:
                should_delete = True
                reason = f"cam{check_cam}/{img_subdir} folder is empty"

        if should_delete:
            episode_name = episode_dir.name
            trash_dest = trash_folder / episode_name

            try:
                # If destination exists in trash, add timestamp
                if trash_dest.exists():
                    import time
                    trash_dest = trash_folder / f"{episode_name}_{int(time.time())}"
                shutil.move(str(episode_dir), str(trash_dest))
                print(f"Auto-cleaned: '{episode_name}' ({reason}) -> {trash_dest}")
                deleted_count += 1
            except Exception as e:
                print(f"Error auto-cleaning {episode_name}: {e}")

    if deleted_count > 0:
        print(f"\nAuto-cleaned {deleted_count} empty episodes (moved to {trash_folder})\n")

    return deleted_count


def load_all_episodes_images(episodes_path, num_cams=3, use_segmented=False, load_eef_pose=False, load_intervention=False, min_traj_length=0):
    """
    Load all RGB images from all episodes.

    Args:
        episodes_path: Path to episodes directory
        num_cams: Number of cameras
        use_segmented: If True, load segmented_rgb instead of rgb
        load_eef_pose: If True, also load eef_pose trajectory data
        load_intervention: If True, also load intervention flags
        min_traj_length: Minimum trajectory length to include episode (default: 0, no filtering)

    Returns:
        episodes_data: list of dicts with {
            'name': episode name,
            'frames': {frame_idx: {cam_idx: image_path}},
            'grasp_actions': numpy array of gripper values,
            'eef_pose': numpy array of end-effector poses (if load_eef_pose=True),
            'intervention_flags': numpy array of intervention flags (if load_intervention=True),
            'false_marked': boolean indicating if false_mark.txt exists
        }
    """
    episodes_path = Path(episodes_path)

    # Find all episode directories
    episodes = sorted([d for d in episodes_path.iterdir()
                      if d.is_dir() and d.name.startswith("episode_")])

    if not episodes:
        print(f"No episodes found in {episodes_path}")
        return []

    print(f"Found {len(episodes)} episodes")
    if use_segmented:
        print(f"Using segmented RGB images")
    else:
        print(f"Using raw RGB images")

    episodes_data = []

    for episode_dir in episodes:
        print(f"Loading {episode_dir.name}...", end=" ")

        # Collect all images from all cameras
        cam_images = {}

        for cam_idx in range(1, num_cams + 1):
            # Choose rgb or segmented_rgb based on flag
            img_subdir = "segmented_rgb" if use_segmented else "rgb"
            rgb_dir = episode_dir / f"cam{cam_idx}" / img_subdir

            if not rgb_dir.exists():
                print(f"\n  Warning: {rgb_dir} does not exist")
                continue

            images = sorted(rgb_dir.glob("*.png"))
            cam_images[cam_idx] = images

        # Verify all cameras have the same number of images
        num_frames_list = [len(imgs) for imgs in cam_images.values()]
        if not num_frames_list:
            print("No images found")
            continue

        if len(set(num_frames_list)) > 1:
            print(f"\n  Warning: Cameras have different number of frames: {num_frames_list}")
            min_frames = min(num_frames_list)
            print(f"  Using minimum: {min_frames} frames")
        else:
            min_frames = num_frames_list[0]

        # Organize by frame index
        frame_data = {}
        for frame_idx in range(min_frames):
            frame_data[frame_idx] = {
                cam_idx: cam_images[cam_idx][frame_idx]
                for cam_idx in cam_images.keys()
            }

        # Load gripper actions
        # For segmented (processed) data, grasp.npy is in episode root
        # For raw data, it's in state/ subdirectory
        if use_segmented:
            grasp_file = episode_dir / "grasp.npy"
        else:
            grasp_file = episode_dir / "state" / "grasp.npy"

        grasp_actions = None
        if grasp_file.exists():
            try:
                grasp_actions = np.load(grasp_file)
                if len(grasp_actions) != min_frames:
                    print(f"\n  Warning: Grasp actions length ({len(grasp_actions)}) != frames ({min_frames})")
                    # Pad or trim to match frames
                    if len(grasp_actions) < min_frames:
                        grasp_actions = np.pad(grasp_actions, (0, min_frames - len(grasp_actions)), mode='edge')
                    else:
                        grasp_actions = grasp_actions[:min_frames]
            except Exception as e:
                print(f"\n  Warning: Could not load grasp actions: {e}")
        else:
            print(f"\n  Warning: {grasp_file} does not exist")

        # Load eef_pose trajectory if requested
        eef_pose = None
        if load_eef_pose:
            if use_segmented:
                eef_pose_file = episode_dir / "eef_pose.npy"
            else:
                eef_pose_file = episode_dir / "state" / "pose_wrt_world.npy"

            if eef_pose_file.exists():
                try:
                    eef_pose = np.load(eef_pose_file)
                    if len(eef_pose) != min_frames:
                        print(f"\n  Warning: EEF pose length ({len(eef_pose)}) != frames ({min_frames})")
                        # Pad or trim to match frames
                        if len(eef_pose) < min_frames:
                            eef_pose = np.pad(eef_pose, ((0, min_frames - len(eef_pose)), (0, 0)), mode='edge')
                        else:
                            eef_pose = eef_pose[:min_frames]
                except Exception as e:
                    print(f"\n  Warning: Could not load eef_pose: {e}")
            else:
                print(f"\n  Warning: {eef_pose_file} does not exist")

        # Load intervention flags if requested
        intervention_flags = None
        if load_intervention:
            if use_segmented:
                intervention_file = episode_dir / "intervention.npy"
            else:
                intervention_file = episode_dir / "state" / "intervention.npy"

            if intervention_file.exists():
                try:
                    intervention_flags = np.load(intervention_file)
                    if len(intervention_flags) != min_frames:
                        print(f"\n  Warning: Intervention flags length ({len(intervention_flags)}) != frames ({min_frames})")
                        # Pad or trim to match frames
                        if len(intervention_flags) < min_frames:
                            intervention_flags = np.pad(intervention_flags, (0, min_frames - len(intervention_flags)), mode='edge')
                        else:
                            intervention_flags = intervention_flags[:min_frames]
                except Exception as e:
                    print(f"\n  Warning: Could not load intervention flags: {e}")
            else:
                print(f"\n  Warning: {intervention_file} does not exist")

        # Load gripper qpos if available
        gripper_qpos = None
        if not use_segmented:
            qpos_file = episode_dir / "state" / "gripper_qpos.npy"
            if qpos_file.exists():
                try:
                    gripper_qpos = np.load(qpos_file)
                    if len(gripper_qpos) != min_frames:
                        if len(gripper_qpos) < min_frames:
                            gripper_qpos = np.pad(gripper_qpos, (0, min_frames - len(gripper_qpos)), mode='edge')
                        else:
                            gripper_qpos = gripper_qpos[:min_frames]
                except Exception as e:
                    print(f"\n  Warning: Could not load gripper_qpos: {e}")

        # Check for false_mark.txt
        if use_segmented:
            false_mark_file = episode_dir / "false_mark.txt"
        else:
            false_mark_file = episode_dir / "state" / "false_mark.txt"
        false_marked = false_mark_file.exists()

        episodes_data.append({
            'name': episode_dir.name,
            'path': episode_dir,
            'frames': frame_data,
            'num_frames': min_frames,
            'grasp_actions': grasp_actions,
            'eef_pose': eef_pose,
            'intervention_flags': intervention_flags,
            'gripper_qpos': gripper_qpos,
            'false_marked': false_marked
        })

        print(f"{min_frames} frames")

    # Filter episodes by minimum trajectory length if specified
    if min_traj_length > 0:
        original_count = len(episodes_data)
        episodes_data = [ep for ep in episodes_data if ep['num_frames'] > min_traj_length]
        filtered_count = original_count - len(episodes_data)
        if filtered_count > 0:
            print(f"\nFiltered out {filtered_count} episodes with trajectory length <= {min_traj_length}")
            print(f"Remaining episodes: {len(episodes_data)}")

    return episodes_data


def load_hdf5_episodes(hdf5_path):
    """
    Load episode metadata from a robomimic-style HDF5 file.
    Images are loaded lazily per frame (the h5py file stays open).

    Expected structure:
        data/
          demo_0/
            obs/
              cam1_image            (T, H, W, 3)
              cam2_image            (T, H, W, 3)
              cam3_image            (T, H, W, 3)
              robot0_eye_in_hand_image  (T, H, W, 3)
              robot0_gripper_state  (T,) or (T, 1)
            actions                 (T, D)

    Returns:
        (episodes_data, h5file) — caller must close h5file when done.
    """
    import h5py

    h5file = h5py.File(hdf5_path, 'r')
    data_group = h5file['data']

    # Camera key → display cam index mapping
    cam_key_map = {
        'cam1_image': 1,
        'cam2_image': 2,
        'cam3_image': 3,
        'robot0_eye_in_hand_image': 4,
    }

    demo_names = sorted(data_group.keys())
    print(f"Found {len(demo_names)} demos in {hdf5_path}")

    episodes_data = []
    for demo_name in demo_names:
        demo = data_group[demo_name]
        obs = demo.get('obs', {})

        # Discover which camera keys are present
        cam_keys = {}
        for key, cam_idx in cam_key_map.items():
            if key in obs:
                cam_keys[cam_idx] = obs[key]  # h5py Dataset, shape (T, H, W, 3)

        if not cam_keys:
            print(f"  {demo_name}: no camera obs found, skipping")
            continue

        T = next(iter(cam_keys.values())).shape[0]

        # Gripper state
        grasp_actions = None
        if 'robot0_gripper_state' in obs:
            gs = obs['robot0_gripper_state'][:]  # load fully — small array
            if gs.ndim == 2:
                gs = gs[:, 0]
            grasp_actions = gs.astype(np.float32)

        episodes_data.append({
            'name': demo_name,
            'path': None,
            'num_frames': T,
            'frames': None,           # not used for HDF5
            'data_source': 'hdf5',
            'cam_keys': cam_keys,     # {cam_idx: h5py.Dataset}
            'grasp_actions': grasp_actions,
            'eef_pose': None,
            'intervention_flags': None,
            'false_marked': False,
        })
        print(f"  {demo_name}: {T} frames, cameras: {sorted(cam_keys.keys())}")

    return episodes_data, h5file


def create_gripper_bar(width, gripper_value, bar_height=40):
    """
    Create a gripper action visualization bar.

    Args:
        width: width of the bar
        gripper_value: value between 0 (open) and 1 (closed)
        bar_height: height of the bar

    Returns:
        numpy array: bar visualization
    """
    bar = np.zeros((bar_height, width, 3), dtype=np.uint8)

    # Background (dark gray)
    bar[:, :] = (40, 40, 40)

    # Border
    cv2.rectangle(bar, (0, 0), (width - 1, bar_height - 1), (100, 100, 100), 2)

    # Filled portion (green for closed gripper)
    if gripper_value > 0:
        fill_width = int(width * gripper_value)
        bar[:, :fill_width] = (0, int(255 * gripper_value), 0)

    # Add text label
    label = f"Gripper: {gripper_value:.2f}"
    if gripper_value >= 0.5:
        label += " (CLOSED)"
        text_color = (255, 255, 255)
    else:
        label += " (OPEN)"
        text_color = (200, 200, 200)

    cv2.putText(bar, label, (10, bar_height - 12),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 2)

    return bar


def create_gripper_qpos_bar(width, qpos_value, bar_height=40):
    """
    Create a gripper qpos progress bar visualization (raw float 0.0–1.0).

    Args:
        width: width of the bar
        qpos_value: raw gripper position (0.0 = open, ~1.0 = closed)
        bar_height: height of the bar

    Returns:
        numpy array: bar visualization
    """
    bar = np.zeros((bar_height, width, 3), dtype=np.uint8)
    bar[:, :] = (30, 30, 30)

    # Border
    cv2.rectangle(bar, (0, 0), (width - 1, bar_height - 1), (80, 80, 80), 1)

    # Progress fill — cyan color, proportional to qpos
    qpos_clamped = float(np.clip(qpos_value, 0.0, 1.0))
    fill_width = int((width - 2) * qpos_clamped)
    if fill_width > 0:
        r = int(255 * qpos_clamped)
        g = int(180 * (1.0 - qpos_clamped))
        b = int(255 * (1.0 - qpos_clamped * 0.5))
        bar[1:bar_height - 1, 1:1 + fill_width] = (b, g, r)  # BGR

    # Text label
    label = f"Gripper qpos: {qpos_value:.3f}"
    cv2.putText(bar, label, (10, bar_height - 10),
               cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

    return bar


def create_intervention_bar(width, intervention_value, bar_height=40):
    """
    Create an intervention indicator visualization bar.

    Args:
        width: width of the bar
        intervention_value: 0 (autonomous) or 1 (intervention)
        bar_height: height of the bar

    Returns:
        numpy array: bar visualization
    """
    bar = np.zeros((bar_height, width, 3), dtype=np.uint8)

    # Background (dark gray)
    bar[:, :] = (40, 40, 40)

    # Border
    cv2.rectangle(bar, (0, 0), (width - 1, bar_height - 1), (100, 100, 100), 2)

    # Fill with color based on intervention state
    if intervention_value == 1:
        # Intervention: orange/red color
        bar[:, :] = (0, 140, 255)  # BGR: orange
        label = "Mode: INTERVENTION"
        text_color = (255, 255, 255)
    else:
        # Autonomous: blue color
        bar[:, :] = (200, 100, 0)  # BGR: blue
        label = "Mode: AUTONOMOUS"
        text_color = (255, 255, 255)

    # Redraw border on top
    cv2.rectangle(bar, (0, 0), (width - 1, bar_height - 1), (100, 100, 100), 2)

    # Add text label
    cv2.putText(bar, label, (10, bar_height - 12),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 2)

    return bar


def create_false_mark_bar(width, is_false_marked, bar_height=40):
    """
    Create a false mark indicator visualization bar.

    Args:
        width: width of the bar
        is_false_marked: True if episode is marked as false
        bar_height: height of the bar

    Returns:
        numpy array: bar visualization
    """
    bar = np.zeros((bar_height, width, 3), dtype=np.uint8)

    if is_false_marked:
        # Red background for false marked episodes
        bar[:, :] = (0, 0, 200)  # BGR: red
        label = "FALSE MARKED EPISODE"
        text_color = (255, 255, 255)
    else:
        # Dark gray background (no mark)
        bar[:, :] = (40, 40, 40)
        label = "Episode OK"
        text_color = (150, 150, 150)

    # Border
    cv2.rectangle(bar, (0, 0), (width - 1, bar_height - 1), (100, 100, 100), 2)

    # Add text label
    cv2.putText(bar, label, (10, bar_height - 12),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 2)

    return bar


def create_episode_frame_visualization(episode_data, frame_idx, max_height=600, show_intervention=False, show_false_mark=True):
    """
    Create visualization for a single frame from one episode.
    Layout: Cam 1-3 horizontally, Cam 4 (in-hand) on the right side.

    Args:
        episode_data: episode data dict
        frame_idx: which frame to visualize
        max_height: maximum height for each camera image
        show_intervention: if True, show intervention indicator bar
        show_false_mark: if True, show false mark indicator bar

    Returns:
        numpy array: concatenated image with all cameras and gripper/intervention bars
    """
    if frame_idx >= episode_data['num_frames']:
        return None

    static_cameras = []  # Cameras 1-3
    inhand_camera = None  # Camera 4

    if episode_data.get('data_source') == 'hdf5':
        for cam_idx in sorted(episode_data['cam_keys'].keys()):
            raw = np.array(episode_data['cam_keys'][cam_idx][frame_idx])  # (H, W, 3) RGB uint8
            img = cv2.cvtColor(raw, cv2.COLOR_RGB2BGR)

            if img.shape[0] > max_height:
                scale = max_height / img.shape[0]
                img = cv2.resize(img, (int(img.shape[1] * scale), max_height))

            label = f"Cam {cam_idx}" + (" (In-hand)" if cam_idx == 4 else "")
            cv2.putText(img, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            if cam_idx == 4:
                inhand_camera = img
            else:
                static_cameras.append(img)
    else:
        frame_data = episode_data['frames'][frame_idx]
        for cam_idx in sorted(frame_data.keys()):
            img = cv2.imread(str(frame_data[cam_idx]))

            if img is None:
                print(f"Warning: Could not load {frame_data[cam_idx]}")
                img = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(img, f"Cam {cam_idx} - Not Found", (50, 240),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            # Resize to max_height while maintaining aspect ratio
            if img.shape[0] > max_height:
                scale = max_height / img.shape[0]
                new_width = int(img.shape[1] * scale)
                img = cv2.resize(img, (new_width, max_height))

            label = f"Cam {cam_idx}" + (" (In-hand)" if cam_idx == 4 else "")
            cv2.putText(img, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            if cam_idx == 4:
                inhand_camera = img
            else:
                static_cameras.append(img)

    if not static_cameras and inhand_camera is None:
        return None

    # Concatenate cameras 1-3 horizontally
    if static_cameras:
        left_panel = np.concatenate(static_cameras, axis=1)
    else:
        left_panel = np.zeros((max_height, 640, 3), dtype=np.uint8)

    # Handle cam4 on the right
    if inhand_camera is not None:
        # Make cam4 fill the height of the left panel
        if inhand_camera.shape[0] < left_panel.shape[0]:
            # Add padding below cam4 to match height
            padding_height = left_panel.shape[0] - inhand_camera.shape[0]
            padding = np.zeros((padding_height, inhand_camera.shape[1], 3), dtype=np.uint8)
            inhand_camera = np.vstack([inhand_camera, padding])
        elif inhand_camera.shape[0] > left_panel.shape[0]:
            # Resize to match height
            scale = left_panel.shape[0] / inhand_camera.shape[0]
            new_width = int(inhand_camera.shape[1] * scale)
            inhand_camera = cv2.resize(inhand_camera, (new_width, left_panel.shape[0]))

        # Concatenate left panel and cam4 horizontally
        concat_img = np.concatenate([left_panel, inhand_camera], axis=1)
    else:
        concat_img = left_panel

    # Add header with episode name
    header_height = 80
    header = np.zeros((header_height, concat_img.shape[1], 3), dtype=np.uint8)

    # Episode name
    episode_name = episode_data['name']
    cv2.putText(header, episode_name, (10, 50),
               cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 0), 2)

    # Add gripper action bar below images
    gripper_value = 0.0
    if episode_data['grasp_actions'] is not None and frame_idx < len(episode_data['grasp_actions']):
        gripper_value = float(episode_data['grasp_actions'][frame_idx])

    gripper_bar = create_gripper_bar(concat_img.shape[1], gripper_value, bar_height=50)

    # Add intervention bar if requested
    bars = [gripper_bar]
    if show_intervention:
        intervention_value = 0
        if episode_data.get('intervention_flags') is not None and frame_idx < len(episode_data['intervention_flags']):
            intervention_value = int(episode_data['intervention_flags'][frame_idx])

        intervention_bar = create_intervention_bar(concat_img.shape[1], intervention_value, bar_height=50)
        bars.append(intervention_bar)

        # Gripper qpos progress bar
        qpos_value = 0.0
        if episode_data.get('gripper_qpos') is not None and frame_idx < len(episode_data['gripper_qpos']):
            qpos_value = float(episode_data['gripper_qpos'][frame_idx])
        qpos_bar = create_gripper_qpos_bar(concat_img.shape[1], qpos_value, bar_height=50)
        bars.append(qpos_bar)

    # Add false mark bar
    if show_false_mark:
        is_false_marked = episode_data.get('false_marked', False)
        false_mark_bar = create_false_mark_bar(concat_img.shape[1], is_false_marked, bar_height=50)
        bars.append(false_mark_bar)

    # Combine header, image, and bars
    final_img = np.vstack([header, concat_img] + bars)

    return final_img


def visualize_all_episodes(episodes_path, num_cams=4, use_segmented=False, teleop_mode=False, intervention_mode=False, min_traj_length=0, auto_clean=True):
    """
    Interactive visualization of episodes' RGB images.

    Args:
        episodes_path: Path to episodes directory
        num_cams: Number of cameras (default: 4)
        use_segmented: If True, visualize segmented_rgb instead of rgb
        teleop_mode: If True, load EEF pose data (no visualization)
        intervention_mode: If True, show intervention indicators
        min_traj_length: Minimum trajectory length to include episode (default: 0, no filtering)
        auto_clean: If True, automatically delete episodes with empty cam3 folders

    Controls:
        - Up/Down arrow: Navigate between episodes
        - Left/Right arrow: Navigate frames within episode
        - 'q' or ESC: Quit
        - Space: Auto-play frames
        - 'r': Reset to first frame
        - 'n': Next episode
        - 'p': Previous episode
        - 'd': Delete current episode (move to trash)
    """
    # Auto-clean empty episodes if requested
    if auto_clean:
        print("Running auto-clean for episodes with empty cam3 folders...")
        auto_clean_empty_episodes(episodes_path, use_segmented, check_cam=3)

    # Intervention mode includes teleop features plus intervention tracking
    load_eef = teleop_mode or intervention_mode
    load_intv = intervention_mode

    episodes_data = load_all_episodes_images(episodes_path, num_cams, use_segmented,
                                             load_eef_pose=load_eef, load_intervention=load_intv,
                                             min_traj_length=min_traj_length)

    if not episodes_data:
        print("No episodes found or loaded!")
        return

    print(f"\nTotal episodes: {len(episodes_data)}")

    # Create trash folder for deleted episodes
    trash_folder = Path(episodes_path) / ".trash"
    trash_folder.mkdir(exist_ok=True)
    print(f"Trash folder: {trash_folder}")

    window_name = "Episode Viewer"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    # Set window to 3x bigger size
    cv2.resizeWindow(window_name, 3840, 2160)

    current_episode_idx = 0
    current_frame = 0
    auto_play = False

    while True:
        episode = episodes_data[current_episode_idx]
        num_frames = episode['num_frames']

        # Ensure frame is within bounds
        current_frame = max(0, min(current_frame, num_frames - 1))

        # Create visualization
        concat_img = create_episode_frame_visualization(episode, current_frame, show_intervention=intervention_mode)

        if concat_img is None:
            print("Failed to create visualization")
            break

        # Add info overlay at the bottom
        info_height = 60
        info_bar = np.zeros((info_height, concat_img.shape[1], 3), dtype=np.uint8)

        # Episode info
        episode_info = f"Episode: {current_episode_idx + 1}/{len(episodes_data)}"
        cv2.putText(info_bar, episode_info, (10, 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

        # Frame info
        frame_info = f"Frame: {current_frame + 1}/{num_frames}"
        cv2.putText(info_bar, frame_info, (10, 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)

        # Controls
        controls_text = "Up/Down: Episodes | Left/Right: Frames | Space: Auto-play | D: Delete | Q: Quit"
        text_size = cv2.getTextSize(controls_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)[0]
        cv2.putText(info_bar, controls_text,
                   (concat_img.shape[1] - text_size[0] - 10, 35),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150, 150, 150), 1)

        final_img = np.vstack([concat_img, info_bar])

        cv2.imshow(window_name, final_img)

        # Handle keyboard input
        wait_time = 30 if auto_play else 0
        key = cv2.waitKey(wait_time) & 0xFF

        if key == ord('q') or key == 27:  # 'q' or ESC
            break
        elif key == 81 or key == 2:  # Left arrow
            current_frame = (current_frame - 1) % num_frames
            auto_play = False
        elif key == 83 or key == 3:  # Right arrow
            current_frame = min(num_frames - 1, current_frame + 1)
            auto_play = False
        elif key == 82 or key == 0:  # Up arrow
            # Previous episode
            current_episode_idx = max(0, current_episode_idx - 1)
            current_frame = 0  # Reset to first frame of new episode
            auto_play = False
            print(f"Switched to episode {current_episode_idx + 1}: {episodes_data[current_episode_idx]['name']}")

        elif key == 84 or key == 1:  # Down arrow
            # Next episode
            current_episode_idx = min(len(episodes_data) - 1, current_episode_idx + 1)
            current_frame = 0  # Reset to first frame of new episode
            auto_play = False
            print(f"Switched to episode {current_episode_idx + 1}: {episodes_data[current_episode_idx]['name']}")
        elif key == ord(' '):  # Space
            auto_play = not auto_play
            print(f"Auto-play: {'ON' if auto_play else 'OFF'}")
        elif key == ord('r'):  # Reset
            current_frame = 0
            auto_play = False
            print("Reset to first frame")
        elif key == ord('n'):  # Next episode
            current_episode_idx = min(len(episodes_data) - 1, current_episode_idx + 1)
            current_frame = 0
            auto_play = False
            print(f"Switched to episode {current_episode_idx + 1}: {episodes_data[current_episode_idx]['name']}")

        elif key == ord('p'):  # Previous episode
            current_episode_idx = max(0, current_episode_idx - 1)
            current_frame = 0
            auto_play = False
            print(f"Switched to episode {current_episode_idx + 1}: {episodes_data[current_episode_idx]['name']}")

        elif key == ord('d'):  # Delete current episode
            auto_play = False
            if len(episodes_data) > 0:
                episode_to_delete = episodes_data[current_episode_idx]
                episode_path = episode_to_delete['path']
                episode_name = episode_to_delete['name']

                # Move to trash folder
                trash_dest = trash_folder / episode_name
                try:
                    # If destination exists in trash, add timestamp
                    if trash_dest.exists():
                        import time
                        trash_dest = trash_folder / f"{episode_name}_{int(time.time())}"
                    shutil.move(str(episode_path), str(trash_dest))
                    print(f"Deleted episode '{episode_name}' -> moved to {trash_dest}")

                    # Remove from episodes_data list
                    episodes_data.pop(current_episode_idx)

                    # Adjust current index if needed
                    if len(episodes_data) == 0:
                        print("No more episodes to display!")
                        break
                    if current_episode_idx >= len(episodes_data):
                        current_episode_idx = len(episodes_data) - 1
                    current_frame = 0
                    print(f"Now viewing episode {current_episode_idx + 1}/{len(episodes_data)}: {episodes_data[current_episode_idx]['name']}")
                except Exception as e:
                    print(f"Error deleting episode: {e}")

        # Auto-play: advance frame
        if auto_play:
            current_frame += 1
            if current_frame >= num_frames:
                # Move to next episode when frames end
                current_episode_idx = min(len(episodes_data) - 1, current_episode_idx + 1)
                current_frame = 0
                if current_episode_idx == len(episodes_data) - 1:
                    # Loop back to first episode
                    current_episode_idx = 0
                print(f"Auto-play: Switched to episode {current_episode_idx + 1}")

    # Cleanup
    cv2.destroyAllWindows()


def visualize_hdf5(hdf5_path):
    """Interactive viewer for an HDF5 dataset file."""
    episodes_data, h5file = load_hdf5_episodes(hdf5_path)

    if not episodes_data:
        print("No demos found or loaded!")
        h5file.close()
        return

    print(f"\nTotal demos: {len(episodes_data)}")

    window_name = "HDF5 Episode Viewer"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 3840, 2160)

    current_episode_idx = 0
    current_frame = 0
    auto_play = False

    try:
        while True:
            episode = episodes_data[current_episode_idx]
            num_frames = episode['num_frames']
            current_frame = max(0, min(current_frame, num_frames - 1))

            concat_img = create_episode_frame_visualization(
                episode, current_frame, show_intervention=False, show_false_mark=False)

            if concat_img is None:
                print("Failed to create visualization")
                break

            info_height = 60
            info_bar = np.zeros((info_height, concat_img.shape[1], 3), dtype=np.uint8)

            cv2.putText(info_bar, f"Demo: {current_episode_idx + 1}/{len(episodes_data)}  [{episode['name']}]",
                       (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
            cv2.putText(info_bar, f"Frame: {current_frame + 1}/{num_frames}",
                       (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
            controls = "Up/Down: Episodes | Left/Right: Frames | Space: Auto-play | Q: Quit"
            text_size = cv2.getTextSize(controls, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)[0]
            cv2.putText(info_bar, controls,
                       (concat_img.shape[1] - text_size[0] - 10, 35),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150, 150, 150), 1)

            cv2.imshow(window_name, np.vstack([concat_img, info_bar]))

            wait_time = 30 if auto_play else 0
            key = cv2.waitKey(wait_time) & 0xFF

            if key == ord('q') or key == 27:
                break
            elif key == 81 or key == 2:  # Left
                current_frame = max(0, current_frame - 1)
                auto_play = False
            elif key == 83 or key == 3:  # Right
                current_frame = min(num_frames - 1, current_frame + 1)
                auto_play = False
            elif key == 82 or key == 0:  # Up
                current_episode_idx = max(0, current_episode_idx - 1)
                current_frame = 0
                auto_play = False
            elif key == 84 or key == 1:  # Down
                current_episode_idx = min(len(episodes_data) - 1, current_episode_idx + 1)
                current_frame = 0
                auto_play = False
            elif key == ord(' '):
                auto_play = not auto_play
            elif key == ord('r'):
                current_frame = 0
                auto_play = False
            elif key == ord('n'):
                current_episode_idx = min(len(episodes_data) - 1, current_episode_idx + 1)
                current_frame = 0
                auto_play = False
            elif key == ord('p'):
                current_episode_idx = max(0, current_episode_idx - 1)
                current_frame = 0
                auto_play = False

            if auto_play:
                current_frame += 1
                if current_frame >= num_frames:
                    current_episode_idx = (current_episode_idx + 1) % len(episodes_data)
                    current_frame = 0
    finally:
        cv2.destroyAllWindows()
        h5file.close()


def main():
    parser = argparse.ArgumentParser(
        description="Visualize episodes' RGB images from multiple cameras",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Auto-clean empty episodes and visualize with intervention mode
  python record/inspect_raw_data.py /home/mingxi/mingxi_ws/crisp/crisp_py/raw_datasets/episodes --auto-clean --intv

  # Filter short episodes and show intervention indicators
  python record/inspect_raw_data.py /home/mingxi/mingxi_ws/crisp/crisp_py/raw_datasets/episodes --min-traj-length 150 --intv

  # Visualize raw RGB images (4 cameras: 3 static + 1 in-hand)
  python record/inspect_raw_data.py /home/mingxi/mingxi_ws/crisp/crisp_py/raw_datasets/episodes

  # Visualize segmented RGB images
  python inspect_raw_data.py /home/mingxi/mingxi_ws/handpi/data/lift_block_realworld/output --segment

  # Visualize with intervention indicators
  python record/inspect_raw_data.py /home/mingxi/mingxi_ws/crisp/crisp_py/raw_datasets/episodes --intv

  # Visualize only episodes with trajectory length > 150 frames
  python record/inspect_raw_data.py /home/mingxi/mingxi_ws/crisp/crisp_py/raw_datasets/episodes --min-traj-length 150

  # Visualize with 3 cameras only (no in-hand cam)
  python record/inspect_raw_data.py /path/to/episodes --num-cams 3

Layout:
  - Cameras 1-3 displayed horizontally on the left
  - Camera 4 (in-hand) displayed on the right side
  - Window initialized at 3840x2160 (3x standard size)

Controls:
  Up/Down arrows: Navigate between episodes
  Left/Right arrows: Navigate frames within current episode
  Space: Toggle auto-play (plays frames and moves to next episode)
  N/P: Next/Previous episode
  R: Reset to first frame
  D: Delete current episode (moves to .trash folder for undo)
  Q/ESC: Quit
        """
    )

    parser.add_argument(
        "path",
        type=str,
        nargs='?',
        default=None,
        help="Path to episodes root directory (not needed when --hdf5 is used)"
    )

    parser.add_argument(
        "--hdf5",
        type=str,
        default=None,
        metavar="PATH",
        help="Load from an HDF5 file instead of an episode directory"
    )

    parser.add_argument(
        "--num-cams",
        type=int,
        default=4,
        help="Number of cameras (default: 4, including in-hand cam)"
    )

    parser.add_argument(
        "--segment",
        action="store_true",
        help="Visualize segmented_rgb instead of raw rgb images"
    )

    parser.add_argument(
        "--teleop",
        action="store_true",
        help="Enable teleop mode: load EEF pose data"
    )

    parser.add_argument(
        "--intv",
        action="store_true",
        help="Enable intervention mode: show RGB images and intervention indicators"
    )

    parser.add_argument(
        "--min-traj-length",
        type=int,
        default=0,
        help="Minimum trajectory length to include episode (default: 0, no filtering). Use 150 to filter short episodes."
    )

    parser.add_argument(
        "--auto-clean",
        action="store_true",
        help="Automatically delete episodes with empty cam3 folders (moves to .trash)"
    )

    args = parser.parse_args()

    if args.hdf5:
        hdf5_path = Path(args.hdf5)
        if not hdf5_path.exists():
            print(f"Error: HDF5 file does not exist: {hdf5_path}")
            return
        visualize_hdf5(hdf5_path)
        return

    if not args.path:
        parser.error("A path argument is required unless --hdf5 is specified")

    path = Path(args.path)
    if not path.exists():
        print(f"Error: Path does not exist: {path}")
        return

    visualize_all_episodes(path, args.num_cams, args.segment, args.teleop, args.intv, args.min_traj_length, args.auto_clean)


if __name__ == "__main__":
    main()
