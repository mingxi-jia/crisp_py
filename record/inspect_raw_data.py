#!/usr/bin/env python3
"""
Interactive viewer for raw dataset RGB images from multiple cameras and episodes.
Navigate through episodes and frames using arrow keys.
"""

import argparse
from pathlib import Path
import cv2
import numpy as np
import open3d as o3d


def load_all_episodes_images(episodes_path, num_cams=3, use_segmented=False, load_eef_pose=False, load_intervention=False):
    """
    Load all RGB images from all episodes.

    Args:
        episodes_path: Path to episodes directory
        num_cams: Number of cameras
        use_segmented: If True, load segmented_rgb instead of rgb
        load_eef_pose: If True, also load eef_pose trajectory data
        load_intervention: If True, also load intervention flags

    Returns:
        episodes_data: list of dicts with {
            'name': episode name,
            'frames': {frame_idx: {cam_idx: image_path}},
            'grasp_actions': numpy array of gripper values,
            'eef_pose': numpy array of end-effector poses (if load_eef_pose=True),
            'intervention_flags': numpy array of intervention flags (if load_intervention=True)
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

        episodes_data.append({
            'name': episode_dir.name,
            'path': episode_dir,
            'frames': frame_data,
            'num_frames': min_frames,
            'grasp_actions': grasp_actions,
            'eef_pose': eef_pose,
            'intervention_flags': intervention_flags
        })

        print(f"{min_frames} frames")

    return episodes_data


def create_trajectory_visualization(eef_pose, episode_name):
    """
    Create a 3D visualization of end-effector trajectory using Open3D.

    Args:
        eef_pose: numpy array of shape (N, 7) containing [x, y, z, qx, qy, qz, qw]
                  or (N, 3) containing just [x, y, z]
        episode_name: name of the episode for window title

    Returns:
        o3d.visualization.Visualizer: the visualizer object
    """
    if eef_pose is None or len(eef_pose) == 0:
        print("Warning: No eef_pose data to visualize")
        return None

    # Extract positions (first 3 columns)
    positions = eef_pose[:, :3] if eef_pose.shape[1] >= 3 else eef_pose

    # Create Open3D visualizer
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=f"Trajectory: {episode_name}", width=800, height=600)

    # Create trajectory line set
    points = positions
    lines = [[i, i + 1] for i in range(len(points) - 1)]

    # Create colors gradient from green (start) to red (end)
    colors = []
    for i in range(len(lines)):
        t = i / max(1, len(lines) - 1)
        # Green -> Yellow -> Red
        r = t
        g = 1.0 - t * 0.5
        b = 0.0
        colors.append([r, g, b])

    # Create LineSet
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(points)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.colors = o3d.utility.Vector3dVector(colors)

    # Add trajectory to visualizer
    vis.add_geometry(line_set)

    # Create point cloud for waypoints
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    # Color waypoints similarly
    point_colors = np.zeros((len(points), 3))
    for i in range(len(points)):
        t = i / max(1, len(points) - 1)
        point_colors[i] = [t, 1.0 - t * 0.5, 0.0]
    pcd.colors = o3d.utility.Vector3dVector(point_colors)
    vis.add_geometry(pcd)

    # Add start point marker (larger, green sphere)
    start_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.015)
    start_sphere.translate(points[0])
    start_sphere.paint_uniform_color([0, 1, 0])  # Green
    vis.add_geometry(start_sphere)

    # Add end point marker (larger, red sphere)
    end_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.015)
    end_sphere.translate(points[-1])
    end_sphere.paint_uniform_color([1, 0, 0])  # Red
    vis.add_geometry(end_sphere)

    # Add coordinate frame at origin
    coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
    vis.add_geometry(coord_frame)

    # Set view options
    render_option = vis.get_render_option()
    render_option.point_size = 5.0
    render_option.line_width = 3.0

    # Update geometry and view
    vis.poll_events()
    vis.update_renderer()

    return vis


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


def create_episode_frame_visualization(episode_data, frame_idx, max_height=600, show_intervention=False):
    """
    Create visualization for a single frame from one episode.

    Args:
        episode_data: episode data dict
        frame_idx: which frame to visualize
        max_height: maximum height for each camera image
        show_intervention: if True, show intervention indicator bar

    Returns:
        numpy array: concatenated image with all cameras and gripper/intervention bars
    """
    if frame_idx >= episode_data['num_frames']:
        return None

    frame_data = episode_data['frames'][frame_idx]
    images = []

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
            new_height = max_height
            img = cv2.resize(img, (new_width, new_height))

        # Add camera label
        label = f"Cam {cam_idx}"
        cv2.putText(img, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                   1, (0, 255, 0), 2)

        images.append(img)

    if not images:
        return None

    # Concatenate cameras horizontally
    concat_img = np.concatenate(images, axis=1)

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

    # Combine header, image, and bars
    final_img = np.vstack([header, concat_img] + bars)

    return final_img


def visualize_all_episodes(episodes_path, num_cams=3, use_segmented=False, teleop_mode=False, intervention_mode=False):
    """
    Interactive visualization of episodes' RGB images.

    Args:
        episodes_path: Path to episodes directory
        num_cams: Number of cameras
        use_segmented: If True, visualize segmented_rgb instead of rgb
        teleop_mode: If True, also show 3D trajectory visualization
        intervention_mode: If True, show intervention indicators and trajectory

    Controls:
        - Up/Down arrow: Navigate between episodes
        - Left/Right arrow: Navigate frames within episode
        - 'q' or ESC: Quit
        - Space: Auto-play frames
        - 'r': Reset to first frame
        - 'n': Next episode
        - 'p': Previous episode
    """
    # Intervention mode includes teleop features plus intervention tracking
    load_eef = teleop_mode or intervention_mode
    load_intv = intervention_mode

    episodes_data = load_all_episodes_images(episodes_path, num_cams, use_segmented,
                                             load_eef_pose=load_eef, load_intervention=load_intv)

    if not episodes_data:
        print("No episodes found or loaded!")
        return

    print(f"\nTotal episodes: {len(episodes_data)}")

    window_name = "Episode Viewer"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    current_episode_idx = 0
    current_frame = 0
    auto_play = False

    # Initialize Open3D visualizer for teleop/intervention mode
    trajectory_vis = None
    if teleop_mode or intervention_mode:
        episode = episodes_data[current_episode_idx]
        if episode.get('eef_pose') is not None:
            trajectory_vis = create_trajectory_visualization(episode['eef_pose'], episode['name'])

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
        controls_text = "Up/Down: Episodes | Left/Right: Frames | Space: Auto-play | Q: Quit"
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
            current_frame = max(0, current_frame - 1)
            auto_play = False
        elif key == 83 or key == 3:  # Right arrow
            current_frame = min(num_frames - 1, current_frame + 1)
            auto_play = False
        elif key == 82 or key == 0:  # Up arrow
            # Previous episode
            prev_idx = current_episode_idx
            current_episode_idx = max(0, current_episode_idx - 1)
            current_frame = 0  # Reset to first frame of new episode
            auto_play = False
            print(f"Switched to episode {current_episode_idx + 1}: {episodes_data[current_episode_idx]['name']}")

            # Update trajectory visualization if in teleop/intervention mode and episode changed
            if (teleop_mode or intervention_mode) and prev_idx != current_episode_idx:
                if trajectory_vis is not None:
                    trajectory_vis.close()
                new_episode = episodes_data[current_episode_idx]
                if new_episode.get('eef_pose') is not None:
                    trajectory_vis = create_trajectory_visualization(new_episode['eef_pose'], new_episode['name'])
                else:
                    trajectory_vis = None

        elif key == 84 or key == 1:  # Down arrow
            # Next episode
            prev_idx = current_episode_idx
            current_episode_idx = min(len(episodes_data) - 1, current_episode_idx + 1)
            current_frame = 0  # Reset to first frame of new episode
            auto_play = False
            print(f"Switched to episode {current_episode_idx + 1}: {episodes_data[current_episode_idx]['name']}")

            # Update trajectory visualization if in teleop/intervention mode and episode changed
            if (teleop_mode or intervention_mode) and prev_idx != current_episode_idx:
                if trajectory_vis is not None:
                    trajectory_vis.close()
                new_episode = episodes_data[current_episode_idx]
                if new_episode.get('eef_pose') is not None:
                    trajectory_vis = create_trajectory_visualization(new_episode['eef_pose'], new_episode['name'])
                else:
                    trajectory_vis = None
        elif key == ord(' '):  # Space
            auto_play = not auto_play
            print(f"Auto-play: {'ON' if auto_play else 'OFF'}")
        elif key == ord('r'):  # Reset
            current_frame = 0
            auto_play = False
            print("Reset to first frame")
        elif key == ord('n'):  # Next episode
            prev_idx = current_episode_idx
            current_episode_idx = min(len(episodes_data) - 1, current_episode_idx + 1)
            current_frame = 0
            auto_play = False
            print(f"Switched to episode {current_episode_idx + 1}: {episodes_data[current_episode_idx]['name']}")

            # Update trajectory visualization if in teleop/intervention mode and episode changed
            if (teleop_mode or intervention_mode) and prev_idx != current_episode_idx:
                if trajectory_vis is not None:
                    trajectory_vis.close()
                new_episode = episodes_data[current_episode_idx]
                if new_episode.get('eef_pose') is not None:
                    trajectory_vis = create_trajectory_visualization(new_episode['eef_pose'], new_episode['name'])
                else:
                    trajectory_vis = None

        elif key == ord('p'):  # Previous episode
            prev_idx = current_episode_idx
            current_episode_idx = max(0, current_episode_idx - 1)
            current_frame = 0
            auto_play = False
            print(f"Switched to episode {current_episode_idx + 1}: {episodes_data[current_episode_idx]['name']}")

            # Update trajectory visualization if in teleop/intervention mode and episode changed
            if (teleop_mode or intervention_mode) and prev_idx != current_episode_idx:
                if trajectory_vis is not None:
                    trajectory_vis.close()
                new_episode = episodes_data[current_episode_idx]
                if new_episode.get('eef_pose') is not None:
                    trajectory_vis = create_trajectory_visualization(new_episode['eef_pose'], new_episode['name'])
                else:
                    trajectory_vis = None

        # Auto-play: advance frame
        if auto_play:
            current_frame += 1
            if current_frame >= num_frames:
                # Move to next episode when frames end
                prev_idx = current_episode_idx
                current_episode_idx = min(len(episodes_data) - 1, current_episode_idx + 1)
                current_frame = 0
                if current_episode_idx == len(episodes_data) - 1:
                    # Loop back to first episode
                    current_episode_idx = 0
                print(f"Auto-play: Switched to episode {current_episode_idx + 1}")

                # Update trajectory visualization if in teleop/intervention mode and episode changed
                if (teleop_mode or intervention_mode) and prev_idx != current_episode_idx:
                    if trajectory_vis is not None:
                        trajectory_vis.close()
                    new_episode = episodes_data[current_episode_idx]
                    if new_episode.get('eef_pose') is not None:
                        trajectory_vis = create_trajectory_visualization(new_episode['eef_pose'], new_episode['name'])
                    else:
                        trajectory_vis = None

        # Update Open3D visualizer if it exists
        if trajectory_vis is not None:
            trajectory_vis.poll_events()
            trajectory_vis.update_renderer()

    # Cleanup
    cv2.destroyAllWindows()
    if trajectory_vis is not None:
        trajectory_vis.close()


def main():
    parser = argparse.ArgumentParser(
        description="Visualize episodes' RGB images from multiple cameras",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Visualize raw RGB images
  python record/inspect_raw_data.py /home/mingxi/mingxi_ws/crisp/crisp_py/raw_datasets/episodes

  # Visualize segmented RGB images
  python inspect_raw_data.py /home/mingxi/mingxi_ws/handpi/data/lift_block_realworld/output --segment

  # Visualize with 3D trajectory (teleop mode)
  python inspect_raw_data.py /home/mingxi/code/h2r_franka_ROS2/raw_datasets/episodes --teleop
  python inspect_raw_data.py /home/mingxi/temp_data/wrist_data --teleop

  # Visualize with intervention indicators (intervention mode)
  python inspect_raw_data.py /home/mingxi/mingxi_ws/crisp/crisp_py/raw_datasets/episodes --intv

Controls:
  Up/Down arrows: Navigate between episodes
  Left/Right arrows: Navigate frames within current episode
  Space: Toggle auto-play (plays frames and moves to next episode)
  N/P: Next/Previous episode
  R: Reset to first frame
  Q/ESC: Quit
        """
    )

    parser.add_argument(
        "path",
        type=str,
        help="Path to episodes root directory"
    )

    parser.add_argument(
        "--num-cams",
        type=int,
        default=3,
        help="Number of cameras (default: 3)"
    )

    parser.add_argument(
        "--segment",
        action="store_true",
        help="Visualize segmented_rgb instead of raw rgb images"
    )

    parser.add_argument(
        "--teleop",
        action="store_true",
        help="Enable teleop mode: show 3D trajectory visualization alongside RGB images"
    )

    parser.add_argument(
        "--intv",
        action="store_true",
        help="Enable intervention mode: show 3D trajectory, RGB images, and intervention indicators"
    )

    args = parser.parse_args()

    path = Path(args.path)

    if not path.exists():
        print(f"Error: Path does not exist: {path}")
        return

    visualize_all_episodes(path, args.num_cams, args.segment, args.teleop, args.intv)


if __name__ == "__main__":
    main()
