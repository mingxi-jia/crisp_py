import h5py
from tqdm import tqdm
import numpy as np
from robot_filter.arm_segmentor import RobotArmSegmentation

import open3d as o3d
import os
import concurrent.futures

def save_visualization(geometries, save_path, width=800, height=600,viewpoint_json="camera_viewpoint.json"):
    """
    Render given Open3D geometries offscreen and save as image.

    Args:
        geometries (list): List of Open3D geometries to visualize.
        save_path (str): Path to save the image (e.g., 'visualizations/output.png').
        width (int): Width of the rendered image.
        height (int): Height of the rendered image.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    vis = o3d.visualization.Visualizer()
    vis.create_window(width=width, height=height, visible=False)

    for geom in geometries:
        vis.add_geometry(geom)

    view_control = vis.get_view_control()
    if os.path.exists(viewpoint_json):
        print(f"Loading custom viewpoint from '{viewpoint_json}'...")
        cam_params = o3d.io.read_pinhole_camera_parameters(viewpoint_json)
        view_control.convert_from_pinhole_camera_parameters(cam_params, allow_arbitrary=True)
    else:
        print("Custom viewpoint file not found. Using default view.")
        view_control.set_zoom(0.6)

    vis.poll_events()
    vis.update_renderer()
    vis.capture_screen_image(save_path, do_render=True)
    vis.destroy_window()



dataset_path = 'example_data/robomimic/stack_d1_rel.hdf5'
dataset = h5py.File(dataset_path, 'r')

demo = dataset['data/demo_0']
robot_joints = demo['obs']['robot0_joint_pos'][:]
gripper_joints = demo['obs']['robot0_gripper_qpos'][:]

num_traj = len(robot_joints)
print(f"Number of trajectories in the dataset: {num_traj}")

# Use multi-camera meta
robot_seg = RobotArmSegmentation(is_simulation=True)
robot_seg.load_camera_metadata("example_data/robomimic/multi_camera_meta.json")
robot_seg.load_urdf("panda_description/urdf/panda_arm_hand.urdf")
camera_names = ['spaceview', 'sideview2', 'backview']

# Preload all images and depths for all cameras
preloaded_rgbs = {}
preloaded_depths = {}
for cam in camera_names:
    preloaded_rgbs[cam] = demo['obs'][f'{cam}_image'][:]
    preloaded_depths[cam] = demo['obs'][f'{cam}_depth'][:]

# Build batch lists for testing
rgbd_dicts = []
joints_batch = []
for i in range(num_traj):
    joints = np.array([*robot_joints[i], gripper_joints[i][0]])
    rgbd_dict = {cam: (preloaded_rgbs[cam][i], preloaded_depths[cam][i]) for cam in camera_names}
    rgbd_dicts.append(rgbd_dict)
    joints_batch.append(joints)

def process_frame(i):
    joints = np.array([*robot_joints[i], gripper_joints[i][0]])
    rgbd_dict = {cam: (preloaded_rgbs[cam][i], preloaded_depths[cam][i]) for cam in camera_names}
    filtered_pcd, scene_pcd = robot_seg.segment_multi_camera(rgbd_dict, joints)
    # Optionally, visualize a few results (still commented out for speed)
    # scene_colored = o3d.geometry.PointCloud(scene_pcd)
    # scene_colored.paint_uniform_color([0, 1, 0])
    # filtered_colored = o3d.geometry.PointCloud(filtered_pcd)
    # save_visualization([scene_colored, filtered_colored], f"visualizations/frame_{i:04d}_overlay.png")
    return None


import time
start_time = time.time()
with concurrent.futures.ProcessPoolExecutor() as executor:
    list(executor.map(process_frame, range(num_traj)))
print(f"Total processing time for {num_traj} frames: {time.time() - start_time:.2f} seconds")
print(f"Average time per frame: {(time.time() - start_time)/num_traj:.2f} seconds")

dataset.close()