import os
import numpy as np
import open3d as o3d

import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from robot_filter.arm_segmentor import RobotArmSegmentation
from utils.camera_utils import load_camera_info
from utils.transform_utils import create_transform_matrix
from utils.pcd_utils import filter_by_workspace, rgbd_to_o3d
from PIL import Image

def get_pcd(data_path):
    
    cam_views = [1, 2, 3]

    # 3. Load camera information and visualize poses
    cam_info = load_camera_info(os.path.join("configs", 'camera_info.yaml'))
    cam_info_dict = {f'cam{i}': cam_info[f'cam{i}'] for i in cam_views}

    merged_pcd = o3d.geometry.PointCloud()
    for i in cam_views:
        cam_name = f'cam{i}'
        intrinsics = np.array(cam_info[cam_name]['k']).reshape(3, 3)
        extrinsics = create_transform_matrix(cam_info[cam_name]['t'], cam_info[cam_name]['q'])

        rgb_folder = os.path.join(data_path, cam_name, 'rgb')
        rgb_files = sorted([f for f in os.listdir(rgb_folder) if f.endswith('.png')])
        if not rgb_files:
            raise FileNotFoundError(f"No PNG files found in {rgb_folder}")
        first_frame = rgb_files[0].replace('.png', '') 
        rgb_path = os.path.join(data_path, cam_name, 'rgb', f'{first_frame}.png')
        depth_path = os.path.join(data_path, cam_name, 'depth', f'{first_frame}.npy')
        rgb = np.asarray(Image.open(rgb_path).convert('RGB'))
        depth = np.load(depth_path) / 1000.0  # Convert to meters

        _, pcd_o3d = rgbd_to_o3d(rgb, depth, intrinsics, extrinsics)
        merged_pcd += pcd_o3d

    merged_pcd = merged_pcd.voxel_down_sample(voxel_size=0.02)

    # 5. Process the merged point cloud
    # workspace_limits = [0.2, 0.7, -0.2, 0.2, -0.04, 0.5]
    # merged_pcd = filter_by_workspace(merged_pcd, workspace_limits)
    print(f"Points after workspace filtering: {len(merged_pcd.points)}")
    o3d.visualization.draw_geometries([merged_pcd])

    return merged_pcd

data_path = "example_data/multiview_rgbd"
pcd = get_pcd(data_path)
#TODO: use arm segmentor (we need to record pcd with robot joint states first)

joint_states = np.load(os.path.join(data_path, "state", "joint_state.npy"))[0]

robot_seg = RobotArmSegmentation()
robot_seg.load_urdf("robot_filter/panda_description/urdf/panda_arm_hand.urdf")

filtered_pcd = robot_seg.segment(pcd, joint_states)
o3d.visualization.draw_geometries([filtered_pcd])