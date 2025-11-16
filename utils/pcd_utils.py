import os
import copy
from pathlib import Path
from typing import List, Tuple

import open3d as o3d
import numpy as np
import yaml
from PIL import Image
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt
from .transform_utils import apply_transform
from .workspace import WORKSPACE, VOXEL_RESOLUTION, WS_SIZE, VOXEL_SIZE


current_file_path = os.path.dirname(os.path.abspath(__file__))
gripper_asset = o3d.io.read_point_cloud(os.path.join(current_file_path, "./gripper.ply"))
xyz_numpy = np.asarray(gripper_asset.points).astype(np.float32)
color_numpy = np.asarray(gripper_asset.colors).astype(np.float32)
GRIPPER = np.concatenate([xyz_numpy, color_numpy], axis=1)
# GRIPPER = GRIPPER[GRIPPER[:,2]<-0.04] # delete gripper

num_points = 1024  # adjust the number of points as needed
r = 0.05  # sphere radius, modify r if needed
indices = np.arange(num_points, dtype=float) + 0.5
phi = np.arccos(1 - 2 * indices / num_points)
theta = np.pi * (1 + 5**0.5) * indices
x = r * np.sin(phi) * np.cos(theta)
y = r * np.sin(phi) * np.sin(theta)
z = r * np.cos(phi)
colors = np.zeros((num_points, 3), dtype=np.float32)
mask1 = (x >= 0) & (y >= 0) & (z >= 0)
mask2 = (x < 0) & (y >= 0) & (z >= 0)
mask3 = (x < 0) & (y < 0) & (z >= 0)
mask4 = (x >= 0) & (y < 0) & (z >= 0)
mask5 = (x >= 0) & (y >= 0) & (z < 0)
mask6 = (x < 0) & (y >= 0) & (z < 0)
mask7 = (x < 0) & (y < 0) & (z < 0)
mask8 = (x >= 0) & (y < 0) & (z < 0)

colors[mask1] = [1.0, 0.0, 0.0]   # red      : (+x, +y, +z)
colors[mask2] = [0.0, 1.0, 0.0]   # green    : (-x, +y, +z)
colors[mask3] = [0.0, 0.0, 1.0]   # blue     : (-x, -y, +z)
colors[mask4] = [1.0, 1.0, 0.0]   # yellow   : (+x, -y, +z)
colors[mask5] = [1.0, 0.0, 1.0]   # magenta  : (+x, +y, -z)
colors[mask6] = [0.0, 1.0, 1.0]   # cyan     : (-x, +y, -z)
colors[mask7] = [0.5, 0.5, 0.5]   # grey     : (-x, -y, -z)
colors[mask8] = [1.0, 0.5, 0.0]   # orange   : (+x, -y, -z)
# colors = np.tile(np.array([0., 1., 0.]), (num_points, 1)) 
# colors = np.where(y[:, None] < 0, np.array([0., 1., 0.]), np.array([0., 0., 1.]))
SPHERE = np.concatenate([np.stack([x, y, z], axis=1), colors], axis=1)

def np2o3d(pcd, color=None):
    # pcd: (n, 3)
    # color: (n, 3)
    pcd_o3d = o3d.geometry.PointCloud()
    pcd_o3d.points = o3d.utility.Vector3dVector(pcd)
    if color is not None and color.shape[0] > 0:
        assert pcd.shape[0] == color.shape[0]
        assert color.max() <= 1
        assert color.min() >= 0
        pcd_o3d.colors = o3d.utility.Vector3dVector(color)
    return pcd_o3d

def populate_point_num(pcd, point_num):
    if pcd.shape[0] < point_num:
        extra_choice = np.random.choice(pcd.shape[0], point_num-pcd.shape[0], replace=True)
        pcd = np.concatenate([pcd, pcd[extra_choice]], axis=0)
    else:
        shuffle_idx = np.random.permutation(pcd.shape[0])[:point_num]
        pcd = pcd[shuffle_idx]
    return pcd

def o3d2np(pcd_o3d, num_samples=4412):
    # pcd: (n, 3)
    # color: (n, 3)
    xyz = np.asarray(pcd_o3d.points)
    rgb = np.asarray(pcd_o3d.colors)
    num_points = xyz.shape[0]

    pcd_np = np.concatenate([xyz, rgb], axis=1)

    return pcd_np


def depth2fgpcd(depth, cam_params):
    # depth: (h, w)
    # fgpcd: (n, 3)
    # mask: (h, w)
    h, w = depth.shape
    mask = (depth > 0.0) & (depth < 2.0)  # filter out invalid depth values
    # mask = (depth <= 0.599/0.8)
    fgpcd = np.zeros((mask.sum(), 3))
    fx, fy, cx, cy = cam_params
    pos_x, pos_y = np.meshgrid(np.arange(w), np.arange(h))
    pos_x = pos_x[mask]
    pos_y = pos_y[mask]
    fgpcd[:, 0] = (pos_x - cx) * depth[mask] / fx
    fgpcd[:, 1] = (pos_y - cy) * depth[mask] / fy
    fgpcd[:, 2] = depth[mask]
    return fgpcd, mask

def convert_RGBD_fast(rgb, depth, intrinsics, extrinsics, max_points=100_000):
    fx, fy, cx, cy = intrinsics[0, 0], intrinsics[1, 1], intrinsics[0, 2], intrinsics[1, 2]
    mask = (depth > 0) & (depth < 2.0)
    ys, xs = np.where(mask)
    z = depth[ys, xs]
    x = (xs - cx) * z / fx
    y = (ys - cy) * z / fy
    points = np.stack((x, y, z), axis=1)

    # Apply extrinsics (rotate+translate)
    points_h = np.hstack((points, np.ones((points.shape[0], 1))))
    points_world = (points_h @ extrinsics.T)[:, :3]

    # Sample to reduce data size
    if points_world.shape[0] > max_points:
        idx = np.random.choice(points_world.shape[0], max_points, replace=False)
        points_world = points_world[idx]
        colors = rgb[ys[idx], xs[idx]] / 255.0
    else:
        colors = rgb[ys, xs] / 255.0

    # Return only numpy, defer Open3D creation
    return np.hstack((points_world, colors))

def convert_RGBD_to_open3d(rgb, depth, intrinsics, extrinsics):
    assert rgb.shape[0] == depth.shape[0] and rgb.shape[1] == depth.shape[1]
    assert intrinsics.shape == (3, 3)
    assert extrinsics.shape == (4, 4)
    assert rgb.dtype == np.uint8
    assert depth.dtype == np.float32 or depth.dtype == np.float64

    cam_param = [intrinsics[0, 0], intrinsics[1, 1], intrinsics[0, 2], intrinsics[1, 2]]
    pcd, mask = depth2fgpcd(depth, cam_param)
    # pose = np.linalg.inv(ext_mat)
    pose = extrinsics
    # trans_pcd = pose @ np.concatenate([pcd.T, np.ones((1, pcd.shape[0]))], axis=0)
    pcd_np = np.einsum('ij,jk->ik', pose, np.concatenate([pcd.T, np.ones((1, pcd.shape[0]))], axis=0))
    pcd_np = pcd_np[:3, :].T
    pcd_o3d = np2o3d(pcd_np, rgb[mask].reshape(-1, 3).astype(np.float64) / 255)
    return pcd_np, pcd_o3d

def get_extrinsics_matrix(t, q):
    """
    Convert translation and quaternion to extrinsics matrix.
    :param t: Translation vector (list or np.array of shape (3,))
    :param q: Quaternion (list or np.array of shape (4,))
    :return: Extrinsics matrix (np.array of shape (4, 4))
    """
    assert len(t) == 3 and len(q) == 4, "Translation must be of length 3 and quaternion of length 4."
    
    # Convert quaternion to rotation matrix
    x, y, z, w = q
    rot = R.from_quat([x, y, z, w]).as_matrix()
    
    # Create extrinsics matrix
    extrinsics = np.eye(4)
    extrinsics[:3, :3] = rot
    extrinsics[:3, 3] = t
    
    return extrinsics

def apply_se3_pcd_transform(points, transform):
    """
    Apply SE3 transformation to a set of points.
    points: a (N, 6) array where N is number of points and the last dimension is (x,y,z,r,g,b)
    """
    new_points = copy.deepcopy(points)
    points_homogeneous = np.hstack([new_points[:, :3], np.ones((new_points.shape[0], 1))])
    transformed_points = (transform @ points_homogeneous.T).T
    new_points[:, :3] = transformed_points[:, :3]
    return new_points

def render_pcd_from_pose(ee_pose, fix_point_num=1024, model_type='gripper'):
    """
    Render the gripper or sphere point cloud at the given end effector pose.
    ee_pose: (N,7) array [x,y,z,qx,qy,qz,qw]
    Returns: (N, fix_point_num, 6) point cloud array

    Coordinate convention: Z-axis points DOWN (robotics standard)
    """
    if model_type == 'gripper':
        base_pcd = GRIPPER[:, :3]
        base_color = GRIPPER[:, 3:6]
    elif model_type == 'sphere':
        base_pcd = SPHERE[:, :3]
        base_color = SPHERE[:, 3:6]
    else:
        raise NotImplementedError(f"model type {model_type} not implemented")

    ee_pose = ee_pose.reshape(-1, 7)
    batch_pcds = []

    for pose in ee_pose:
        rot = R.from_quat(pose[3:7]).as_matrix()
        # Quaternion already has Z-flip applied in apply_alignment_rotation()
        transformed_points = (rot @ base_pcd.T).T + pose[:3]
        batch_pcds.append(np.hstack([transformed_points, base_color]))

    return np.stack(batch_pcds, axis=0)[0]  # shape: (batch, fix_point_num, 6)

def o3d2np(pcd_o3d, num_samples=4412):
    # pcd: (n, 3)
    # color: (n, 3)
    xyz = np.asarray(pcd_o3d.points)
    rgb = np.asarray(pcd_o3d.colors)
    num_points = xyz.shape[0]

    return pcd_np
def filter_point_cloud_by_workspace(pcd, workspace_limits):
    """
    Efficiently filters an Open3D point cloud to keep only points within workspace_limits.

    Args:
        pcd (open3d.geometry.PointCloud): Input point cloud.
        workspace_limits (np.ndarray): (3,2) array [[x_min, x_max], [y_min, y_max], [z_min, z_max]]

    Returns:
        open3d.geometry.PointCloud: Filtered point cloud.
    """
    # Unpack limits
    x_min, x_max = workspace_limits[0]
    y_min, y_max = workspace_limits[1]
    z_min, z_max = workspace_limits[2]

    # Vectorized boolean mask
    pts = np.asarray(pcd.points)
    mask = (
        (pts[:, 0] >= x_min) & (pts[:, 0] <= x_max) &
        (pts[:, 1] >= y_min) & (pts[:, 1] <= y_max) &
        (pts[:, 2] >= z_min) & (pts[:, 2] <= z_max)
    )

    # Select valid indices using flatnonzero (faster than where)
    return pcd.select_by_index(np.flatnonzero(mask))



def random_downsample(pcd, num_samples):
    points = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors)
    idx = np.random.choice(points.shape[0], num_samples, replace=False)
    sampled_pcd = o3d.geometry.PointCloud()
    sampled_pcd.points = o3d.utility.Vector3dVector(points[idx])
    sampled_pcd.colors = o3d.utility.Vector3dVector(colors[idx])
    return sampled_pcd

def add_coordinate_axes_from_pose(position, quaternion, axis_length=0.1, fixed_point_num=1024):
    """
    Generate a coordinate axes mesh at a given pose and return its downsampled point cloud.

    Coordinate convention: Right-handed system with Z-axis pointing DOWN.
    - Red = X axis (forward/right)
    - Green = Y axis (left/right)
    - Blue = Z axis (DOWN - pointing toward ground)

    Args:
        position (array-like): (x, y, z) position.
        quaternion (array-like): (qx, qy, qz, qw) quaternion orientation.
        axis_length (float): Length of the coordinate axes.
        fixed_point_num (int): Number of points to sample from the coordinate axes.

    Returns:
        o3d.geometry.PointCloud: Downsampled point cloud of the coordinate axes.
    """
    # Convert quaternion to rotation matrix
    rot_matrix = R.from_quat(quaternion).as_matrix()

    # Construct 4x4 transformation matrix
    transform = np.eye(4)
    transform[:3, :3] = rot_matrix
    transform[:3, 3] = position

    # Create coordinate frame (default: Z-up)
    frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=axis_length)

    # # Flip Z-axis to point down (rotate 180° around X-axis)
    # # This converts from Z-up (default) to Z-down (robotics convention)
    # transform_axes = np.array([
    #     [ -1,  0,  0, 0],
    #     [ 0, -1,  0, 0],
    #     [ 0,  0, 1, 0],
    #     [ 0,  0,  0, 1]
    # ])
    # frame.transform(transform_axes)

    # Apply pose transformation
    frame.transform(transform)

    # convert frame mesh to point cloud
    frame_pcd = o3d.geometry.PointCloud()
    frame_pcd.points = o3d.utility.Vector3dVector(np.asarray(frame.vertices))
    frame_pcd.colors = o3d.utility.Vector3dVector(np.asarray(frame.vertex_colors))
    # Downsample the point cloud to fixed_point_num
    frame_pcd = frame_pcd.farthest_point_down_sample(num_samples=fixed_point_num)

    return frame_pcd

def pcd_to_voxel(pcds: np.ndarray, gripper_crop: float = None):
    assert pcds.shape[2] == 6, "PCD CONVERSION ERROR: pcd shape is incorrect"
    assert (pcds[0, :, 3:6] <= 1.).all(), "PCD CONVERSION ERROR: pcd color is incorrect"

    # Define voxel bounds
    if gripper_crop is None:
        voxel_bound = WORKSPACE.T
    else:
        voxel_bound = np.array([
            [-gripper_crop, gripper_crop],
            [-gripper_crop, gripper_crop],
            [-gripper_crop, gripper_crop]
        ]).T

    # Precompute voxel grid dimensions
    grid_min = voxel_bound[0]
    grid_max = voxel_bound[1]
    grid_size = ((grid_max - grid_min) / VOXEL_SIZE).astype(int)
    grid_size = np.clip(grid_size, 0, VOXEL_RESOLUTION)

    batch_voxels= []
    for i, pcd in enumerate(pcds):
        voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud_within_bounds(np2o3d(pcd[:,:3], pcd[:,3:]), voxel_size=VOXEL_SIZE, min_bound=voxel_bound[0], max_bound=voxel_bound[1])
        voxels = voxel_grid.get_voxels()  # returns list of voxels
        if len(voxels) == 0:
            np_voxels = np.zeros([4, VOXEL_RESOLUTION, VOXEL_RESOLUTION, VOXEL_RESOLUTION], dtype=np.uint8)
        else:
            indices = np.stack(list(vx.grid_index for vx in voxels))
            colors = np.stack(list(vx.color for vx in voxels))

            mask = (indices > 0) * (indices < VOXEL_RESOLUTION)
            indices = indices[mask.all(axis=1)]
            colors = colors[mask.all(axis=1)]

            np_voxels = np.zeros([4, VOXEL_RESOLUTION, VOXEL_RESOLUTION, VOXEL_RESOLUTION], dtype=np.uint8)
            np_voxels[0, indices[:, 0], indices[:, 1], indices[:, 2]] = 1
            np_voxels[1:, indices[:, 0], indices[:, 1], indices[:, 2]] = colors.T * 255
        
        batch_voxels.append(np_voxels)
    batch_voxels = np.stack(batch_voxels)
    return batch_voxels


def filter_by_workspace(pcd: o3d.geometry.PointCloud, workspace: List[float]) -> o3d.geometry.PointCloud:
    """
    Filters an Open3D point cloud based on workspace limits (a bounding box).

    Args:
        pcd (o3d.geometry.PointCloud): The input point cloud.
        workspace (List[float]): A list of 6 floats [x_min, x_max, y_min, y_max, z_min, z_max].

    Returns:
        o3d.geometry.PointCloud: The filtered point cloud.
    """
    if len(workspace) != 6:
        raise ValueError("Workspace must be a list of 6 floats: [x_min, x_max, y_min, y_max, z_min, z_max].")
    
    bbox = o3d.geometry.AxisAlignedBoundingBox(
        min_bound=(workspace[0], workspace[2], workspace[4]),
        max_bound=(workspace[1], workspace[3], workspace[5])
    )
    return pcd.crop(bbox)

def depth_to_point_cloud(depth_image: np.ndarray, intrinsics: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Converts a depth image to a point cloud in the camera frame.

    Args:
        depth_image (np.ndarray): The (H, W) depth image.
        intrinsics (np.ndarray): The (3, 3) camera intrinsics matrix.

    Returns:
        Tuple[np.ndarray, np.ndarray]: A tuple containing:
            - np.ndarray: The foreground point cloud (N, 3).
            - np.ndarray: The (H, W) boolean mask of the foreground points.
    """
    h, w = depth_image.shape
    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]

    mask = (depth_image > 0.0) & (depth_image < 2.0)  # Filter invalid depth
    
    y_map, x_map = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
    x_map = x_map[mask]
    y_map = y_map[mask]

    points = np.zeros((mask.sum(), 3), dtype=np.float32)
    points[:, 0] = (x_map - cx) * depth_image[mask] / fx
    points[:, 1] = (y_map - cy) * depth_image[mask] / fy
    points[:, 2] = depth_image[mask]

    return points, mask

def numpy_to_open3d(pcd_np: np.ndarray) -> o3d.geometry.PointCloud:
    """
    Converts a numpy array to an Open3D PointCloud object.

    Args:
        pcd_np (np.ndarray): The input numpy array. Can be (N, 3) for xyz
                                or (N, 6) for xyzrgb.

    Returns:
        o3d.geometry.PointCloud: The converted Open3D point cloud.
    """
    if pcd_np.shape[1] not in [3, 6]:
        raise ValueError("Input numpy array must have 3 or 6 columns (xyz or xyzrgb).")

    pcd_o3d = o3d.geometry.PointCloud()
    pcd_o3d.points = o3d.utility.Vector3dVector(pcd_np[:, :3])

    if pcd_np.shape[1] == 6:
        colors = pcd_np[:, 3:6]
        if np.max(colors) > 1.0 or np.min(colors) < 0.0:
            # Clamping color values to be safe
            colors = np.clip(colors, 0, 1)
        pcd_o3d.colors = o3d.utility.Vector3dVector(colors)

    return pcd_o3d

def rgbd_to_o3d(
        rgb_image: np.ndarray,
        depth_image: np.ndarray,
        intrinsics_mat: np.ndarray,
        extrinsics_mat: np.ndarray,
    ) -> Tuple[np.ndarray, o3d.geometry.PointCloud]:
        """
        Converts RGB and depth images to a transformed Open3D point cloud in the world frame.

        Args:
            rgb_image (np.ndarray): The (H, W, 3) RGB image (uint8).
            depth_image (np.ndarray): The (H, W) depth image (float).
            intrinsics (np.ndarray): The (3, 3) camera intrinsics matrix.
            extrinsics (np.ndarray): The (4, 4) camera extrinsics matrix (camera to world).

        Returns:
            Tuple[np.ndarray, o3d.geometry.PointCloud]: A tuple containing:
                - np.ndarray: The transformed point cloud as a numpy array (N, 3).
                - o3d.geometry.PointCloud: The transformed Open3D point cloud.
        """
        if not (rgb_image.shape[:2] == depth_image.shape and rgb_image.ndim == 3 and depth_image.ndim == 2):
            raise ValueError("RGB and Depth image dimensions do not match.")
        if not (intrinsics_mat.shape == (3, 3) and extrinsics_mat.shape == (4, 4)):
            raise ValueError("Invalid intrinsics or extrinsics matrix shape.")

        pcd, mask = depth_to_point_cloud(depth_image, intrinsics_mat)
        
        # Apply the extrinsics to transform points from camera to world frame
        pcd_world_np = apply_transform(pcd, extrinsics_mat)

        colors = rgb_image[mask].astype(np.float64) / 255.0
        pcd_world_o3d = numpy_to_open3d(np.hstack([pcd_world_np, colors]))

        return pcd_world_np, pcd_world_o3d

if __name__ == "__main__":

    data_path = "/home/mingxi/data/realworld/red_on_yellow_hand_0703/episode_20250703_125142_958"
    save_pcd_path = "/home/mingxi/data/realworld/debug"
    cam_views = [1, 2, 3]

    # load camera_info.yaml
    with open(os.path.join("configs", 'camera_info.yaml'), 'r') as f:
        cam_info = yaml.safe_load(f)

    frame_dict = dict()
    info_dict = dict()
    camera_frames = []
    num_frames = 0
    for i in cam_views:
        cam_name = f'cam{i}'
        intrinsics = np.array(cam_info[cam_name]['k']).reshape(3, 3)
        extrinsics = get_extrinsics_matrix(cam_info[cam_name]['t'], cam_info[cam_name]['q'])
        info_dict[cam_name] = {'intrinsics': intrinsics, 'extrinsics': extrinsics}

        frame_list = [f.split('.png')[0] for f in os.listdir(os.path.join(data_path, cam_name, 'rgb')) if f.endswith('.png')]
        frame_list.sort(key=lambda x: (int(x.split('_')[0]), int(x.split('_')[1])))
        if num_frames == 0:
            num_frames = len(frame_list)
        else:
            assert num_frames == len(frame_list), f"Number of frames mismatch in {cam_name}: {num_frames} vs {len(frame_list)}"
        frame_dict[cam_name] = frame_list

        cam_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
        cam_frame.transform(extrinsics)
        camera_frames.append(cam_frame)
    
    coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3, origin=[0, 0, 0])
    camera_frames.append(coordinate_frame)
    o3d.visualization.draw_geometries(camera_frames)
    

    # multi-view reconstruction example
    max_point_num = 4412
    x_min, y_min, z_min, ws_size = 0.32, -0.25, -0.02, 0.5
    

    for frame_i in range(num_frames):
        all_pcds = o3d.geometry.PointCloud()
        for i in cam_views:
            cam_name = f'cam{i}'

            # simulate different RGB and depth images for each view
            frame_name = frame_dict[cam_name][frame_i]
            rgb = np.asarray(Image.open(os.path.join(data_path, cam_name, 'rgb', f'{frame_name}.png')).convert('RGB'))
            depth = np.load(os.path.join(data_path, cam_name, 'depth', f'{frame_name}.npy')) / 1000.


            pcd_np, pcd_o3d = convert_RGBD_to_open3d(rgb, depth, info_dict[cam_name]['intrinsics'], info_dict[cam_name]['extrinsics'])
            all_pcds += pcd_o3d
            

            # coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.8, origin=[0, 0, 0])
            # # o3d.visualization.draw_geometries([filter_point_cloud_by_workspace(pcd_o3d, x_min, x_min+ws_size, y_min, y_min+ws_size, z_min, z_min+ws_size), coordinate_frame])
            # o3d.visualization.draw_geometries([pcd_o3d, coordinate_frame])

        
        all_pcds = filter_point_cloud_by_workspace(all_pcds, x_min, x_min+ws_size, y_min, y_min+ws_size, z_min, z_min+ws_size)
        all_pcds = all_pcds.farthest_point_down_sample(num_samples=max_point_num)

        exmaple_pose = np.array([0.4, 0., 0.2, 0., 0., 0., 1.])
        sphere = render_pcd_from_pose(exmaple_pose[None,...], fix_point_num=1024, model_type='sphere')[0]
        coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.8, origin=[0, 0, 0])

        # all_pcds += np2o3d(sphere[:, :3], sphere[:, 3:6])
        # o3d.visualization.draw_geometries([all_pcds, coordinate_frame])

        # Save the point cloud for this frame
        os.makedirs(save_pcd_path, exist_ok=True)
        file_name = os.path.join(save_pcd_path, f"{frame_i:04d}.ply")
        o3d.io.write_point_cloud(os.path.join(save_pcd_path, file_name), all_pcds)

