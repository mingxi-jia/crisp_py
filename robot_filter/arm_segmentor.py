# now we start coding the function

import time
import os
import copy 

from utils.pcd_utils import convert_RGBD_to_open3d
os.environ["MUJOCO_GL"] = "osmesa"
# from robosuite.controllers import load_controller_config
import numpy as np
import open3d as o3d
from urchin import URDF
import matplotlib.pyplot as plt
import tempfile
import imageio.v2 as imageio
import json
from scipy.spatial.transform import Rotation as R
from scipy.spatial import cKDTree

# take the camera data out of the function
# make this into a class that can be reused, so the camera data is stored in the class
# and the function can be called with different rgb, depth, and joints
# class RobotArmSegmentation:
class RobotArmSegmentation:
    def __init__(self, is_simulation=False):
        self.robot_urdf = None
        self.robot_urdf = None
        self.T_world_urdf = None  # Will be set later
        self.camera_name = None  # Will be set later

        if is_simulation:
            self.base_pose = np.array([-0.56, 0., 0.912])
            self.base_quat = np.array([1., 0., 0., 0.])  # Assuming no rotation, identity quaternion
        else:
            self.base_pose = np.array([0., 0., 0.])
            self.base_quat = np.array([1., 0., 0., 0.])
    
    def load_camera_metadata(self, camera_json_path):
        with open(camera_json_path, 'r') as f:
            self.camera_json = json.load(f)

    def set_camera(self, camera_name):  
        if camera_name not in self.camera_json:
            raise ValueError(f"Camera '{camera_name}' not found in camera metadata.")
        self.camera_name = camera_name
        
    def get_intrinsics_extrinsics(self):
        if not hasattr(self, 'camera_name'):
            raise ValueError("Camera name not set. Use set_camera() to set it.")
        intrinsics = np.array(self.camera_json[self.camera_name]["intrinsic"])
        extrinsics = np.array(self.camera_json[self.camera_name]["extrinsic"])
        extrinsics = np.linalg.inv(extrinsics)

        fx = intrinsics[0, 0]
        fy = intrinsics[1, 1]
        cx = intrinsics[0, 2]
        cy = intrinsics[1, 2]
        width = 128
        height = 128

        intrinsics = o3d.camera.PinholeCameraIntrinsic()
        intrinsics.set_intrinsics(width, height, fx, fy, cx, cy)
        return intrinsics, extrinsics
    
    #load urdf as part of the class as well
    def load_urdf(self, urdf_path):
        with open(urdf_path, 'r') as f:
            urdf_content = f.read()

        # Get the package directory (parent of 'urdf' directory)
        urdf_dir = os.path.dirname(os.path.abspath(urdf_path))
        package_dir = os.path.dirname(urdf_dir)

        # Replace package:// URIs with absolute paths
        urdf_str = urdf_content.replace("package://panda_description", package_dir)

        with tempfile.NamedTemporaryFile(mode="w+", suffix=".urdf", delete=False) as f:
            f.write(urdf_str)
            f.flush()
            self.robot_urdf = URDF.load(f.name)
            temp_file_path = f.name

        # Clean up the temp file after loading
        try:
            os.unlink(temp_file_path)
        except:
            pass

        # Assuming the robot is at the origin, no transformation needed
        # If you have a specific pose, set T_world_urdf accordingly

        self.T_world_urdf = np.eye(4)  # Identity matrix for no transformation
        # Convert quaternion from [w, x, y, z] to [x, y, z, w] format for scipy
        quat_xyzw = np.array([self.base_quat[1], self.base_quat[2], self.base_quat[3], self.base_quat[0]])
        self.T_world_urdf[:3, :3] = R.from_quat(quat_xyzw).as_matrix()  # rotation part
        self.T_world_urdf[:3, 3] = self.base_pose[:3]  # translation part

        return self.robot_urdf

    # Robot arm segmentation function
    def segment_one_camera(self, rgb, depth, joints):
        intrinsics, extrinsics = self.get_intrinsics_extrinsics()

        tsdf_volume = o3d.pipelines.integration.ScalableTSDFVolume(
            voxel_length=0.01,  # 1 cm voxel size
            sdf_trunc=0.04,     # 4 cm truncation distance
            color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8
        )

        # for loop three cams
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            color=o3d.geometry.Image(rgb.astype(np.uint8)),
            depth=o3d.geometry.Image((depth * 1000).astype(np.uint16)),  # mm for Open3D
            depth_scale=1000.0,
            convert_rgb_to_intensity=False
        )

        tsdf_volume.integrate(
            rgbd,
            intrinsic=intrinsics,
            extrinsic=extrinsics  # camera at origin
        )

        pcd = tsdf_volume.extract_point_cloud()

        # downsample pcd

        # Load URDF and create a mesh for the robot arm
        if not hasattr(self, 'robot_urdf'):
            raise ValueError("URDF not loaded. Use load_urdf() to load it.")

        # Get joint angles from simulation
        joint_names = [j.name for j in self.robot_urdf.actuated_joints]
        joint_positions = joints  # robosuite joint angles

        # Map names to values
        joint_angles = dict(zip(joint_names, joint_positions))

        # Compute transformed robot meshes
        robot_mesh_dict = self.robot_urdf.visual_trimesh_fk(cfg=joint_angles)  # returns mesh, transformation matrix relative to base

        # Sample points from each mesh surface
        sampled_points = []
        for mesh, pose in robot_mesh_dict.items():
            transformed = mesh.copy()
            transformed.apply_transform(pose)
            transformed.apply_transform(self.T_world_urdf)  # transform to world frame
            sampled_points.append(transformed.sample(5000))

        robot_points = np.vstack(sampled_points)

        # Create Open3D point cloud for KDTree
        robot_pcd = o3d.geometry.PointCloud()
        robot_pcd.points = o3d.utility.Vector3dVector(robot_points)

        # # Visual check: original scene (green) + robot samples (red)
        # scene_vis = o3d.geometry.PointCloud(pcd)
        # robot_vis = o3d.geometry.PointCloud(robot_pcd)

        # # print("Previewing alignment (green = scene, red = robot mesh)…")
        # # o3d.visualization.draw_geometries([scene_vis, robot_vis])

        # Filter scene points close to robot mesh
        robot_pcd = robot_pcd.voxel_down_sample(voxel_size=0.01)  # downsample for efficiency
        pcd = pcd.voxel_down_sample(voxel_size=0.01)  # downsample for efficiency

        FILTER_THRESH = 0.01
        robot_points_np = np.asarray(robot_pcd.points)  # Convert to numpy array for fast distance computation          
        scene_points_np = np.asarray(pcd.points)
        
        starting_time = time.time()
        # Use scipy.spatial.cKDTree for efficient vectorized distance queries
        robot_tree = cKDTree(robot_points_np)
        # Query for the minimum distance from each scene point to the robot mesh points
        dists, _ = robot_tree.query(scene_points_np)
        kept_mask = dists > FILTER_THRESH
        kept = scene_points_np[kept_mask]
        
        print(f"Filtering took {time.time() - starting_time:.2f} seconds")
        filtered_pcd = o3d.geometry.PointCloud(
            o3d.utility.Vector3dVector(np.asarray(kept))
        )
        filtered_pcd.colors = o3d.utility.Vector3dVector(np.asarray(pcd.colors)[kept_mask])  # Keep original colors
        # filtered_pcd.paint_uniform_color([0, 0, 1])
        print(f"Filtered point cloud has {len(filtered_pcd.points):,} points")
        
        # Optional: alpha-shape surface for debug
        mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(
            filtered_pcd, alpha=0.02
        )
        mesh.compute_vertex_normals()
        o3d.io.write_triangle_mesh("filtered_mesh.ply", mesh)
        o3d.io.write_point_cloud("filtered_point_cloud.ply", filtered_pcd)
        
        print(f"Filtered mesh  : {len(mesh.vertices):,} verts / {len(mesh.triangles):,} tris")

        # Visualise BEFORE vs AFTER
        # print("⬛  Showing green (orig) + blue (kept) overlay …")
        # o3d.visualization.draw_geometries([filtered_pcd])

        return filtered_pcd, pcd  # Return the filtered point cloud and original scene point cloud
    
    def segment_multi_camera(self, rgbd_dict, joints):
        if not hasattr(self, 'camera_json'):
            raise ValueError("Camera metadata not loaded.")

        # Integrate all camera views
        all_pcd = o3d.geometry.PointCloud()
        for camera_name, (rgb, depth) in rgbd_dict.items():
            cam_meta = self.camera_json[camera_name]
            intrinsics = np.array(cam_meta["intrinsic"])
            extrinsics = np.array(cam_meta["extrinsic"])
            pcd = convert_RGBD_to_open3d(rgb, depth[...,0], intrinsics, extrinsics)
            all_pcd += pcd

        # pcd = tsdf_volume.extract_point_cloud()
        pcd = all_pcd.voxel_down_sample(voxel_size=0.02)  # Downsample for efficiency

        # Load URDF and create a mesh for the robot arm
        if not hasattr(self, 'robot_urdf'):
            raise ValueError("URDF not loaded. Use load_urdf() to load it.")

        # Get joint angles from simulation
        joint_names = [j.name for j in self.robot_urdf.actuated_joints]
        joint_positions = joints  # robosuite joint angles

        # Map names to values
        joint_angles = dict(zip(joint_names, joint_positions))

        # Compute transformed robot meshes
        robot_mesh_dict = self.robot_urdf.visual_trimesh_fk(cfg=joint_angles)  # returns mesh, transformation matrix relative to base

        # Sample points from each mesh surface
        sampled_points = []
        for mesh, pose in robot_mesh_dict.items():
            transformed = mesh.copy()
            transformed.apply_transform(pose)
            transformed.apply_transform(self.T_world_urdf)  # transform to world frame
            sampled_points.append(transformed.sample(2500))

        robot_points = np.vstack(sampled_points)

        # Create Open3D point cloud for KDTree
        robot_pcd = o3d.geometry.PointCloud()
        robot_pcd.points = o3d.utility.Vector3dVector(robot_points)

        # Filter scene points close to robot mesh
        robot_pcd = robot_pcd.voxel_down_sample(voxel_size=0.02)  # downsample for efficiency

        FILTER_THRESH = 0.01
        robot_points_np = np.asarray(robot_pcd.points)  # Convert to numpy array for fast distance computation          
        scene_points_np = np.asarray(pcd.points)
        
        # Use scipy.spatial.cKDTree for efficient vectorized distance queries
        robot_tree = cKDTree(robot_points_np)
        # Query for the minimum distance from each scene point to the robot mesh points
        dists, _ = robot_tree.query(scene_points_np)
        kept_mask = dists > FILTER_THRESH
        kept = scene_points_np[kept_mask]
        
        filtered_pcd = o3d.geometry.PointCloud(
            o3d.utility.Vector3dVector(np.asarray(kept))
        )
        # filtered_pcd.colors = o3d.utility.Vector3dVector(np.asarray(pcd.colors)[kept_mask])  # Keep original colors
        
        # o3d.io.write_point_cloud("filtered_point_cloud.ply", filtered_pcd)

        return filtered_pcd, pcd  # Return the filtered point cloud and original scene point cloud

    def segment(self, original_pcd, joints):
        # Handle both Open3D PointCloud and numpy array inputs
        if isinstance(original_pcd, o3d.geometry.PointCloud):
            pcd = original_pcd
        else:
            # Assume it's a numpy array
            assert original_pcd.shape[1] == 6, "Input point cloud must be of shape (N, 6)"
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(original_pcd[:, :3])
            pcd.colors = o3d.utility.Vector3dVector(original_pcd[:, 3:6])   

        # pcd = tsdf_volume.extract_point_cloud()
        pcd = pcd.voxel_down_sample(voxel_size=0.02)  # Downsample for efficiency

        # Load URDF and create a mesh for the robot arm
        if not hasattr(self, 'robot_urdf'):
            raise ValueError("URDF not loaded. Use load_urdf() to load it.")

        # Get joint angles from simulation
        joint_names = [j.name for j in self.robot_urdf.actuated_joints]
        joint_positions = joints  # robosuite joint angles

        # Map names to values
        joint_angles = dict(zip(joint_names, joint_positions))

        # Compute transformed robot meshes
        robot_mesh_dict = self.robot_urdf.visual_trimesh_fk(cfg=joint_angles)  # returns mesh, transformation matrix relative to base

        # Sample points from each mesh surface
        sampled_points = []
        for mesh, pose in robot_mesh_dict.items():
            transformed = mesh.copy()
            transformed.apply_transform(pose)
            transformed.apply_transform(self.T_world_urdf)  # transform to world frame
            sampled_points.append(transformed.sample(2500))

        robot_points = np.vstack(sampled_points)

        # Create Open3D point cloud for KDTree
        robot_pcd = o3d.geometry.PointCloud()
        robot_pcd.points = o3d.utility.Vector3dVector(robot_points)

        # Filter scene points close to robot mesh
        robot_pcd = robot_pcd.voxel_down_sample(voxel_size=0.02)  # downsample for efficiency

        FILTER_THRESH = 0.025
        robot_points_np = np.asarray(robot_pcd.points)  # Convert to numpy array for fast distance computation          
        scene_points_np = np.asarray(pcd.points)
        
        # Use scipy.spatial.cKDTree for efficient vectorized distance queries
        robot_tree = cKDTree(robot_points_np)
        # Query for the minimum distance from each scene point to the robot mesh points
        dists, _ = robot_tree.query(scene_points_np)
        kept_mask = dists > FILTER_THRESH
        kept = scene_points_np[kept_mask]
        
        # filtered_pcd = o3d.geometry.PointCloud()
        # filtered_pcd.points = o3d.utility.Vector3dVector(np.asarray(kept))
        # filtered_pcd.colors = o3d.utility.Vector3dVector(np.asarray(pcd.colors)[kept_mask])  # Keep original colors
        filtered_pcd = np.concatenate([np.asarray(kept), np.asarray(pcd.colors)[kept_mask]], axis=1)

        return filtered_pcd
