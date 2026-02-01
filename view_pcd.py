#!/usr/bin/env python3
"""Simple script to visualize a random point cloud from an HDF5 file."""

import argparse
import random
import h5py
import open3d as o3d
import numpy as np

point_size = 15
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("hdf5_path", type=str, help="Path to the HDF5 file")
    parser.add_argument("-n", "--num_samples", type=int, default=1, help="Number of times to sample")
    args = parser.parse_args()

    with h5py.File(args.hdf5_path, "r") as f:
        dataset = f['data']
        demo_keys = [k for k in dataset.keys() if k.startswith("demo")]

        for i in range(args.num_samples):
            demo_key = random.choice(demo_keys)
            pcd_data = dataset[demo_key]["obs"]["render_pcd"][:]
            frame_idx = random.randint(0, len(pcd_data) - 1)
            points = pcd_data[frame_idx]

            print(f"[{i+1}/{args.num_samples}] Demo: {demo_key}, Frame index: {frame_idx}")

            pcd = o3d.geometry.PointCloud()
            print(f"points.shape: {points.shape}")
            pcd.points = o3d.utility.Vector3dVector(points[:, :3])
            if points.shape[1] >= 6:
                pcd.colors = o3d.utility.Vector3dVector(points[:, 3:6])
            
            # Create visualizer with custom point size
            vis = o3d.visualization.Visualizer()
            vis.create_window()
            vis.add_geometry(pcd)

            x_range = [0.3, 0.9]
            y_range = [-0.3, 0.3]
            z_ground = 0.01
            
            # Create a mesh for the ground
            ground_mesh = o3d.geometry.TriangleMesh()
            vertices = np.array([
                [x_range[0], y_range[0], z_ground],  # corner 1
                [x_range[1], y_range[0], z_ground],  # corner 2
                [x_range[1], y_range[1], z_ground],  # corner 3
                [x_range[0], y_range[1], z_ground],  # corner 4
            ])
            triangles = np.array([
                [0, 1, 2],  # first triangle
                [0, 2, 3],  # second triangle
            ])
            ground_mesh.vertices = o3d.utility.Vector3dVector(vertices)
            ground_mesh.triangles = o3d.utility.Vector3iVector(triangles)
            
            # Color the ground (e.g., light gray)
            ground_mesh.paint_uniform_color([0.9, 0.2, 0.3])
            
            # Add to visualizer
            vis.add_geometry(ground_mesh)
            vis.get_render_option().point_size = point_size
            vis.run()
            vis.destroy_window()


if __name__ == "__main__":
    main()