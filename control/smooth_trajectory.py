"""Smooth end-effector trajectory (position + orientation) and visualize in Open3D.

Usage:
    conda activate pink_solver
    python smooth_trajectory.py --window 5
    python smooth_trajectory.py --window 7 --save
"""

import argparse
import numpy as np
import h5py
import open3d as o3d
from scipy.ndimage import uniform_filter1d
from scipy.signal import savgol_filter
from scipy.spatial.transform import Rotation as R, Slerp

# Configuration
HDF5_FILE = "/media/mingxi/T7/XEMB_Experiment/coffee_prep/replay_hand_test/test_2.hdf5"
HDF5_FILE = "../coffee_making/coffee_making_d1_24_realworld_pretrain.hdf5"


def load_ee_trajectory_from_hdf5(file_path: str):
    """Load end-effector positions and orientations from HDF5 file."""
    dataset = h5py.File(file_path, "r")
    data = dataset['data']['demo_0']
    ee_pos = data['obs']['robot0_eef_pos'][:]
    ee_quat = data['obs']['robot0_eef_quat'][:]
    dataset.close()
    return ee_pos.astype(float), ee_quat.astype(float)


def save_smoothed_hdf5(original_path: str, smoothed_pos, smoothed_quat):
    """Save smoothed trajectory to a new HDF5 file."""
    if original_path.endswith('.hdf5'):
        output_path = original_path[:-5] + '_smoothed.hdf5'
    else:
        output_path = original_path + '_smoothed.hdf5'

    with h5py.File(original_path, 'r') as src:
        with h5py.File(output_path, 'w') as dst:
            def copy_item(name, obj):
                if isinstance(obj, h5py.Dataset):
                    if name == 'data/demo_0/obs/robot0_eef_pos':
                        dst.create_dataset(name, data=smoothed_pos.astype(obj.dtype))
                    elif name == 'data/demo_0/obs/robot0_eef_quat':
                        dst.create_dataset(name, data=smoothed_quat.astype(obj.dtype))
                    else:
                        dst.create_dataset(name, data=obj[:])
                elif isinstance(obj, h5py.Group):
                    dst.create_group(name)

            src.visititems(copy_item)

            for key, val in src.attrs.items():
                dst.attrs[key] = val

    print(f"Saved smoothed trajectory to: {output_path}")
    return output_path


def smooth_positions(trajectory, method='savgol', window=5, poly_order=2):
    """Smooth position trajectory."""
    if window < 3:
        return trajectory

    if method == 'savgol' and window % 2 == 0:
        window += 1

    if method == 'savgol':
        smoothed = savgol_filter(trajectory, window, poly_order, axis=0)
    elif method == 'moving_avg':
        smoothed = uniform_filter1d(trajectory, size=window, axis=0, mode='nearest')
    elif method == 'gaussian':
        from scipy.ndimage import gaussian_filter1d
        sigma = window / 4.0
        smoothed = gaussian_filter1d(trajectory, sigma=sigma, axis=0, mode='nearest')
    else:
        raise ValueError(f"Unknown method: {method}")

    return smoothed


def smooth_orientations(quaternions, method='slerp', window=5):
    """Smooth orientation trajectory using SLERP-based averaging.

    Args:
        quaternions: (N, 4) array of quaternions [qx, qy, qz, qw]
        method: 'slerp' (SLERP-based) or 'rotvec' (rotation vector smoothing)
        window: Window size for smoothing

    Returns:
        Smoothed quaternions
    """
    if window < 3:
        return quaternions

    n = len(quaternions)
    smoothed = np.zeros_like(quaternions)
    half_window = window // 2

    if method == 'slerp':
        # SLERP-based smoothing: interpolate to middle of window
        for i in range(n):
            start_idx = max(0, i - half_window)
            end_idx = min(n - 1, i + half_window)

            if start_idx == end_idx:
                smoothed[i] = quaternions[i]
            else:
                # Use SLERP between start and end of window
                rotations = R.from_quat(quaternions[[start_idx, end_idx]])
                times = [0, 1]
                slerp = Slerp(times, rotations)
                # Interpolate to middle
                t = (i - start_idx) / (end_idx - start_idx)
                smoothed[i] = slerp(t).as_quat()

    elif method == 'rotvec':
        # Convert to rotation vectors, smooth, convert back
        rotvecs = R.from_quat(quaternions).as_rotvec()

        # Ensure odd window for savgol
        if window % 2 == 0:
            window += 1

        # Smooth rotation vectors
        rotvecs_smooth = savgol_filter(rotvecs, window, 2, axis=0)

        # Convert back to quaternions
        smoothed = R.from_rotvec(rotvecs_smooth).as_quat()

    else:
        raise ValueError(f"Unknown method: {method}")

    return smoothed


def create_line_set(positions, color):
    """Create Open3D line set from positions."""
    lines = [[i, i + 1] for i in range(len(positions) - 1)]
    colors = [color for _ in lines]

    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(positions)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.colors = o3d.utility.Vector3dVector(colors)
    return line_set


def create_orientation_arrows(positions, quaternions, color, scale=0.02, step=5):
    """Create arrows showing orientation at each position."""
    arrows = []
    for i in range(0, len(positions), step):
        # Create arrow mesh
        arrow = o3d.geometry.TriangleMesh.create_arrow(
            cylinder_radius=scale * 0.3,
            cone_radius=scale * 0.6,
            cylinder_height=scale * 0.7,
            cone_height=scale * 0.3,
        )
        arrow.paint_uniform_color(color)

        # Rotate arrow to match orientation (arrow points in +Z by default)
        rot = R.from_quat(quaternions[i])
        arrow.rotate(rot.as_matrix(), center=[0, 0, 0])

        # Translate to position
        arrow.translate(positions[i])
        arrows.append(arrow)

    return arrows


def visualize_trajectories(pos_orig, quat_orig, pos_smooth, quat_smooth):
    """Visualize original and smoothed trajectories in Open3D."""
    # Lines
    line_original = create_line_set(pos_orig, [1, 0, 0])
    line_smoothed = create_line_set(pos_smooth, [0, 1, 0])

    # Orientation arrows
    arrows_orig = create_orientation_arrows(pos_orig, quat_orig, [1, 0.5, 0.5], step=10)
    arrows_smooth = create_orientation_arrows(pos_smooth, quat_smooth, [0.5, 1, 0.5], step=10)

    # Coordinate frame
    coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)

    print("\n" + "=" * 60)
    print("TRAJECTORY VISUALIZATION")
    print("=" * 60)
    print(f"Original (RED):   {len(pos_orig)} points")
    print(f"Smoothed (GREEN): {len(pos_smooth)} points")
    print("\nArrows show orientation (every 10th point)")
    print("Controls: Left=Rotate, Right=Pan, Scroll=Zoom, Q=Quit")
    print("=" * 60)

    geometries = [coord_frame, line_original, line_smoothed] + arrows_orig + arrows_smooth

    o3d.visualization.draw_geometries(
        geometries,
        window_name="Red=Original, Green=Smoothed (with orientation arrows)",
        width=1200,
        height=800,
    )


def main():
    parser = argparse.ArgumentParser(description='Smooth and visualize EE trajectory')
    parser.add_argument('--hdf5', type=str, default=HDF5_FILE, help='HDF5 file path')
    parser.add_argument('--method', type=str, default='savgol',
                        choices=['savgol', 'moving_avg', 'gaussian'],
                        help='Position smoothing method')
    parser.add_argument('--ori-method', type=str, default='rotvec',
                        choices=['slerp', 'rotvec'],
                        help='Orientation smoothing method')
    parser.add_argument('--window', type=int, default=5, help='Smoothing window size')
    parser.add_argument('--save', action='store_true', help='Save smoothed trajectory')
    parser.add_argument('--no-viz', action='store_true', help='Skip visualization')
    args = parser.parse_args()

    # Load trajectory
    print(f"Loading trajectory from {args.hdf5}...")
    positions, quaternions = load_ee_trajectory_from_hdf5(args.hdf5)
    print(f"Loaded {len(positions)} points")

    # Smooth positions
    print(f"Smoothing positions with {args.method} (window={args.window})...")
    pos_smoothed = smooth_positions(positions, method=args.method, window=args.window)

    # Smooth orientations
    print(f"Smoothing orientations with {args.ori_method} (window={args.window})...")
    quat_smoothed = smooth_orientations(quaternions, method=args.ori_method, window=args.window)

    # Save if requested
    if args.save:
        save_smoothed_hdf5(args.hdf5, pos_smoothed, quat_smoothed)

    # Visualize
    if not args.no_viz:
        visualize_trajectories(positions, quaternions, pos_smoothed, quat_smoothed)


if __name__ == "__main__":
    main()
