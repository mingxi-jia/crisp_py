#!/usr/bin/env python3
"""Interactive debug visualization for chunking control data.

This GUI application loads the latest debug data and provides:
- Interactive 3D point cloud visualization
- Timeline slider for navigating through events
- Side panel with event information and statistics
- Real-time EEF pose and coordinate frame visualization
"""

import numpy as np
from pathlib import Path
from datetime import datetime
import open3d as o3d
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt



class DebugChunkingViewer:
    """Interactive viewer for chunking control debug data using Open3D GUI."""

    def __init__(self, debug_data_dir='debug_data'):
        """Initialize the viewer.

        Args:
            debug_data_dir: Path to debug data directory
        """
        self.debug_data_dir = Path(debug_data_dir)
        self.data = None
        self.window = None
        self.scene_widget = None
        self.info_text = None
        self.timeline_slider = None
        self.current_index = 0

        # Geometry objects for dynamic updating
        self.pcd_geometry = None
        self.eef_sphere = None
        self.coord_frame = None

        # Load latest data
        self._load_latest_data()


    def _load_latest_data(self):
        """Load the most recent debug data files."""
        print("Loading latest debug data...")

        # Find latest files in each subdirectory
        action_chunks_dir = self.debug_data_dir / 'action_chunks'
        actions_dir = self.debug_data_dir / 'actions'
        eef_poses_dir = self.debug_data_dir / 'eef_poses'
        pcd_dir = self.debug_data_dir / 'pcd'
        robot0_eef_pos_dir = self.debug_data_dir / 'robot0_eef_pos'
        actual_eef_pos_dir = self.debug_data_dir / 'actual_eef_pos'
        pcd_timestamps_dir = self.debug_data_dir / 'pcd_timestamps'

        # Get latest files
        latest_action_chunks = self._get_latest_file(action_chunks_dir, 'action_chunks_*.npy')
        latest_actions = self._get_latest_file(actions_dir, 'executed_actions_*.npy')
        latest_eef_poses = self._get_latest_file(eef_poses_dir, 'eef_poses_*.npy')
        latest_pcd = self._get_latest_file(pcd_dir, 'point_clouds_*.npy')
        latest_robot0_eef_pos = self._get_latest_file(robot0_eef_pos_dir, 'robot0_eef_pos_*.npy')
        latest_actual_eef_pos = self._get_latest_file(actual_eef_pos_dir, 'actual_eef_pos_*.npy')
        latest_pcd_timestamps = self._get_latest_file(pcd_timestamps_dir, 'pcd_timestamps_*.npy')

        if not all([latest_action_chunks, latest_actions, latest_eef_poses, latest_pcd, latest_robot0_eef_pos, latest_actual_eef_pos, latest_pcd_timestamps]):
            print("ERROR: Could not find all required data files!")
            print(f"  Action chunks: {latest_action_chunks}")
            print(f"  Executed actions: {latest_actions}")
            print(f"  EEF poses: {latest_eef_poses}")
            print(f"  Point clouds: {latest_pcd}")
            print(f"  Robot0 EEF pos: {latest_robot0_eef_pos}")
            print(f"  Actual EEF pos: {latest_actual_eef_pos}")
            print(f"  RGBD timestamps: {latest_pcd_timestamps}")
            raise FileNotFoundError("Missing debug data files")

        # Load data
        print(f"\nLoading files:")
        print(f"  Action chunks: {latest_action_chunks.name}")
        print(f"  Executed actions: {latest_actions.name}")
        print(f"  EEF poses: {latest_eef_poses.name}")
        print(f"  Point clouds: {latest_pcd.name}")
        print(f"  Robot0 EEF pos: {latest_robot0_eef_pos.name}")
        print(f"  Actual EEF pos: {latest_actual_eef_pos.name}")
        print(f"  RGBD timestamps: {latest_pcd_timestamps.name}")

        action_chunks_data = np.load(latest_action_chunks, allow_pickle=True).item()
        actions_data = np.load(latest_actions, allow_pickle=True).item()
        eef_poses_data = np.load(latest_eef_poses, allow_pickle=True).item()
        pcd_data = np.load(latest_pcd, allow_pickle=True).item()
        robot0_eef_pos_data = np.load(latest_robot0_eef_pos, allow_pickle=True).item()
        actual_eef_pos_data = np.load(latest_actual_eef_pos, allow_pickle=True).item()
        pcd_timestamps_data = np.load(latest_pcd_timestamps, allow_pickle=True).item()

        self.data = {
            'action_chunks': {
                'timestamps_ms': action_chunks_data['timestamps_ms'],
                'data': action_chunks_data['data']
            },
            'executed_actions': {
                'timestamps_ms': actions_data['timestamps_ms'],
                'data': actions_data['data']
            },
            'eef_poses': {
                'timestamps_ms': eef_poses_data['timestamps_ms'],
                'positions': eef_poses_data['positions'],
                'orientations_quat': eef_poses_data['orientations_quat'],
                'orientations_matrix': eef_poses_data['orientations_matrix']
            },
            'point_clouds': {
                'timestamps_ms': pcd_data['timestamps_ms'],
                'point_clouds': pcd_data['point_clouds'],
                'point_counts': pcd_data['point_counts']
            },
            'robot0_eef_pos': {
                'timestamps_ms': robot0_eef_pos_data['timestamps_ms'],
                'data': robot0_eef_pos_data['data']
            },
            'actual_eef_pos': {
                'timestamps_ms': actual_eef_pos_data['timestamps_ms'],
                'data': actual_eef_pos_data['data']
            },
            'pcd_timestamps': {
                'timestamps_ms': pcd_timestamps_data['timestamps_ms'],
                'data': pcd_timestamps_data['data']
            }
        }

        print(f"\nData loaded successfully:")
        print(f"  Action chunks: {len(self.data['action_chunks']['timestamps_ms'])} samples")
        print(f"  Executed actions: {len(self.data['executed_actions']['timestamps_ms'])} samples")
        print(f"  EEF poses: {len(self.data['eef_poses']['timestamps_ms'])} samples")
        print(f"  Point clouds: {len(self.data['point_clouds']['timestamps_ms'])} samples")
        print(f"  Robot0 EEF pos: {len(self.data['robot0_eef_pos']['timestamps_ms'])} samples")
        print(f"  Actual EEF pos: {len(self.data['actual_eef_pos']['timestamps_ms'])} samples")
        print(f"  RGBD timestamps: {len(self.data['pcd_timestamps']['timestamps_ms'])} samples")

    def _get_latest_file(self, directory, pattern):
        """Get the most recent file matching pattern in directory.

        Args:
            directory: Path to search
            pattern: Glob pattern to match

        Returns:
            Path to latest file or None
        """
        if not directory.exists():
            return None

        files = list(directory.glob(pattern))
        if not files:
            return None

        return max(files, key=lambda p: p.stat().st_mtime)
    
    def _item_idx_to_chunk_idx(self, item_index):
        chunk_idx = (item_index - self.policy_delay) // self.action_exec_size + 1
        return chunk_idx

    def go(self):
        self.action_exec_size = 8
        self.policy_delay = 3

        chunk_index_start = 1
        chunk_index_end = 5
        ci_s = chunk_index_start
        ci_e = chunk_index_end
        actual_index_start = (ci_s - 1) * self.action_exec_size + self.policy_delay
        actual_index_end = (ci_e - 1) * self.action_exec_size + self.policy_delay

        
        print("\n========== TCP Positions (xyz): ==========")
        print(self.data['robot0_eef_pos']['data'].shape)
        positions = np.asarray(self.data['robot0_eef_pos']['data'])
        for i, pos in enumerate(positions[ci_s:ci_e]):
            i += ci_s
            x, y, z = pos
            # print(f"{i}: \t{x:.6f}\t{y:.6f}\t{z:.6f}")
            print(f"{i}: \t{z:.6f}")

        action_chunks = self.data['action_chunks']['data'][:3, ...]
        print(self.data['action_chunks']['data'].shape)
        print("Sample action chunks (first 3):")
        for i, chunk in enumerate(action_chunks[ci_s:ci_e]):
            i += ci_s
            for j, action in enumerate(chunk[:12]):
                x, y, z = action[:3]
                # print(f"{i, j}: \t{x:.6f}\t{y:.6f}\t{z:.6f}")
                print(f"{i, j}: \t{z:.6f}")                
        print("========== Actual robot xyz: ==========")
        print(self.data['actual_eef_pos']['data'].shape)
        for i, pos in enumerate(self.data['actual_eef_pos']['data'][actual_index_start:actual_index_end]):
            i += actual_index_start
            x, y, z = pos
            # print(f"{i}: \t{x:.6f}\t{y:.6f}\t{z:.6f}")
            print(f"{i}: \t{z:.6f}")

        print("\nActual Action Executed:")
        print(self.data['executed_actions']['data'].shape)
        for i, pos in enumerate(self.data['executed_actions']['data'][actual_index_start:actual_index_end]):
            i += actual_index_start
            x, y, z = pos[:3]
            # print(f"{i}: \t{x:.6f}\t{y:.6f}\t{z:.6f}")
            print(f"{i}: \t{z:.6f}")


        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Robot End-Effector Trajectories (Y vs Z)', fontsize=16)

        # 1. TCP Positions with chunk index labels
        ax1 = axes[0, 0]
        positions = np.asarray(self.data['robot0_eef_pos']['data'])
        positions_slice = positions[ci_s:ci_e]
        ax1.plot(positions_slice[:, 1], positions_slice[:, 2], 'b-', linewidth=2, label='TCP trajectory')
        ax1.scatter(positions_slice[0, 1], positions_slice[0, 2], c='green', s=100, marker='o', label='Start', zorder=5)
        ax1.scatter(positions_slice[-1, 1], positions_slice[-1, 2], c='red', s=100, marker='x', label='End', zorder=5)

        # Add chunk index labels (ci_s, ci_s+1, ci_s+2, ...)
        labeled_points_y = []
        labeled_points_z = []
        for i, pos in enumerate(positions_slice):
            chunk_idx = ci_s + i
            # Label every few points to avoid cluttering
            labeled_points_y.append(pos[1])
            labeled_points_z.append(pos[2])
            ax1.annotate(f'{chunk_idx}', (pos[1], pos[2]), 
                        textcoords="offset points", xytext=(5, 5), 
                        fontsize=8, alpha=0.7)

        # Mark the labeled points
        ax1.scatter(labeled_points_y, labeled_points_z, c='blue', s=50, marker='o', 
                    edgecolors='black', linewidths=1, zorder=4, label='Labeled chunks')

        ax1.set_xlabel('Y position')
        ax1.set_ylabel('Z position')
        ax1.set_title('TCP Positions (with chunk indices)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # 2. Action Chunks with color gradient and chunk index labels
        ax2 = axes[0, 1]
        action_chunks = self.data['action_chunks']['data']
        action_chunks_slice = action_chunks[ci_s:ci_e]

        # Create color gradient from light to dark
        n_chunks = len(action_chunks_slice)
        colors = plt.cm.viridis(np.linspace(0.3, 0.9, n_chunks))

        for i, (chunk, color) in enumerate(zip(action_chunks_slice, colors)):
            chunk_idx = ci_s + i

            # chunk_12 = chunk[4:12]
            # y_vals = chunk_12[:, 1]
            # z_vals = chunk_12[:, 2]
            # ax2.plot(y_vals, z_vals, '-o', color=color, linewidth=1.5, 
            #         markersize=4, label=f'Chunk {ci_s+i}', alpha=0.7)
            chunk_16 = chunk[:16]
            y_vals = chunk_16[:, 1]
            z_vals = chunk_16[:, 2]
            
            # Plot all 16 steps with thinner line and smaller markers
            ax2.plot(y_vals, z_vals, '-o', color=color, linewidth=1.0, 
                    markersize=3, alpha=0.4)
            
            # Highlight the middle 8 (indices 4-11)
            y_vals_middle = chunk_16[self.policy_delay:self.policy_delay+self.action_exec_size, 1]
            z_vals_middle = chunk_16[self.policy_delay:self.policy_delay+self.action_exec_size, 2]
            ax2.plot(y_vals_middle, z_vals_middle, '-o', color=color, linewidth=2.0, 
                    markersize=5, label=f'Chunk {chunk_idx}', alpha=0.9)
            
            # Label the start of each chunk trajectory
            ax2.annotate(f'{chunk_idx}', (y_vals[0], z_vals[0]), 
                        textcoords="offset points", xytext=(5, 5), 
                        fontsize=8, fontweight='bold', color=color)

        ax2.set_xlabel('Y position')
        ax2.set_ylabel('Z position')
        ax2.set_title('Action Chunks (light→dark = early→late)')
        ax2.grid(True, alpha=0.3)
        # Only show legend if not too many chunks
        if n_chunks <= 10:
            ax2.legend(fontsize=8, loc='best')

        # 3. Actual End-Effector Positions with chunk index labels
        ax3 = axes[1, 0]
        actual_positions = self.data['actual_eef_pos']['data'][actual_index_start:actual_index_end]
        ax3.plot(actual_positions[:, 1], actual_positions[:, 2], 'g-', linewidth=2, label='Actual trajectory')
        ax3.scatter(actual_positions[0, 1], actual_positions[0, 2], c='green', s=100, marker='o', label='Start', zorder=5)
        ax3.scatter(actual_positions[-1, 1], actual_positions[-1, 2], c='red', s=100, marker='x', label='End', zorder=5)

        # Add chunk index labels
        labeled_points_y_3 = []
        labeled_points_z_3 = []
        for i, pos in enumerate(actual_positions):
            item_index = actual_index_start + i
            chunk_idx = self._item_idx_to_chunk_idx(item_index)
            # Label every few points to avoid cluttering
            if i % max(1, len(actual_positions) // self.action_exec_size) == 0:
                labeled_points_y_3.append(pos[1])
                labeled_points_z_3.append(pos[2])
                ax3.annotate(f'{chunk_idx}', (pos[1], pos[2]), 
                            textcoords="offset points", xytext=(5, 5), 
                            fontsize=8, alpha=0.7)

        # Mark the labeled points
        ax3.scatter(labeled_points_y_3, labeled_points_z_3, c='darkgreen', s=50, marker='o', 
                    edgecolors='black', linewidths=1, zorder=4, label='Labeled chunks')

        ax3.set_xlabel('Y position')
        ax3.set_ylabel('Z position')
        ax3.set_title('Actual Robot Positions (with chunk indices)')
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # 4. Executed Actions with chunk index labels
        ax4 = axes[1, 1]
        executed_actions = self.data['executed_actions']['data'][actual_index_start:actual_index_end]
        ax4.plot(executed_actions[:, 1], executed_actions[:, 2], 'r-', linewidth=2, label='Executed actions')
        ax4.scatter(executed_actions[0, 1], executed_actions[0, 2], c='green', s=100, marker='o', label='Start', zorder=5)
        ax4.scatter(executed_actions[-1, 1], executed_actions[-1, 2], c='red', s=100, marker='x', label='End', zorder=5)

        # Add chunk index labels
        labeled_points_y_4 = []
        labeled_points_z_4 = []
        for i, action in enumerate(executed_actions):
            item_index = actual_index_start + i
            chunk_idx = self._item_idx_to_chunk_idx(item_index)
            # Label every few points to avoid cluttering
            if i % max(1, len(executed_actions) // self.action_exec_size) == 0:
                labeled_points_y_4.append(action[1])
                labeled_points_z_4.append(action[2])
                ax4.annotate(f'{chunk_idx}', (action[1], action[2]), 
                            textcoords="offset points", xytext=(5, 5), 
                            fontsize=8, alpha=0.7)

        # Mark the labeled points
        ax4.scatter(labeled_points_y_4, labeled_points_z_4, c='darkred', s=50, marker='o', 
                    edgecolors='black', linewidths=1, zorder=4, label='Labeled chunks')

        ax4.set_xlabel('Y position')
        ax4.set_ylabel('Z position')
        ax4.set_title('Executed Actions (with chunk indices)')
        ax4.legend()
        ax4.grid(True, alpha=0.3)

        # ax1.set_xlim([-0.01, 0.18])
        # ax2.set_xlim([-0.01, 0.18])
        # ax3.set_xlim([-0.01, 0.18])
        # ax4.set_xlim([-0.01, 0.18])

        # ax1.set_ylim([-0.01, 0.26])
        # ax2.set_ylim([-0.01, 0.26])
        # ax3.set_ylim([-0.01, 0.26])
        # ax4.set_ylim([-0.01, 0.26])


        plt.tight_layout()
        plt.show()

def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description='Interactive debug viewer for chunking control data'
    )
    parser.add_argument(
        '--data-dir',
        type=str,
        default='debug_data',
        help='Path to debug data directory (default: debug_data)'
    )

    args = parser.parse_args()

    print("="*70)
    print("Debug Chunking Control Viewer - Open3D GUI")
    print("="*70)
    print("\nInstructions:")
    print("  1. Use the slider to navigate through timeline events")
    print("  2. CHUNK events = Action chunk inferences (~1-2 Hz)")
    print("  3. EXEC events = Action executions (4 Hz)")
    print("  4. Red sphere in 3D view = EEF position")
    print("  5. Right panel shows event details and statistics")
    print("  6. Use mouse to rotate/zoom the 3D view")
    print("="*70)

    viewer = DebugChunkingViewer(debug_data_dir=args.data_dir)
    viewer.go()


if __name__ == '__main__':
    main()
