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
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering
from scipy.spatial.transform import Rotation as R


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

        # Combined timeline data
        self.timeline_events = []  # List of (timestamp_ms, event_type, index)

        # Load latest data
        self._load_latest_data()

        # Build combined timeline
        self._build_timeline()

        # Create GUI
        self._create_gui()

    def _load_latest_data(self):
        """Load the most recent debug data files."""
        print("Loading latest debug data...")

        # Find latest files in each subdirectory
        action_chunks_dir = self.debug_data_dir / 'action_chunks'
        actions_dir = self.debug_data_dir / 'actions'
        eef_poses_dir = self.debug_data_dir / 'eef_poses'
        pcd_dir = self.debug_data_dir / 'pcd'

        # Get latest files
        latest_action_chunks = self._get_latest_file(action_chunks_dir, 'action_chunks_*.npy')
        latest_actions = self._get_latest_file(actions_dir, 'executed_actions_*.npy')
        latest_eef_poses = self._get_latest_file(eef_poses_dir, 'eef_poses_*.npy')
        latest_pcd = self._get_latest_file(pcd_dir, 'point_clouds_*.npy')

        if not all([latest_action_chunks, latest_actions, latest_eef_poses, latest_pcd]):
            print("ERROR: Could not find all required data files!")
            print(f"  Action chunks: {latest_action_chunks}")
            print(f"  Executed actions: {latest_actions}")
            print(f"  EEF poses: {latest_eef_poses}")
            print(f"  Point clouds: {latest_pcd}")
            raise FileNotFoundError("Missing debug data files")

        # Load data
        print(f"\nLoading files:")
        print(f"  Action chunks: {latest_action_chunks.name}")
        print(f"  Executed actions: {latest_actions.name}")
        print(f"  EEF poses: {latest_eef_poses.name}")
        print(f"  Point clouds: {latest_pcd.name}")

        action_chunks_data = np.load(latest_action_chunks, allow_pickle=True).item()
        actions_data = np.load(latest_actions, allow_pickle=True).item()
        eef_poses_data = np.load(latest_eef_poses, allow_pickle=True).item()
        pcd_data = np.load(latest_pcd, allow_pickle=True).item()

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
            }
        }

        print(f"\nData loaded successfully:")
        print(f"  Action chunks: {len(self.data['action_chunks']['timestamps_ms'])} samples")
        print(f"  Executed actions: {len(self.data['executed_actions']['timestamps_ms'])} samples")
        print(f"  EEF poses: {len(self.data['eef_poses']['timestamps_ms'])} samples")
        print(f"  Point clouds: {len(self.data['point_clouds']['timestamps_ms'])} samples")

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

    def _build_timeline(self):
        """Build combined timeline from action chunks and executed actions."""
        # Combine all events into one sorted timeline
        chunk_ts = self.data['action_chunks']['timestamps_ms']
        exec_ts = self.data['executed_actions']['timestamps_ms']

        # Create list of (timestamp_ms, event_type, original_index)
        events = []
        for i, ts in enumerate(chunk_ts):
            events.append((ts, 'chunk', i))
        for i, ts in enumerate(exec_ts):
            events.append((ts, 'exec', i))

        # Sort by timestamp
        events.sort(key=lambda x: x[0])
        self.timeline_events = events

        # Calculate time range
        self.start_time_ms = events[0][0]
        self.end_time_ms = events[-1][0]
        self.duration_s = (self.end_time_ms - self.start_time_ms) / 1000.0

        print(f"\nTimeline built:")
        print(f"  Total events: {len(events)}")
        print(f"  Duration: {self.duration_s:.2f} s")
        print(f"  Chunk events: {len(chunk_ts)}")
        print(f"  Exec events: {len(exec_ts)}")

    def _create_gui(self):
        """Create the Open3D GUI window."""
        # Create application and window
        app = gui.Application.instance
        app.initialize()

        self.window = app.create_window("Debug Chunking Control Viewer", 1920, 1080)

        # Create main layout
        self._setup_layout()

        # Initialize 3D scene
        self._setup_scene()

        # Update to show first event
        self._update_visualization(0)

    def _setup_layout(self):
        """Setup the window layout with scene, slider, and info panel."""
        # Create main vertical layout
        main_layout = gui.Vert(0, gui.Margins(10, 10, 10, 10))

        # Create horizontal layout for 3D view and info panel
        h_layout = gui.Horiz(spacing=10)

        # Create 3D scene widget (left side, takes most space)
        self.scene_widget = gui.SceneWidget()
        self.scene_widget.scene = rendering.Open3DScene(self.window.renderer)
        h_layout.add_child(self.scene_widget)

        # Create info panel (right side, fixed width)
        info_panel = gui.Vert(0, gui.Margins(10, 10, 10, 10))

        # Title
        title_label = gui.Label("=== Event Information ===")
        info_panel.add_child(title_label)
        info_panel.add_fixed(10)

        # Info text (scrollable)
        self.info_text = gui.Label("")
        info_panel.add_child(self.info_text)

        # Statistics panel
        info_panel.add_fixed(20)
        stats_label = gui.Label("=== Statistics ===")
        info_panel.add_child(stats_label)
        info_panel.add_fixed(10)

        chunk_count = len(self.data['action_chunks']['timestamps_ms'])
        exec_count = len(self.data['executed_actions']['timestamps_ms'])
        avg_freq = exec_count / self.duration_s if self.duration_s > 0 else 0

        stats_text = (
            f"Total Events: {len(self.timeline_events)}\n"
            f"Duration: {self.duration_s:.2f} s\n"
            f"Chunk Inferences: {chunk_count}\n"
            f"Action Executions: {exec_count}\n"
            f"Avg Exec Freq: {avg_freq:.2f} Hz"
        )
        self.stats_text = gui.Label(stats_text)
        info_panel.add_child(self.stats_text)

        # Add info panel to horizontal layout
        h_layout.add_child(info_panel)

        main_layout.add_child(h_layout)

        # Create timeline slider at bottom
        slider_layout = gui.Vert(0, gui.Margins(10, 5, 10, 5))

        slider_label = gui.Label("Timeline (drag to navigate events)")
        slider_layout.add_child(slider_label)

        self.timeline_slider = gui.Slider(gui.Slider.INT)
        self.timeline_slider.set_limits(0, len(self.timeline_events) - 1)
        self.timeline_slider.int_value = 0
        self.timeline_slider.set_on_value_changed(self._on_slider_changed)
        slider_layout.add_child(self.timeline_slider)

        # Current event type label
        self.event_type_label = gui.Label("")
        slider_layout.add_child(self.event_type_label)

        main_layout.add_child(slider_layout)

        # Set the layout
        self.window.add_child(main_layout)

    def _setup_scene(self):
        """Setup the 3D scene with initial geometries."""
        # Setup camera
        bounds = o3d.geometry.AxisAlignedBoundingBox([-1, -1, -1], [1, 1, 1])
        self.scene_widget.setup_camera(60, bounds, [0, 0, 0])
        self.scene_widget.scene.set_background([0.1, 0.1, 0.1, 1.0])

        # Create and add coordinate frame
        self.coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=0.1, origin=[0, 0, 0]
        )
        mat = rendering.MaterialRecord()
        mat.shader = "defaultUnlit"
        self.scene_widget.scene.add_geometry("coord_frame", self.coord_frame, mat)

        # Create placeholder point cloud (will be populated on first update)
        self.pcd_geometry = o3d.geometry.PointCloud()
        # Don't add empty point cloud to scene - will be added in first update

        # Create placeholder EEF marker (will be populated on first update)
        self.eef_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.02)
        # Don't add empty marker to scene - will be added in first update

    def _on_slider_changed(self, value):
        """Handle slider value changes.

        Args:
            value: New slider value (event index)
        """
        index = int(value)
        self._update_visualization(index)

    def _update_visualization(self, event_index):
        """Update the 3D visualization for the selected event.

        Args:
            event_index: Index into timeline_events list
        """
        if event_index < 0 or event_index >= len(self.timeline_events):
            return

        self.current_index = event_index
        timestamp_ms, event_type, original_idx = self.timeline_events[event_index]

        # Update event type label
        time_from_start = (timestamp_ms - self.start_time_ms) / 1000.0
        event_label = f"Event {event_index + 1}/{len(self.timeline_events)} - " \
                     f"{'CHUNK' if event_type == 'chunk' else 'EXEC'} @ {time_from_start:.3f}s"
        self.event_type_label.text = event_label

        # Find nearest point cloud
        pcd_idx = self._find_nearest_timestamp_index(
            self.data['point_clouds']['timestamps_ms'],
            timestamp_ms
        )

        if pcd_idx is not None:
            # Update point cloud
            pcd_array = self.data['point_clouds']['point_clouds'][pcd_idx]
            points = pcd_array[:, :3]
            colors = pcd_array[:, 3:6]

            self.pcd_geometry.points = o3d.utility.Vector3dVector(points)
            self.pcd_geometry.colors = o3d.utility.Vector3dVector(colors)

            # Remove and re-add geometry to update it
            if self.scene_widget.scene.has_geometry("point_cloud"):
                self.scene_widget.scene.remove_geometry("point_cloud")
            mat_pcd = rendering.MaterialRecord()
            mat_pcd.shader = "defaultUnlit"
            mat_pcd.point_size = 2.0
            self.scene_widget.scene.add_geometry("point_cloud", self.pcd_geometry, mat_pcd)

        # Update EEF marker position
        eef_idx = self._find_nearest_timestamp_index(
            self.data['eef_poses']['timestamps_ms'],
            timestamp_ms
        )

        if eef_idx is not None:
            eef_pos = self.data['eef_poses']['positions'][eef_idx]
            self.eef_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.02)
            self.eef_sphere.compute_vertex_normals()
            self.eef_sphere.translate(eef_pos)
            self.eef_sphere.paint_uniform_color([1.0, 0.0, 0.0])

            mat_eef = rendering.MaterialRecord()
            mat_eef.shader = "defaultUnlit"
            mat_eef.base_color = [1.0, 0.0, 0.0, 1.0]
            if self.scene_widget.scene.has_geometry("eef_marker"):
                self.scene_widget.scene.remove_geometry("eef_marker")
            self.scene_widget.scene.add_geometry("eef_marker", self.eef_sphere, mat_eef)

        # Update info panel
        self._update_info_panel(timestamp_ms, event_type, original_idx, eef_idx)

        # Force redraw
        self.window.post_redraw()

    def _update_info_panel(self, timestamp_ms, event_type, original_idx, eef_idx):
        """Update the information panel with event details.

        Args:
            timestamp_ms: Event timestamp in milliseconds
            event_type: 'chunk' or 'exec'
            original_idx: Index in original data array
            eef_idx: Index of nearest EEF pose
        """
        time_from_start = (timestamp_ms - self.start_time_ms) / 1000.0
        dt = datetime.fromtimestamp(timestamp_ms / 1000.0)

        info_lines = []
        info_lines.append(f"Event Type: {event_type.upper()}")
        info_lines.append(f"Time: {time_from_start:.3f} s")
        info_lines.append(f"Timestamp: {timestamp_ms} ms")
        info_lines.append(f"DateTime: {dt.strftime('%H:%M:%S.%f')[:-3]}")
        info_lines.append("")

        # EEF Pose
        if eef_idx is not None:
            pos = self.data['eef_poses']['positions'][eef_idx]
            quat = self.data['eef_poses']['orientations_quat'][eef_idx]

            r = R.from_quat(quat)
            euler = r.as_euler('xyz', degrees=True)

            info_lines.append("EEF Pose:")
            info_lines.append(f"  Pos: [{pos[0]:.4f}, {pos[1]:.4f}, {pos[2]:.4f}]")
            info_lines.append(f"  Quat: [{quat[0]:.3f}, {quat[1]:.3f},")
            info_lines.append(f"         {quat[2]:.3f}, {quat[3]:.3f}]")
            info_lines.append(f"  Euler: [{euler[0]:.1f}, {euler[1]:.1f}, {euler[2]:.1f}]°")
            info_lines.append("")

        # Action data
        if event_type == 'chunk':
            action_chunk = self.data['action_chunks']['data'][original_idx]
            info_lines.append("Action Chunk:")
            info_lines.append(f"  Shape: {action_chunk.shape}")
            info_lines.append(f"  First: {action_chunk[0][:3]}")
            info_lines.append(f"  Last: {action_chunk[-1][:3]}")
        else:
            action = self.data['executed_actions']['data'][original_idx]
            info_lines.append("Executed Action:")
            info_lines.append(f"  Pos: [{action[0]:.4f}, {action[1]:.4f}, {action[2]:.4f}]")
            info_lines.append(f"  Rot6D: {action[3:9]}")
            info_lines.append(f"  Gripper: {action[9]:.4f}")

        self.info_text.text = "\n".join(info_lines)

    def _find_nearest_timestamp_index(self, timestamps, target_timestamp):
        """Find index of nearest timestamp.

        Args:
            timestamps: Array of timestamps
            target_timestamp: Target timestamp to find

        Returns:
            Index of nearest timestamp or None
        """
        if len(timestamps) == 0:
            return None

        idx = np.argmin(np.abs(timestamps - target_timestamp))
        return idx

    def run(self):
        """Run the GUI application."""
        gui.Application.instance.run()


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
    viewer.run()


if __name__ == '__main__':
    main()
