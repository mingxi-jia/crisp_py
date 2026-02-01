from diff_eval_utils.controllers.simple_controller import SimpleSequentialController
import time
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


class TestController(SimpleSequentialController):
    """Test controller for evaluating impedance controller tracking accuracy.

    Supports both joint and cartesian control spaces.
    Records current and target positions at each frame and visualizes
    tracking error over time for xyz and rpy.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Storage for tracking data
        self.tracking_data = {
            'timestamps': [],
            'current_pos': [],
            'target_pos': [],
            'current_rpy': [],
            'target_rpy': [],
        }

    def _record_tracking_data(self, target_pos, gripper_pose):
        """Record current and target positions for tracking analysis."""
        current_pos = self.robot.end_effector_pose.position.copy()
        current_rpy = self.robot.end_effector_pose.orientation.as_euler('XYZ')
        target_rpy = gripper_pose.as_euler('XYZ')

        self.tracking_data['timestamps'].append(time.time())
        self.tracking_data['current_pos'].append(current_pos)
        self.tracking_data['target_pos'].append(target_pos[:3])
        self.tracking_data['current_rpy'].append(current_rpy)
        self.tracking_data['target_rpy'].append(target_rpy)

    def _execute_joint_action(self, action, gripper_pose):
        """Override to record tracking data before execution."""
        self._record_tracking_data(action, gripper_pose)
        super()._execute_joint_action(action, gripper_pose)

    def _execute_cartesian_action(self, action, gripper_pose, move_to=False):
        """Override to record tracking data before execution."""
        self._record_tracking_data(action, gripper_pose)
        super()._execute_cartesian_action(action, gripper_pose, move_to=move_to)

    def run(self, n_steps: int):
        """Execute control and generate visualization at the end."""
        # Clear tracking data
        self.tracking_data = {
            'timestamps': [],
            'current_pos': [],
            'target_pos': [],
            'current_rpy': [],
            'target_rpy': [],
        }

        # Run parent's control loop
        super().run(n_steps)

        # Generate visualizations
        self._plot_tracking_error()
        self._plot_realtime_tracking()

    def _plot_tracking_error(self):
        """Plot tracking error for xyz and rpy over step index.

        Uses current position at step i+1 as the actual outcome of target at step i,
        which gives a fair comparison of tracking performance.
        """
        if len(self.tracking_data['timestamps']) < 2:
            print("Not enough tracking data to plot (need at least 2 steps)")
            return

        # Convert to numpy arrays
        current_pos = np.array(self.tracking_data['current_pos'])
        target_pos = np.array(self.tracking_data['target_pos'])
        current_rpy = np.array(self.tracking_data['current_rpy'])
        target_rpy = np.array(self.tracking_data['target_rpy'])

        # Use current position at next step as the actual outcome
        # target[i] -> actual outcome is current[i+1]
        n_steps = len(target_pos) - 1
        step_indices = np.arange(n_steps)
        target_pos_aligned = target_pos[:-1]  # target at step i
        actual_pos = current_pos[1:]           # actual outcome at step i+1
        target_rpy_aligned = target_rpy[:-1]
        actual_rpy = current_rpy[1:]

        # Create figure with 2 rows: position and orientation
        fig, axes = plt.subplots(2, 3, figsize=(15, 8))

        pos_labels = ['X', 'Y', 'Z']
        rpy_labels = ['Roll', 'Pitch', 'Yaw']

        # Draw vertical lines at action chunk boundaries
        chunk_size = self.config.action_exec_size
        chunk_boundaries = np.arange(0, n_steps, chunk_size)

        # Plot position tracking (top row)
        for i, label in enumerate(pos_labels):
            ax = axes[0, i]
            ax.plot(step_indices, actual_pos[:, i], 'b-', label='Actual', linewidth=1.5)
            ax.plot(step_indices, target_pos_aligned[:, i], 'r--', label='Target', linewidth=1.5)
            # Add vertical lines at chunk boundaries
            for boundary in chunk_boundaries:
                ax.axvline(x=boundary, color='gray', linestyle=':', alpha=0.7)
            ax.set_xlabel('Step Index')
            ax.set_ylabel(f'{label} (m)')
            ax.set_title(f'{label} Position Tracking')
            ax.legend()
            ax.grid(True, alpha=0.3)

            # Add error subplot as twin axis
            ax_err = ax.twinx()
            error = (actual_pos[:, i] - target_pos_aligned[:, i]) * 1000  # Convert to mm
            ax_err.fill_between(step_indices, 0, error, alpha=0.2, color='green')
            ax_err.set_ylabel('Error (mm)', color='green')
            ax_err.tick_params(axis='y', labelcolor='green')

        # Plot orientation tracking (bottom row)
        for i, label in enumerate(rpy_labels):
            ax = axes[1, i]
            ax.plot(step_indices, np.rad2deg(actual_rpy[:, i]), 'b-', label='Actual', linewidth=1.5)
            ax.plot(step_indices, np.rad2deg(target_rpy_aligned[:, i]), 'r--', label='Target', linewidth=1.5)
            # Add vertical lines at chunk boundaries
            for boundary in chunk_boundaries:
                ax.axvline(x=boundary, color='gray', linestyle=':', alpha=0.7)
            ax.set_xlabel('Step Index')
            ax.set_ylabel(f'{label} (deg)')
            ax.set_title(f'{label} Orientation Tracking')
            ax.legend()
            ax.grid(True, alpha=0.3)

            # Add error subplot as twin axis
            ax_err = ax.twinx()
            error = np.rad2deg(actual_rpy[:, i] - target_rpy_aligned[:, i])
            ax_err.fill_between(step_indices, 0, error, alpha=0.2, color='green')
            ax_err.set_ylabel('Error (deg)', color='green')
            ax_err.tick_params(axis='y', labelcolor='green')

        title = f'{self.control_space.capitalize()} Impedance Controller Tracking Performance'
        plt.suptitle(title, fontsize=14, fontweight='bold')
        plt.tight_layout()

        # Save plot
        output_dir = Path('debug_plots')
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f'{self.control_space}_impedance_tracking.png'
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"\nTracking plot saved to: {output_path.absolute()}")

        # Print summary statistics
        self._print_tracking_stats(actual_pos, target_pos_aligned, actual_rpy, target_rpy_aligned)

        plt.close()

    def _plot_realtime_tracking(self):
        """Plot target vs current position/orientation over real time.

        Shows:
        - Target positions with stars at each command point
        - Current robot position as a continuous line
        """
        if len(self.tracking_data['timestamps']) < 2:
            print("Not enough tracking data for realtime plot")
            return

        # Convert to numpy arrays
        timestamps = np.array(self.tracking_data['timestamps'])
        timestamps = timestamps - timestamps[0]  # Relative time from start
        current_pos = np.array(self.tracking_data['current_pos'])
        target_pos = np.array(self.tracking_data['target_pos'])
        current_rpy = np.array(self.tracking_data['current_rpy'])
        target_rpy = np.array(self.tracking_data['target_rpy'])

        # Create figure with 2 rows: position and orientation
        fig, axes = plt.subplots(2, 3, figsize=(15, 8))

        pos_labels = ['X', 'Y', 'Z']
        rpy_labels = ['Roll', 'Pitch', 'Yaw']

        # Plot position tracking (top row)
        for i, label in enumerate(pos_labels):
            ax = axes[0, i]
            # Current position - continuous line, no markers
            ax.plot(timestamps, current_pos[:, i], 'b-', label='Current', linewidth=1.5)
            # Target position - line with stars at each command point
            ax.plot(timestamps, target_pos[:, i], 'r-', linewidth=1, alpha=0.5)
            ax.plot(timestamps, target_pos[:, i], 'r*', markersize=8, label='Target')
            ax.set_xlabel('Time (s)')
            ax.set_ylabel(f'{label} (m)')
            ax.set_title(f'{label} Position vs Time')
            ax.legend()
            ax.grid(True, alpha=0.3)

        # Plot orientation tracking (bottom row)
        for i, label in enumerate(rpy_labels):
            ax = axes[1, i]
            # Current orientation - continuous line, no markers
            ax.plot(timestamps, np.rad2deg(current_rpy[:, i]), 'b-', label='Current', linewidth=1.5)
            # Target orientation - line with stars at each command point
            ax.plot(timestamps, np.rad2deg(target_rpy[:, i]), 'r-', linewidth=1, alpha=0.5)
            ax.plot(timestamps, np.rad2deg(target_rpy[:, i]), 'r*', markersize=8, label='Target')
            ax.set_xlabel('Time (s)')
            ax.set_ylabel(f'{label} (deg)')
            ax.set_title(f'{label} Orientation vs Time')
            ax.legend()
            ax.grid(True, alpha=0.3)

        title = f'{self.control_space.capitalize()} Controller - Real-time Tracking'
        plt.suptitle(title, fontsize=14, fontweight='bold')
        plt.tight_layout()

        # Save plot
        output_dir = Path('debug_plots')
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f'{self.control_space}_realtime_tracking.png'
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Realtime tracking plot saved to: {output_path.absolute()}")

        plt.close()

    def _print_tracking_stats(self, actual_pos, target_pos, actual_rpy, target_rpy):
        """Print summary statistics for tracking error."""
        pos_error = actual_pos - target_pos
        rpy_error = actual_rpy - target_rpy

        print("\n" + "="*60)
        print(f"TRACKING ERROR STATISTICS ({self.control_space.upper()})")
        print("="*60)

        print("\nPosition Error (mm):")
        print(f"  X: mean={np.mean(pos_error[:, 0])*1000:.2f}, std={np.std(pos_error[:, 0])*1000:.2f}, max={np.max(np.abs(pos_error[:, 0]))*1000:.2f}")
        print(f"  Y: mean={np.mean(pos_error[:, 1])*1000:.2f}, std={np.std(pos_error[:, 1])*1000:.2f}, max={np.max(np.abs(pos_error[:, 1]))*1000:.2f}")
        print(f"  Z: mean={np.mean(pos_error[:, 2])*1000:.2f}, std={np.std(pos_error[:, 2])*1000:.2f}, max={np.max(np.abs(pos_error[:, 2]))*1000:.2f}")

        print("\nOrientation Error (deg):")
        print(f"  Roll:  mean={np.mean(np.rad2deg(rpy_error[:, 0])):.2f}, std={np.std(np.rad2deg(rpy_error[:, 0])):.2f}, max={np.max(np.abs(np.rad2deg(rpy_error[:, 0]))):.2f}")
        print(f"  Pitch: mean={np.mean(np.rad2deg(rpy_error[:, 1])):.2f}, std={np.std(np.rad2deg(rpy_error[:, 1])):.2f}, max={np.max(np.abs(np.rad2deg(rpy_error[:, 1]))):.2f}")
        print(f"  Yaw:   mean={np.mean(np.rad2deg(rpy_error[:, 2])):.2f}, std={np.std(np.rad2deg(rpy_error[:, 2])):.2f}, max={np.max(np.abs(np.rad2deg(rpy_error[:, 2]))):.2f}")

        print("="*60 + "\n")
