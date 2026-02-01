from diff_eval_utils.controllers.teleop_controller import TeleopController
import time
from typing import Optional
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


class TestTeleopController(TeleopController):
    """Test controller for evaluating teleop tracking accuracy.

    Inherits from TeleopController and adds functionality to record and plot
    the difference between commanded poses and actual robot poses.

    Key Controls:
    - 'r': Toggle recording on/off (generates plots when stopped)
    - 's': Reset robot to home position
    - Spacemouse: Control robot movement
    - Spacemouse button: Toggle gripper
    - Ctrl+C: Exit (generates plots if recording)
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Recording state
        self.is_recording = False

        # Storage for tracking data
        self.tracking_data = {
            'timestamps': [],
            'current_pos': [],
            'target_pos': [],
            'current_rpy': [],
            'target_rpy': [],
        }

    def _setup_keyboard_listener(self):
        """Initialize keyboard listener for 'r' (record) and 's' (reset) keys."""
        from pynput import keyboard

        # Add 's' key for reset (since 'r' is now for recording)
        self.key_commands['s'] = False

        def on_press(key):
            with self.listener_lock:
                try:
                    if hasattr(key, 'char'):
                        if key.char == 'r':
                            self.key_commands['r'] = True
                        elif key.char == 's':
                            self.key_commands['s'] = True
                            print("\n[KEY: s] Reset requested...")
                except AttributeError:
                    pass

        self.keyboard_listener = keyboard.Listener(on_press=on_press)
        self.keyboard_listener.start()
        print("Keyboard listener started:")
        print("  'r' - Toggle recording on/off")
        print("  's' - Reset robot to home position")

    def _handle_record_toggle(self):
        """Handle 'r' key press: toggle recording on/off."""
        if not self.is_recording:
            # Start recording
            self._clear_tracking_data()
            self.is_recording = True
            print(f"\n{'='*60}")
            print("[RECORDING] Started recording tracking data...")
            print(f"{'='*60}\n")
        else:
            # Stop recording and generate plots
            self.is_recording = False
            print(f"\n{'='*60}")
            print("[RECORDING] Stopped recording. Generating plots...")
            print(f"{'='*60}")
            self._plot_tracking_error()
            self._plot_realtime_tracking()
            print(f"{'='*60}")
            print("[RECORDING] Ready. Press 'r' to start a new recording.")
            print(f"{'='*60}\n")

    def _clear_tracking_data(self):
        """Clear all tracking data."""
        self.tracking_data = {
            'timestamps': [],
            'current_pos': [],
            'target_pos': [],
            'current_rpy': [],
            'target_rpy': [],
        }

    def _record_tracking_data(self, target_pos, gripper_pose):
        """Record current and target positions for tracking analysis.

        Only records when is_recording is True.
        Records data in gripper frame (after fingertip-to-gripper conversion).

        Args:
            target_pos: Target position in gripper frame (x, y, z)
            gripper_pose: Target orientation as scipy Rotation object in gripper frame
        """
        if not self.is_recording:
            return

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

    def run(self, n_steps: Optional[int] = None):
        """Execute teleoperation control with recording capability.

        Args:
            n_steps: Not used (runs indefinitely until Ctrl+C)
        """
        _ = n_steps  # Unused, teleop runs indefinitely
        from crisp_py.spacemouse import Spacemouse

        # Clear tracking data
        self._clear_tracking_data()

        # Setup
        self._setup_keyboard_listener()
        self._setup_teleop_publisher()

        print("\n" + "="*60)
        print("TEST TELEOPERATION MODE ACTIVE")
        print("="*60)
        print("Controls:")
        print("  'r' - Toggle recording on/off (plots generated when stopped)")
        print("  's' - Reset robot to home position")
        print("  Spacemouse - Control robot movement")
        print("  Spacemouse button - Toggle gripper")
        print("  Ctrl+C - Exit (generates plots if recording)")
        print("="*60)
        print("\nRecording is OFF. Press 'r' to start recording.\n")

        try:
            from diff_eval_utils.diffusion_transforms import get_pose_from_robot
            with Spacemouse(deadzone=self.DEADZONE) as sm:
                self.spacemouse = sm
                self.teleop_target_pose = get_pose_from_robot(self.robot.end_effector_pose.copy(), ret_pose=True)

                while True:
                    # Check for 'r' key (toggle recording)
                    if self._check_key_command('r'):
                        self._handle_record_toggle()

                    # Check for 's' key (reset)
                    if self._check_key_command('s'):
                        self._handle_reset()

                    # Execute spacemouse control
                    self._execute_teleop_step(sm)

        except KeyboardInterrupt:
            print("\n\nTeleoperation stopped by user")
        finally:
            self._cleanup()
            # Generate visualizations if we were recording
            if self.is_recording or len(self.tracking_data['timestamps']) > 0:
                print("\nGenerating final plots...")
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
        _, axes = plt.subplots(2, 3, figsize=(15, 8))

        pos_labels = ['X', 'Y', 'Z']
        rpy_labels = ['Roll', 'Pitch', 'Yaw']

        # Plot position tracking (top row)
        for i, label in enumerate(pos_labels):
            ax = axes[0, i]
            ax.plot(step_indices, actual_pos[:, i], 'b-', label='Actual', linewidth=1.5)
            ax.plot(step_indices, target_pos_aligned[:, i], 'r--', label='Target', linewidth=1.5)
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

        title = f'{self.control_space.capitalize()} Teleop Controller Tracking Performance'
        plt.suptitle(title, fontsize=14, fontweight='bold')
        plt.tight_layout()

        # Save plot
        output_dir = Path('debug_plots')
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f'{self.control_space}_teleop_tracking.png'
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
        _, axes = plt.subplots(2, 3, figsize=(15, 8))

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

        title = f'{self.control_space.capitalize()} Teleop Controller - Real-time Tracking'
        plt.suptitle(title, fontsize=14, fontweight='bold')
        plt.tight_layout()

        # Save plot
        output_dir = Path('debug_plots')
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f'{self.control_space}_teleop_realtime_tracking.png'
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Realtime tracking plot saved to: {output_path.absolute()}")

        plt.close()

    def _print_tracking_stats(self, actual_pos, target_pos, actual_rpy, target_rpy):
        """Print summary statistics for tracking error."""
        pos_error = actual_pos - target_pos
        rpy_error = actual_rpy - target_rpy

        print("\n" + "="*60)
        print(f"TELEOP TRACKING ERROR STATISTICS ({self.control_space.upper()})")
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
