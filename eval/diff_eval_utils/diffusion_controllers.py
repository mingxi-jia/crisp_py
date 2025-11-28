"""Controller hierarchy for different diffusion policy execution modes."""

from abc import ABC, abstractmethod
import time
import numpy as np
from pathlib import Path
import shutil
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt
from matplotlib import cm, ticker

from .diffusion_transforms import (
    get_pose_from_robot,
    convert_action_from_fingertip_to_gripper,
    franka_obs_to_diff_obs
)
from .diffusion_constants import GRIPPER_NORM_CONST


class RobotController(ABC):
    """Abstract base class for robot controllers."""

    def __init__(self, robot, gripper, obs_manager, joint_state_subscriber,
                 policy_client, pcd_client, rotation_transformer,
                 config, ctrl_freq=10.0):
        """Initialize the controller.

        Args:
            robot: Robot instance
            gripper: Gripper instance
            obs_manager: PointCloudManager instance
            joint_state_subscriber: JointStateSubscriber instance
            policy_client: PolicyClient instance
            pcd_client: PcdProcessingClient instance
            rotation_transformer: RotationTransformer for 6D rotations
            config: DPEvalConfig instance
            ctrl_freq: Control frequency in Hz
        """
        self.robot = robot
        self.gripper = gripper
        self.obs_manager = obs_manager
        self.joint_state_subscriber = joint_state_subscriber
        self.policy_client = policy_client
        self.pcd_client = pcd_client
        self.rotation_transformer = rotation_transformer
        self.config = config
        self.ctrl_freq = ctrl_freq

        # Common state
        self.target_pose = robot.end_effector_pose.copy()
        self.arm_rate = robot.node.create_rate(ctrl_freq)
        self.gripper_rate = gripper.node.create_rate(ctrl_freq)
        self.prev_grasp_value = 0.0

    @abstractmethod
    def run(self, n_steps: int):
        """Execute the controller for n_steps.

        Args:
            n_steps: Number of control steps to execute
        """
        pass

    def _get_observation(self):
        """Get current observation from sensors."""
        joint_state = self.joint_state_subscriber.joint_values
        gripper_val = self.gripper.value
        joint_state = np.concatenate([joint_state, [GRIPPER_NORM_CONST * gripper_val]])
        gripper_state = not self.gripper.is_open()

        time.sleep(0.05)  # Wait for robot stability

        eef_pose = get_pose_from_robot(self.robot.end_effector_pose)

        obs_dict = franka_obs_to_diff_obs(
            self.obs_manager, eef_pose, gripper_state,
            self.pcd_client, joint_state
        )

        return obs_dict

    def _execute_action(self, action):
        """Execute a single action.

        Args:
            action: 10-element action array [x, y, z, rot6d(6), grasp(1)]
        """
        action = convert_action_from_fingertip_to_gripper(action, self.rotation_transformer)

        self.target_pose.position = action[:3]
        self.target_pose.orientation = R.from_euler('XYZ', [np.pi, 0, 0])

        self.robot.set_target(pose=self.target_pose)
        self.arm_rate.sleep()

        grasp_value = np.round(np.clip(action[-1], 0, 1))
        if grasp_value != self.prev_grasp_value:
            self.gripper.set_target(1 - grasp_value)
            self.gripper_rate.sleep()
            time.sleep(1.0)  # Wait for gripper (Franka driver limitation)
        self.prev_grasp_value = grasp_value


class SimpleSequentialController(RobotController):
    """Simple controller that executes actions sequentially without buffering."""

    def run(self, n_steps: int):
        """Execute simple sequential control.

        Each iteration:
        1. Get observation
        2. Query policy for actions
        3. Execute all actions sequentially
        """
        n_steps_done = 0

        print("Starting simple sequential control...")
        while n_steps_done < n_steps:
            iter_start = time.time()

            # Get observation and action
            obs_dict = self._get_observation()
            actions = self.policy_client.predict_action(obs_dict)

            # Execute first 8 actions
            for action in actions[:8]:
                if n_steps_done >= n_steps:
                    break
                self._execute_action(action.copy())
                n_steps_done += 1

            iter_total = time.time() - iter_start
            print(f"Step {n_steps_done}: {iter_total*1000:.1f} ms")


class ChunkingController(RobotController):
    """Controller with action buffering and policy_delay offset."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.action_queue = []
        self.last_inference_time = None
        self.inference_count = 0

        # Debug plotting setup
        if self.config.debug_plotting:
            self._setup_debug_plotting()

    def _setup_debug_plotting(self):
        """Initialize debug plotting data structures."""
        if self.config.debug_output_dir.exists():
            shutil.rmtree(self.config.debug_output_dir)
        self.config.debug_output_dir.mkdir(parents=True, exist_ok=True)

        self.predicted_actions = []
        self.actual_poses = []
        self.control_loop_times = []
        self.all_predicted_chunks = []
        self.start_time = None

    def _need_inference(self):
        """Check if we need to run policy inference."""
        return len(self.action_queue) <= self.config.policy_delay

    def _run_inference(self):
        """Run policy inference and add actions to queue."""
        current_time = time.time()

        if self.last_inference_time is not None:
            time_since_last = current_time - self.last_inference_time
            print(f"\n{'='*50}")
            print(f"[Inference #{self.inference_count}] Time since last: {time_since_last*1000:.1f} ms")
            print(f"{'='*50}")

        # Get observation and predict
        obs_dict = self._get_observation()
        actions = self.policy_client.predict_action(obs_dict)

        # Store for debug plotting
        if self.config.debug_plotting:
            self.all_predicted_chunks.append((self.inference_count, actions.copy()))

        # Add actions to queue
        if len(self.action_queue) == 0:
            # First time: add first action_exec_size actions
            for action in actions[:self.config.action_exec_size]:
                self.action_queue.append(action.copy())
        else:
            # Subsequent: add actions starting from policy_delay
            start_idx = self.config.policy_delay
            end_idx = start_idx + self.config.action_exec_size
            for action in actions[start_idx:end_idx]:
                self.action_queue.append(action.copy())

        self.last_inference_time = time.time()
        print(f"Inference took {(self.last_inference_time - current_time)*1000:.1f} ms")
        self.inference_count += 1

    def run(self, n_steps: int):
        """Execute chunking control with action buffering."""
        n_steps_done = 0

        if self.config.debug_plotting:
            self.start_time = time.time()

        # Initial inference
        self._run_inference()

        print("\nStarting chunking control...")
        while n_steps_done < n_steps:
            loop_start = time.time()

            # Check if inference needed
            if self._need_inference():
                self._run_inference()

            # Execute action from queue
            if len(self.action_queue) > 0:
                action = self.action_queue.pop(0)

                # Record for debug plotting
                if self.config.debug_plotting:
                    current_pose = self.robot.end_effector_pose
                    rotmat = current_pose.orientation.as_matrix()
                    yaw = np.arctan2(rotmat[1, 0], rotmat[0, 0])
                    elapsed = time.time() - self.start_time
                    self.actual_poses.append((elapsed, current_pose.position[0],
                                             current_pose.position[1],
                                             current_pose.position[2], yaw))

                self._execute_action(action)
                n_steps_done += 1
            else:
                print("Warning: Action queue empty, skipping execution")
                time.sleep(1.0 / self.config.ctrl_freq)

            if self.config.debug_plotting:
                loop_time = (time.time() - loop_start) * 1000
                self.control_loop_times.append(loop_time)

        # Generate debug plots
        if self.config.debug_plotting:
            self._plot_debug_results()
            self._plot_action_chunks()
            self._plot_action_execution_timing()
            print(f"\n{'='*60}")
            print(f"Debug plots saved to: {self.config.debug_output_dir.absolute()}")
            print(f"{'='*60}")

    def _plot_debug_results(self):
        """Generate debug visualization plots."""
        print("\nGenerating control loop timing histogram...")

        loop_times = np.array(self.control_loop_times)

        # Create histogram
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        ax.hist(loop_times, bins=30, color='steelblue', alpha=0.7, edgecolor='black')

        # Add statistics
        mean_time = np.mean(loop_times)
        median_time = np.median(loop_times)
        std_time = np.std(loop_times)
        min_time = np.min(loop_times)
        max_time = np.max(loop_times)

        # Add vertical lines
        ax.axvline(mean_time, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_time:.2f} ms')
        ax.axvline(median_time, color='green', linestyle='--', linewidth=2, label=f'Median: {median_time:.2f} ms')

        # Add text box with statistics
        stats_text = (f'Statistics:\n'
                     f'Mean: {mean_time:.2f} ms\n'
                     f'Median: {median_time:.2f} ms\n'
                     f'Std Dev: {std_time:.2f} ms\n'
                     f'Min: {min_time:.2f} ms\n'
                     f'Max: {max_time:.2f} ms\n'
                     f'Total loops: {len(loop_times)}')

        ax.text(0.98, 0.97, stats_text, transform=ax.transAxes,
               verticalalignment='top', horizontalalignment='right',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
               fontsize=10, family='monospace')

        ax.set_xlabel('Control Loop Execution Time (ms)', fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)
        ax.set_title('Control Loop Timing Distribution', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        ax.legend(loc='upper left', fontsize=10)

        plt.tight_layout()
        filename = self.config.debug_output_dir / "control_loop_timing.png"
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        print(f"Timing histogram saved to: {filename}")
        plt.close()

    def _plot_action_chunks(self):
        """Generate plots showing all predicted actions from each inference chunk."""
        print("Generating action chunk plots...")

        if len(self.all_predicted_chunks) == 0:
            print("No action chunks to plot")
            return

        n_inferences = len(self.all_predicted_chunks)
        colors = cm.rainbow(np.linspace(0, 1, n_inferences))

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('All Predicted Action Chunks (16 actions per inference)', fontsize=14, fontweight='bold')

        labels = ['X Position (m)', 'Y Position (m)', 'Z Position (m)', 'Yaw (rad)']

        for ax, label in zip(axes.flatten(), labels):
            for inf_idx, (chunk_id, actions) in enumerate(self.all_predicted_chunks):
                if inf_idx == 0:
                    start_step = 0
                else:
                    start_step = self.config.action_exec_size * inf_idx

                x_vals = []
                y_vals = []

                for action_idx in range(len(actions)):
                    action = actions[action_idx]
                    n_step = start_step + action_idx

                    if label == 'X Position (m)':
                        value = action[0]
                    elif label == 'Y Position (m)':
                        value = action[1]
                    elif label == 'Z Position (m)':
                        value = action[2]
                    else:  # Yaw
                        rot6d = action[3:9]
                        rotmat = self.rotation_transformer.forward(rot6d.reshape(1, 6))[0]
                        value = np.arctan2(rotmat[1, 0], rotmat[0, 0])

                    x_vals.append(n_step)
                    y_vals.append(value)

                ax.plot(x_vals, y_vals, 'o-', color=colors[inf_idx],
                       linewidth=2, markersize=6, alpha=0.7,
                       label=f'Inference #{chunk_id}')

            ax.set_xlabel('n_step', fontsize=11, fontweight='bold')
            ax.set_ylabel(label, fontsize=11, fontweight='bold')
            ax.xaxis.set_major_locator(ticker.MultipleLocator(4))
            ax.grid(which='major', color='black', linewidth=1)
            ax.grid(which='minor', color='gray', linestyle='--', alpha=0.3)
            ax.grid(True, alpha=0.3)
            ax.set_title(label, fontsize=12, fontweight='bold')
            ax.legend(loc='best', fontsize=8, ncol=2)

        plt.tight_layout()
        filename = self.config.debug_output_dir / "action_chunks.png"
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        print(f"Action chunks plot saved to: {filename}")
        plt.close()

    def _plot_action_execution_timing(self):
        """Generate plot showing execution time vs action index to identify delays."""
        print("Generating action execution timing plot...")

        if len(self.control_loop_times) == 0:
            print("No timing data to plot")
            return

        loop_times = np.array(self.control_loop_times)
        action_indices = np.arange(len(loop_times))

        # Calculate cumulative time for reference
        cumulative_time = np.cumsum(loop_times)

        # Calculate expected time at ideal frequency
        ideal_dt = 1000.0 / self.config.ctrl_freq  # ms per step
        expected_cumulative = action_indices * ideal_dt

        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
        fig.suptitle('Action Execution Timing Analysis', fontsize=14, fontweight='bold')

        # Subplot 1: Loop time per action
        ax1.plot(action_indices, loop_times, 'o-', color='steelblue',
                linewidth=1.5, markersize=4, alpha=0.7, label='Actual loop time')
        ax1.axhline(ideal_dt, color='red', linestyle='--', linewidth=2,
                   label=f'Expected (1/{self.config.ctrl_freq} Hz = {ideal_dt:.1f} ms)')

        # Highlight outliers (>1.5x expected)
        outliers = loop_times > (ideal_dt * 1.5)
        if np.any(outliers):
            ax1.scatter(action_indices[outliers], loop_times[outliers],
                       color='red', s=80, zorder=5, label='Delays (>1.5x expected)', marker='x')

        ax1.set_xlabel('Action Index', fontsize=11, fontweight='bold')
        ax1.set_ylabel('Loop Execution Time (ms)', fontsize=11, fontweight='bold')
        ax1.set_title('Control Loop Time per Action', fontsize=12, fontweight='bold')
        ax1.xaxis.set_major_locator(ticker.MultipleLocator(4))
        ax1.grid(which='major', color='black', linewidth=0.5, alpha=0.3)
        ax1.grid(which='minor', color='gray', linestyle='--', alpha=0.2)
        ax1.legend(loc='best', fontsize=9)

        # Add statistics box
        mean_time = np.mean(loop_times)
        std_time = np.std(loop_times)
        outlier_count = np.sum(outliers)
        stats_text = (f'Mean: {mean_time:.2f} ms\n'
                     f'Std: {std_time:.2f} ms\n'
                     f'Delays: {outlier_count}/{len(loop_times)}')
        ax1.text(0.02, 0.98, stats_text, transform=ax1.transAxes,
                verticalalignment='top', horizontalalignment='left',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
                fontsize=9, family='monospace')

        # Subplot 2: Cumulative time deviation
        time_deviation = cumulative_time - expected_cumulative
        ax2.plot(action_indices, time_deviation, 'o-', color='purple',
                linewidth=1.5, markersize=4, alpha=0.7)
        ax2.axhline(0, color='black', linestyle='-', linewidth=1, alpha=0.5)
        ax2.fill_between(action_indices, 0, time_deviation,
                        where=(time_deviation >= 0), color='red', alpha=0.2, label='Behind schedule')
        ax2.fill_between(action_indices, 0, time_deviation,
                        where=(time_deviation < 0), color='green', alpha=0.2, label='Ahead of schedule')

        ax2.set_xlabel('Action Index', fontsize=11, fontweight='bold')
        ax2.set_ylabel('Cumulative Time Deviation (ms)', fontsize=11, fontweight='bold')
        ax2.set_title('Cumulative Timing Deviation from Expected', fontsize=12, fontweight='bold')
        ax2.xaxis.set_major_locator(ticker.MultipleLocator(4))
        ax2.grid(which='major', color='black', linewidth=0.5, alpha=0.3)
        ax2.grid(which='minor', color='gray', linestyle='--', alpha=0.2)
        ax2.legend(loc='best', fontsize=9)

        # Add final deviation text
        final_deviation = time_deviation[-1]
        deviation_text = f'Final deviation: {final_deviation:.1f} ms'
        ax2.text(0.98, 0.98, deviation_text, transform=ax2.transAxes,
                verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8),
                fontsize=9, family='monospace')

        plt.tight_layout()
        filename = self.config.debug_output_dir / "action_execution_timing.png"
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        print(f"Action execution timing plot saved to: {filename}")
        plt.close()


class BlendingChunkingController(ChunkingController):
    """Controller with action blending in overlap regions."""

    def _setup_debug_plotting(self):
        """Initialize debug plotting with blending-specific data."""
        super()._setup_debug_plotting()
        self.action_queue_snapshots = []

    def _need_inference(self):
        """Check if we need to run policy inference (accounts for merge_range)."""
        return len(self.action_queue) <= self.config.policy_delay + self.config.merge_range

    def _run_inference(self):
        """Run policy inference and blend overlapping actions."""
        current_time = time.time()

        if self.last_inference_time is not None:
            time_since_last = current_time - self.last_inference_time
            print(f"\n{'='*50}")
            print(f"[Inference #{self.inference_count}] Time since last: {time_since_last*1000:.1f} ms")
            print(f"{'='*50}")

        # Get observation and predict
        obs_dict = self._get_observation()
        actions = self.policy_client.predict_action(obs_dict)

        # Store for debug plotting
        if self.config.debug_plotting:
            self.all_predicted_chunks.append((self.inference_count, actions.copy()))

        # Add/blend actions
        if len(self.action_queue) == 0:
            # First time: add first action_exec_size actions
            for action in actions[:self.config.action_exec_size]:
                self.action_queue.append(action.copy())
        else:
            # Merge overlapping actions
            start_idx = self.config.policy_delay
            end_idx = start_idx + self.config.action_exec_size

            for action_idx, action in enumerate(actions[start_idx:end_idx]):
                if action_idx < self.config.merge_range:
                    # Blend with existing action in queue
                    prior_weight = (self.config.merge_range - action_idx) / self.config.merge_range
                    prior_weight = prior_weight / 2  # Reduce blending strength

                    queue_idx = -(action_idx + 1)
                    action_merged = (prior_weight * self.action_queue[queue_idx] +
                                    (1 - prior_weight) * action.copy())
                    self.action_queue[queue_idx] = action_merged
                else:
                    # Append new action
                    self.action_queue.append(action.copy())

        # Store queue snapshot for debug plotting
        if self.config.debug_plotting:
            self.action_queue_snapshots.append(
                (self.inference_count, [a.copy() for a in self.action_queue])
            )

        self.last_inference_time = time.time()
        print(f"Inference took {(self.last_inference_time - current_time)*1000:.1f} ms")
        self.inference_count += 1

    def _plot_debug_results(self):
        """Generate debug plots showing action blending."""
        # Call parent for timing histogram
        super()._plot_debug_results()

        # Add merged queue visualization
        print("Generating merged action queue plots...")

        if len(self.action_queue_snapshots) == 0:
            print("No action queue snapshots to plot")
            return

        n_snapshots = len(self.action_queue_snapshots)
        colors = cm.rainbow(np.linspace(0, 1, n_snapshots))

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Merged Action Queue Snapshots (After Blending)', fontsize=14, fontweight='bold')

        labels = ['X Position (m)', 'Y Position (m)', 'Z Position (m)', 'Yaw (rad)']

        for ax, label in zip(axes.flatten(), labels):
            for snap_idx, (inf_idx, action_queue) in enumerate(self.action_queue_snapshots):
                if snap_idx == 0:
                    start_step = 0
                else:
                    start_step = (self.config.action_exec_size - self.config.merge_range) * snap_idx

                x_vals = []
                y_vals = []

                for action_idx, action in enumerate(action_queue):
                    n_step = start_step + action_idx

                    if label == 'X Position (m)':
                        value = action[0]
                    elif label == 'Y Position (m)':
                        value = action[1]
                    elif label == 'Z Position (m)':
                        value = action[2]
                    else:  # Yaw
                        rot6d = action[3:9]
                        rotmat = self.rotation_transformer.forward(rot6d.reshape(1, 6))[0]
                        value = np.arctan2(rotmat[1, 0], rotmat[0, 0])

                    x_vals.append(n_step)
                    y_vals.append(value)

                ax.plot(x_vals, y_vals, 'o-', color=colors[snap_idx],
                       linewidth=2, markersize=6, alpha=0.7,
                       label=f'After Inference #{inf_idx}')

            ax.set_xlabel('n_step', fontsize=11, fontweight='bold')
            ax.set_ylabel(label, fontsize=11, fontweight='bold')
            ax.xaxis.set_major_locator(ticker.MultipleLocator(4))
            ax.grid(which='major', color='black', linewidth=1)
            ax.grid(which='minor', color='gray', linestyle='--', alpha=0.3)
            ax.grid(True, alpha=0.3)
            ax.set_title(label, fontsize=12, fontweight='bold')
            ax.legend(loc='best', fontsize=8, ncol=2)

        plt.tight_layout()
        filename = self.config.debug_output_dir / "merged_action_queue.png"
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        print(f"Merged action queue plot saved to: {filename}")
        plt.close()


def create_controller(mode: str, *args, **kwargs) -> RobotController:
    """Factory function to create controller based on mode.

    Args:
        mode: Controller mode ('simple', 'chunking', 'blending')
        *args, **kwargs: Arguments passed to controller constructor

    Returns:
        RobotController instance
    """
    if mode == 'simple':
        return SimpleSequentialController(*args, **kwargs)
    elif mode == 'chunking':
        return ChunkingController(*args, **kwargs)
    elif mode == 'blending':
        return BlendingChunkingController(*args, **kwargs)
    else:
        raise ValueError(f"Unknown controller mode: {mode}")
