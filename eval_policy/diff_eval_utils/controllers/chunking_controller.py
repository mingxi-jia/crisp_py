from diff_eval_utils.controllers.base_controller import RobotController
import time
import numpy as np
from pathlib import Path
import threading
import shutil
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib import cm, ticker


class ChunkingController(RobotController):
    """Controller with action buffering and policy_delay offset."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.action_queue = []
        self.last_inference_time = None
        self.last_execution_time = None
        self.inference_count = 0
        self.enforce_fixed_freq = True

        self.inference_thread = None
        self.inference_in_progress = False
        self.inference_lock = threading.Lock()

        self.execute_dt = 1.0 / self.ctrl_freq

        # Debug plotting setup
        if self.config.debug_plotting:
            self._setup_debug_plotting()

        # Debug data collection
        if self.config.save_debug_data:
            self.debug_data_collection = {
                'action_chunks': [],      # (timestamp_ms, array(16,10))
                'executed_actions': [],   # (timestamp_ms, array(10,))
                'eef_poses': [],          # (timestamp_ms, pose_dict)
                'point_clouds': [],       # (timestamp_ms, array(N,6))
                'robot0_eef_pos': [],     # (timestamp_ms, array(3,))
                'actual_eef_pos': [],     # (timestamp_ms, array(3,))
                'pcd_timestamps': []     # (timestamp_ms, pcd_timestamp)
            }
        else:
            self.debug_data_collection = None


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
        need_inf = len(self.action_queue) <= self.config.policy_delay
        if need_inf:
            print(f"\n[Inference Needed] Current action queue size: {len(self.action_queue)}")
        return need_inf

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
                self.action_queue.append((action.copy(), self.inference_count))
        else:
            # Subsequent: add actions starting from policy_delay
            start_idx = self.config.policy_delay + 4
            end_idx = start_idx + self.config.action_exec_size
            for action in actions[start_idx:end_idx]:
                self.action_queue.append((action.copy(), self.inference_count))

        self.last_inference_time = time.time()
        print(f"Inference took {(self.last_inference_time - current_time)*1000:.1f} ms")
        self.inference_count += 1

    def _run_inference_async(self):
        """Run inference in background thread."""
        def inference_worker():
            try:
                current_time = time.time()

                if self.last_inference_time is not None:
                    time_since_last = current_time - self.last_inference_time
                    print(f"\n{'='*50}")
                    print(f"[Inference #{self.inference_count}] Time since last: {time_since_last*1000:.1f} ms")
                    print(f"{'='*50}")

                # Get observation and predict
                obs_dict = self._get_observation()
                actions = self.policy_client.predict_action(obs_dict)
                inference_time = time.time()

                # Store for debug plotting
                if self.config.debug_plotting:
                    self.all_predicted_chunks.append((self.inference_count, actions.copy()))
                if len(self.action_queue) == 0:
                    print("no actions in queue")
                # Thread-safe queue update
                with self.inference_lock:
                    if self.first_inference_round:
                        for action in actions[:self.config.action_exec_size]:
                            self.action_queue.append((action.copy(), self.inference_count))
                        self.first_inference_round = False
                    else:
                        start_idx = self.config.policy_delay + 4
                        end_idx = start_idx + self.config.action_exec_size
                        for action in actions[start_idx:end_idx]:
                            self.action_queue.append((action.copy(), self.inference_count))

                    # Debug data collection
                    if self.debug_data_collection is not None:
                        timestamp_ms = int(inference_time * 1000)

                        # Save action chunk
                        self.debug_data_collection['action_chunks'].append(
                            (timestamp_ms, actions.copy())
                        )

                        # Save point cloud
                        pcd = obs_dict.get('pcd', None)
                        if pcd is not None:
                            self.debug_data_collection['point_clouds'].append(
                                (timestamp_ms, pcd.copy())
                            )

                        # Save robot0_eef_pos
                        robot0_eef_pos = obs_dict.get('robot0_eef_pos', None)
                        if robot0_eef_pos is not None:
                            self.debug_data_collection['robot0_eef_pos'].append(
                                (timestamp_ms, robot0_eef_pos.copy())
                            )

                        # Save RGBD timestamp
                        pcd_timestamp = obs_dict.get('pcd_timestamp', None)
                        if pcd_timestamp is not None:
                            # Ensure we create an independent copy (not a reference)
                            if isinstance(pcd_timestamp, np.ndarray):
                                ts_copy = pcd_timestamp.copy()
                            else:
                                ts_copy = np.array(pcd_timestamp).copy()
                            print(f"Saving RGBD timestamp: {ts_copy[0]}")
                            self.debug_data_collection['pcd_timestamps'].append(
                                (timestamp_ms, ts_copy[0])
                            )

                    self.last_inference_time = time.time()
                    print(f"Inference took {(self.last_inference_time - current_time)*1000:.1f} ms")
                    self.inference_count += 1

            finally:
                self.inference_in_progress = False

        # Start thread
        self.inference_in_progress = True
        self.inference_thread = threading.Thread(target=inference_worker, daemon=True)
        self.inference_thread.start()

    def _enforce_fixed_frequency(self):
        curr_time = time.time()
        if self.last_execution_time is not None:
            timing_adjust_constant = 0.001
            time_since_last_exec = curr_time - self.last_execution_time
            time_since_last_exec += timing_adjust_constant
            if time_since_last_exec < self.execute_dt:
                time_to_wait = self.execute_dt - time_since_last_exec
                time.sleep(time_to_wait - timing_adjust_constant)

    def run(self, n_steps: int):
        """Execute chunking control with async inference."""
        n_steps_done = 0

        if self.config.debug_plotting:
            self.start_time = time.time()

        # Initial inference (synchronous to populate queue)
        self._run_inference_async()
        while self.inference_in_progress:
            time.sleep(0.002)  # Wait for first inference

        print("\nStarting chunking control...")
        self.first_inference_round = True
        while n_steps_done < n_steps:
            loop_start = time.time()

            # Trigger async inference if needed (don't wait)
            if self._need_inference() and not self.inference_in_progress:
                self._run_inference_async()

            # Execute action from queue
            with self.inference_lock:
                queue_has_actions = len(self.action_queue) > 0
                if queue_has_actions:
                    action, inference_count = self.action_queue.pop(0)
    
            if queue_has_actions:
                # Enforce fixed control frequency
                if self.enforce_fixed_freq:
                    self._enforce_fixed_frequency()
                if inference_count == 0:
                    time.sleep(0.05)
                print("Executing action from inference #{}".format(inference_count))
                self._execute_action(action)
                self.last_execution_time = time.time() # Update execution time

                if self.config.debug_plotting:
                    current_pose = self.robot.end_effector_pose
                    rotmat = current_pose.orientation.as_matrix()
                    yaw = np.arctan2(rotmat[1, 0], rotmat[0, 0])
                    elapsed = time.time() - self.start_time
                    self.actual_poses.append((elapsed, current_pose.position[0],
                                             current_pose.position[1],
                                             current_pose.position[2], yaw))
                n_steps_done += 1
            else:
                print("Warning: Action queue empty, waiting for inference...")
                time.sleep(0.01)

            if self.config.debug_plotting:
                loop_time = (time.time() - loop_start) * 1000
                self.control_loop_times.append(loop_time)

        # Wait for any ongoing inference to complete
        if self.inference_thread is not None:
            self.inference_thread.join(timeout=5.0)

        # Save debug data at end
        if self.debug_data_collection is not None and len(self.debug_data_collection['executed_actions']) > 0:
            self._save_debug_data()

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
                       linewidth=1, markersize=3, alpha=0.7,
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

    def _save_debug_data(self):
        """Save all collected debug data at end of execution."""
        if self.debug_data_collection is None:
            return

        try:
            # Create directory structure
            debug_dir = Path('debug_data')
            subdirs = {
                'action_chunks': debug_dir / 'action_chunks',
                'executed_actions': debug_dir / 'actions',
                'eef_poses': debug_dir / 'eef_poses',
                'point_clouds': debug_dir / 'pcd',
                'robot0_eef_pos': debug_dir / 'robot0_eef_pos',
                'actual_eef_pos': debug_dir / 'actual_eef_pos',
                'pcd_timestamps': debug_dir / 'pcd_timestamps'
            }

            for subdir in subdirs.values():
                subdir.mkdir(parents=True, exist_ok=True)

            # Session timestamp for file naming
            session_ts = datetime.now().strftime('%Y%m%d_%H%M%S')

            # Save each data type
            if len(self.debug_data_collection['action_chunks']) > 0:
                self._save_timestamped_array_data(
                    self.debug_data_collection['action_chunks'],
                    subdirs['action_chunks'] / f'action_chunks_{session_ts}.npy'
                )

            if len(self.debug_data_collection['executed_actions']) > 0:
                self._save_timestamped_array_data(
                    self.debug_data_collection['executed_actions'],
                    subdirs['executed_actions'] / f'executed_actions_{session_ts}.npy'
                )

            if len(self.debug_data_collection['eef_poses']) > 0:
                self._save_eef_poses(
                    self.debug_data_collection['eef_poses'],
                    subdirs['eef_poses'] / f'eef_poses_{session_ts}.npy'
                )

            if len(self.debug_data_collection['point_clouds']) > 0:
                self._save_point_clouds(
                    self.debug_data_collection['point_clouds'],
                    subdirs['point_clouds'],
                    session_ts
                )

            if len(self.debug_data_collection['robot0_eef_pos']) > 0:
                self._save_timestamped_array_data(
                    self.debug_data_collection['robot0_eef_pos'],
                    subdirs['robot0_eef_pos'] / f'robot0_eef_pos_{session_ts}.npy'
                )

            if len(self.debug_data_collection['actual_eef_pos']) > 0:
                self._save_timestamped_array_data(
                    self.debug_data_collection['actual_eef_pos'],
                    subdirs['actual_eef_pos'] / f'actual_eef_pos_{session_ts}.npy'
                )

            if len(self.debug_data_collection['pcd_timestamps']) > 0:
                self._save_timestamped_array_data(
                    self.debug_data_collection['pcd_timestamps'],
                    subdirs['pcd_timestamps'] / f'pcd_timestamps_{session_ts}.npy'
                )

            # Summary
            print(f"\n{'='*60}")
            print(f"Debug data saved: {debug_dir.absolute()}")
            print(f"  Session: {session_ts}")
            print(f"  Action chunks: {len(self.debug_data_collection['action_chunks'])} samples")
            print(f"  Executed actions: {len(self.debug_data_collection['executed_actions'])} samples")
            print(f"  EEF poses: {len(self.debug_data_collection['eef_poses'])} samples")
            print(f"  Point clouds: {len(self.debug_data_collection['point_clouds'])} samples")
            print(f"  Robot0 EEF pos: {len(self.debug_data_collection['robot0_eef_pos'])} samples")
            print(f"  Actual EEF pos: {len(self.debug_data_collection['actual_eef_pos'])} samples")
            print(f"  RGBD timestamps: {len(self.debug_data_collection['pcd_timestamps'])} samples")
            print(f"{'='*60}\n")

        except Exception as e:
            print(f"WARNING: Failed to save debug data: {e}")
            import traceback
            traceback.print_exc()

    def _save_timestamped_array_data(self, data_list, output_path):
        """Save list of (timestamp_ms, array) tuples."""
        if len(data_list) == 0:
            return

        timestamps = np.array([ts for ts, _ in data_list], dtype=np.int64)
        arrays = np.array([arr for _, arr in data_list])

        np.save(output_path, {
            'timestamps_ms': timestamps,
            'data': arrays
        })

    def _save_eef_poses(self, poses_list, output_path):
        """Save end-effector poses with timestamps."""
        if len(poses_list) == 0:
            return

        timestamps = np.array([ts for ts, _ in poses_list], dtype=np.int64)
        positions = np.array([pose['position'] for _, pose in poses_list], dtype=np.float32)
        quats = np.array([pose['orientation_quat'] for _, pose in poses_list], dtype=np.float32)
        matrices = np.array([pose['orientation_matrix'] for _, pose in poses_list], dtype=np.float32)

        np.save(output_path, {
            'timestamps_ms': timestamps,
            'positions': positions,
            'orientations_quat': quats,
            'orientations_matrix': matrices
        })

    def _save_point_clouds(self, pcd_list, output_dir, session_timestamp):
        """Save point clouds with timestamps.

        Saves all PCDs in single .npy file for easy batch loading.
        """
        if self.config.save_pcd_format == 'npy':
            # Single file with all point clouds
            timestamps = np.array([ts for ts, _ in pcd_list], dtype=np.int64)
            point_clouds = [pcd for _, pcd in pcd_list]
            point_counts = np.array([len(pcd) for pcd in point_clouds], dtype=np.int32)

            output_path = output_dir / f'point_clouds_{session_timestamp}.npy'
            np.save(output_path, {
                'timestamps_ms': timestamps,
                'point_clouds': point_clouds,
                'point_counts': point_counts
            }, allow_pickle=True)

        elif self.config.save_pcd_format == 'ply':
            # Individual PLY files
            import open3d as o3d

            timestamps = []
            filenames = []

            for timestamp_ms, pcd_array in pcd_list:
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(pcd_array[:, :3])
                pcd.colors = o3d.utility.Vector3dVector(pcd_array[:, 3:6])

                filename = f'pcd_{session_timestamp}_{timestamp_ms}.ply'
                o3d.io.write_point_cloud(str(output_dir / filename), pcd)

                timestamps.append(timestamp_ms)
                filenames.append(filename)

            # Save metadata
            np.save(output_dir / f'metadata_{session_timestamp}.npy', {
                'timestamps_ms': np.array(timestamps, dtype=np.int64),
                'filenames': filenames
            })
