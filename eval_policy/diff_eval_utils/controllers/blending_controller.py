from diff_eval_utils.controllers.chunking_controller import ChunkingController
import time
import numpy as np
import threading
import matplotlib.pyplot as plt
from matplotlib import cm, ticker


class BlendingChunkingController(ChunkingController):
    """Controller with action blending in overlap regions."""

    def _setup_debug_plotting(self):
        """Initialize debug plotting with blending-specific data."""
        super()._setup_debug_plotting()
        self.action_queue_snapshots = []

    def _need_inference(self):
        """Check if we need to run policy inference (accounts for merge_range)."""
        return len(self.action_queue) <= self.config.policy_delay + self.config.merge_range

    def _run_inference_async(self):
        """Run inference with blending in background thread."""
        def inference_worker():
            try:
                current_time = time.time()

                if self.last_inference_time is not None:
                    time_since_last = current_time - self.last_inference_time
                    print(f"\n{'='*50}")
                    print(f"[Inference #{self.inference_count}] Time since last: {time_since_last*1000:.1f} ms")
                    print(f"{'='*50}")

                # Record observation arrival
                if self.config.visualize:
                    self.live_events.append((time.time(), 'obs_arrival'))

                # Get observation and predict
                obs_dict = self._get_observation()
                actions = self.policy_client.predict_action(obs_dict)
                inference_time = time.time()

                # Record inference done
                if self.config.visualize:
                    self.live_events.append((inference_time, 'inference_done'))

                # Store for debug plotting
                if self.config.debug_plotting:
                    self.all_predicted_chunks.append((self.inference_count, actions.copy()))

                # Thread-safe queue update with blending
                with self.inference_lock:
                    if self.first_inference_round:
                        # First time: add first action_exec_size actions
                        for action in actions[:self.config.action_exec_size]:
                            self.action_queue.append(action.copy())
                        self.first_inference_round = False
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
                            print(f"Saving RGBD timestamp: {ts_copy}")
                            self.debug_data_collection['pcd_timestamps'].append(
                                (timestamp_ms, ts_copy)
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

    def run(self, n_steps: int):
        """Execute blending chunking control with async inference."""
        n_steps_done = 0

        if self.config.debug_plotting:
            self.start_time = time.time()

        # Initialize start time for live visualization
        if self.config.visualize:
            self.start_time = time.time()

        # Initial inference (synchronous to populate queue)
        self._run_inference_async()
        while self.inference_in_progress:
            time.sleep(0.005)  # Wait for first inference

        self.first_inference_round = True
        print("\nStarting blending chunking control...")
        while n_steps_done < n_steps:
            loop_start = time.time()

            # Trigger async inference if needed (don't wait)
            if self._need_inference() and not self.inference_in_progress:
                self._run_inference_async()


            # Execute action from queue
            with self.inference_lock:
                queue_has_actions = len(self.action_queue) > 0
                if queue_has_actions:
                    action = self.action_queue.pop(0)

            if queue_has_actions:
                if self.config.debug_plotting:
                    current_pose = self.robot.end_effector_pose
                    rotmat = current_pose.orientation.as_matrix()
                    yaw = np.arctan2(rotmat[1, 0], rotmat[0, 0])
                    elapsed = time.time() - self.start_time
                    self.actual_poses.append((elapsed, current_pose.position[0],
                                             current_pose.position[1],
                                             current_pose.position[2], yaw))
                # Enforce fixed control frequency
                if self.enforce_fixed_freq:
                    self._enforce_fixed_frequency()
                self._execute_action(action)
                self.last_execution_time = time.time() # Update execution time

                # Record action execution
                if self.config.visualize:
                    self.live_events.append((time.time(), 'action_executed'))
                    self._update_live_plot()

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

        # Final plot update and close
        if self.live_plot_fig is not None:
            self._update_live_plot(force=True)  # Final update with all events
            plt.close(self.live_plot_fig)
            self.live_plot_fig = None
            self.live_plot_ax = None

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
                       linewidth=2, markersize=3, alpha=0.7,
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
