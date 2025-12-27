import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (needed for 3D projection)
from pathlib import Path
from scipy.signal import savgol_filter
from scipy.interpolate import splprep, splev


def visualize_3d_trajectory_positions(
    poses,
    title="3D Trajectory",
    save_path=None,
    frame_label_interval=5,
    show_points=True,
    show_frame_labels=True,
    line_alpha=0.75,
    point_alpha=0.8,
    interventions=None,
):
    """
    Visualize a single 3D trajectory using only XYZ positions (no action arrows),
    with the same fixed axis limits as your existing script.

    Args:
        poses: array-like of shape (N, 3) or (N, >=3). First 3 columns are XYZ.
               If you have a dict trajectory like {'positions': ..., ...},
               pass traj['positions'].
        title: plot title string.
        save_path: optional path to save the figure (e.g. "figures/traj.png").
        frame_label_interval: label every k frames (only if show_frame_labels=True).
        show_points: if True, scatter points along the line.
        show_frame_labels: if True, annotate frame indices along the trajectory.
        line_alpha: alpha for the trajectory line.
        point_alpha: alpha for the scatter points.

    Returns:
        (fig, ax) matplotlib figure and axis objects.
    """
    poses = np.asarray(poses)
    # count the number of instances where the norm between two consecutive points is less than 0.01
    diffs = np.linalg.norm(np.diff(poses[:, :3], axis=0), axis=1)
    num_small_moves = np.sum(diffs < 0.005)
    print(f"Number of small moves (<0.005m) between consecutive points: {num_small_moves} out of {len(diffs)}")
    if poses.ndim != 2 or poses.shape[1] < 3:
        raise ValueError(f"`poses` must have shape (N,3) or (N,>=3). Got {poses.shape}.")
    
    # Only keep the points where diff is >= 0.005
    # mask = np.ones(len(poses), dtype=bool)
    # mask[1:] = diffs >= 0.005
    # poses = poses[mask]

    positions = poses[:, :3]
    n = positions.shape[0]
    if n == 0:
        raise ValueError("Empty trajectory: `poses` has N=0.")

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    # Line
    ax.plot(
        positions[:, 0], positions[:, 1], positions[:, 2],
        alpha=line_alpha, linewidth=2.0
    )

    # Points
    if show_points:
        ax.scatter(
            positions[:, 0], positions[:, 1], positions[:, 2],
            s=30, alpha=point_alpha, marker="o"
        )

    # Frame labels
    if show_frame_labels and frame_label_interval > 0:
        for i in range(n):
            if i % frame_label_interval == 0:
                ax.text(
                    positions[i, 0], positions[i, 1], positions[i, 2],
                    f"{i}", fontsize=8, fontweight="bold"
                )

    # Mark intervention points with red crosses
    if interventions is not None:
        interventions = np.asarray(interventions)
        intervention_indices = np.where(interventions == 1)[0]
        if len(intervention_indices) > 0:
            # Filter to valid indices within the positions array
            valid_indices = intervention_indices[intervention_indices < len(positions)]
            if len(valid_indices) > 0:
                ax.scatter(
                    positions[valid_indices, 0],
                    positions[valid_indices, 1],
                    positions[valid_indices, 2],
                    c="red", s=100, marker="X", alpha=0.9, label="Intervention"
                )
                print(f"Marked {len(valid_indices)} intervention points with red crosses")

    # Mark end point (grasp point analog) with X marker
    ax.scatter(
        positions[-1, 0], positions[-1, 1], positions[-1, 2],
        c="black", s=120, marker="X", alpha=0.9, label="End Point"
    )

    ax.legend()

    # Same axis config + fixed limits as your existing visualization
    ax.set_xlabel("X Position", fontsize=10)
    ax.set_ylabel("Y Position", fontsize=10)
    ax.set_zlabel("Z Position", fontsize=10)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.grid(True, alpha=0.3)

    ax.set_xlim(0.4, 0.8)
    ax.set_ylim(-0.25, 0.25)
    ax.set_zlim(-0.05, 0.3)

    plt.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(str(save_path), dpi=300, bbox_inches="tight")

    plt.show()
    return fig, ax

def visualize_3d_trajectory_from_poses(
    poses,
    title="3D Trajectory",
    frame_interval=5,
    show_points=True,
    axis_limits=((0.4, 0.8), (-0.25, 0.25), (-0.05, 0.3)),
    ax=None,
):
    """
    Visualize a 3D trajectory from poses, using fixed XYZ axis limits.

    Input:
      poses: array-like, shape (T, >=3). Uses poses[:, :3] as XYZ.
            (Works for either EEF positions or full poses [x,y,z,...].)

    Output:
      ax: matplotlib 3D axis containing the plot.

    Side-effects:
      Creates a figure (if ax is None) and plots the trajectory.
    """
    poses = np.asarray(poses)
    if poses.ndim != 2 or poses.shape[1] < 3:
        raise ValueError(f"`poses` must have shape (T, >=3). Got {poses.shape}.")

    xyz = poses[:, :3]

    if ax is None:
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection="3d")

    # Line + points
    ax.plot(xyz[:, 0], xyz[:, 1], xyz[:, 2], alpha=0.8, linewidth=2)
    if show_points:
        ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], s=25, alpha=0.7, marker="o")

    # Frame labels
    for i in range(len(xyz)):
        if frame_interval is not None and frame_interval > 0 and (i % frame_interval == 0):
            ax.text(xyz[i, 0], xyz[i, 1], xyz[i, 2], f"{i}", fontsize=8, fontweight="bold")

    # Mark end point
    ax.scatter(xyz[-1, 0], xyz[-1, 1], xyz[-1, 2], s=90, alpha=0.9, marker="X")

    # Axis config (match your existing limits)
    (xlim, ylim, zlim) = axis_limits
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_zlim(*zlim)

    ax.set_xlabel("X Position")
    ax.set_ylabel("Y Position")
    ax.set_zlabel("Z Position")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)

    return ax

def smooth_xyz_trajectory(
    xyz,
    method="savgol",
    savgol_window=11,
    savgol_polyorder=3,
    spline_s=0.001,
    num_points=None,
):
    """
    Smooth a 3D trajectory.

    Input:
      xyz: array-like, shape (T, 3)

    Output:
      smoothed_xyz: np.ndarray, shape (T, 3) for savgol
                    or shape (num_points, 3) for spline (if num_points provided)

    Side-effects:
      None.
    """
    xyz = np.asarray(xyz)
    if xyz.ndim != 2 or xyz.shape[1] != 3:
        raise ValueError(f"`xyz` must have shape (T, 3). Got {xyz.shape}.")

    T = xyz.shape[0]
    if T < 3:
        return xyz.copy()

    if method == "savgol":
        # Window must be odd and <= T
        w = int(savgol_window)
        if w % 2 == 0:
            w += 1
        w = min(w, T if T % 2 == 1 else T - 1)
        if w < 3:
            return xyz.copy()

        p = int(savgol_polyorder)
        p = min(p, w - 1)

        sm = np.empty_like(xyz, dtype=float)
        for d in range(3):
            sm[:, d] = savgol_filter(xyz[:, d], window_length=w, polyorder=p, mode="interp")
        return sm

    if method == "spline":
        # Parametrize by cumulative arc-length for nicer behavior than raw timestep.
        diffs = np.diff(xyz, axis=0)
        seglen = np.linalg.norm(diffs, axis=1)
        t = np.concatenate([[0.0], np.cumsum(seglen)])
        if t[-1] == 0:
            return xyz.copy()
        t = t / t[-1]

        # splprep expects data as separate arrays; k<=5 and <= T-1
        k = min(3, T - 1)
        tck, _ = splprep([xyz[:, 0], xyz[:, 1], xyz[:, 2]], u=t, s=float(spline_s), k=k)

        if num_points is None:
            u_new = t
        else:
            u_new = np.linspace(0, 1, int(num_points))

        x_new, y_new, z_new = splev(u_new, tck)
        return np.stack([x_new, y_new, z_new], axis=1)

    raise ValueError("method must be 'savgol' or 'spline'.")


def compare_actual_vs_desired_trajectory(
    actual_poses,
    desired_actions,
    title="Actual vs Desired Trajectory",
    save_path=None,
    frame_label_interval=5,
    show_error_arrows=True,
    arrow_scale=1.0,
    axis_limits=((0.4, 0.8), (-0.25, 0.25), (-0.05, 0.3)),
    traj_only=False,
    interventions=None,
):
    """
    Compare actual trajectory (from poses) vs desired trajectory (from actions).

    Args:
        actual_poses: array-like of shape (N, >=3). Actual executed poses (XYZ positions).
        desired_actions: array-like of shape (M, >=3). Desired actions (first 3 are XYZ target positions).
        title: plot title string.
        save_path: optional path to save the figure.
        frame_label_interval: label every k frames.
        show_error_arrows: if True, draw arrows from actual to desired positions.
        arrow_scale: scale factor for error arrows (larger = longer arrows).
        axis_limits: tuple of ((xmin, xmax), (ymin, ymax), (zmin, zmax)).
        traj_only: if True, show only the 3D trajectory plot (no error analysis subplots).
        interventions: array-like of shape (N,). Binary array where 1 indicates intervention.
                      Points with intervention are excluded from error calculation and marked with crosses.

    Returns:
        (fig, ax, stats) where stats is a dict with error metrics.
    """
    actual_poses = np.asarray(actual_poses)
    desired_actions = np.asarray(desired_actions)

    if actual_poses.ndim != 2 or actual_poses.shape[1] < 3:
        raise ValueError(f"`actual_poses` must have shape (N,>=3). Got {actual_poses.shape}.")
    if desired_actions.ndim != 2 or desired_actions.shape[1] < 3:
        raise ValueError(f"`desired_actions` must have shape (M,>=3). Got {desired_actions.shape}.")

    actual_xyz = actual_poses[:, :3]
    desired_xyz = desired_actions[:, :3]

    # Process interventions and create proper action-state alignment
    # Action at index i results in state at index i+1
    # Skip states that are interventions (no corresponding policy action)
    if interventions is not None:
        interventions = np.asarray(interventions)

        # Build mapping: for each action, find the next non-intervention state
        action_list = []
        state_list = []
        original_action_indices = []  # Track original action index for labels

        action_idx = 0
        state_idx = 1  # Start at state 1 (temporal offset)

        while action_idx < len(desired_xyz) and state_idx < len(actual_xyz):
            # Skip intervention states
            while state_idx < len(actual_xyz) and state_idx < len(interventions) and interventions[state_idx] == 1:
                state_idx += 1

            if state_idx < len(actual_xyz):
                action_list.append(desired_xyz[action_idx])
                state_list.append(actual_xyz[state_idx])
                original_action_indices.append(action_idx)
                action_idx += 1
                state_idx += 1

        if len(action_list) == 0:
            print("Warning: No valid action-state pairs found (all states are interventions)")
            return None, None, {}

        actual_xyz = np.array(state_list)
        desired_xyz = np.array(action_list)
        original_action_indices = np.array(original_action_indices)
        n_min = len(actual_xyz)

        # No interventions in the filtered data
        non_intervention_mask = np.ones(n_min, dtype=bool)
        intervention_indices = np.array([], dtype=int)
        n_interventions = np.sum(interventions == 1)
    else:
        # No interventions - use simple temporal offset
        n_actual = len(actual_xyz)
        n_desired = len(desired_xyz)
        n_min = min(n_actual - 1, n_desired)

        actual_xyz = actual_xyz[1:n_min+1]  # States at i+1
        desired_xyz = desired_xyz[:n_min]   # Actions at i
        original_action_indices = np.arange(n_min)

        non_intervention_mask = np.ones(n_min, dtype=bool)
        intervention_indices = np.array([], dtype=int)
        n_interventions = 0

    # Calculate tracking errors (excluding intervention points)
    errors_all = np.linalg.norm(actual_xyz - desired_xyz, axis=1)
    errors = errors_all[non_intervention_mask]

    if len(errors) > 0:
        mean_error = np.mean(errors)
        max_error = np.max(errors)
        std_error = np.std(errors)
    else:
        mean_error = max_error = std_error = 0.0

    print(f"\n{'='*60}")
    print(f"Trajectory Comparison Statistics")
    print(f"{'='*60}")
    print(f"Note: Comparing action[i] with next non-intervention state")
    print(f"      (intervention states are skipped in alignment)")
    total_states = len(actual_poses)
    print(f"Total states: {total_states}")
    if n_interventions > 0:
        print(f"Intervention states: {n_interventions} ({n_interventions/total_states*100:.1f}%)")
    print(f"Policy action-state pairs compared: {n_min}")
    print(f"Mean tracking error: {mean_error*1000:.2f} mm")
    print(f"Max tracking error:  {max_error*1000:.2f} mm")
    print(f"Std tracking error:  {std_error*1000:.2f} mm")
    print(f"{'='*60}\n")

    # Create figure based on mode
    if traj_only:
        fig = plt.figure(figsize=(12, 10))
        ax1 = fig.add_subplot(111, projection='3d')
    else:
        fig = plt.figure(figsize=(14, 10))
        ax1 = fig.add_subplot(221, projection='3d')

    # Plot actual trajectory (blue)
    ax1.plot(actual_xyz[:, 0], actual_xyz[:, 1], actual_xyz[:, 2],
            'b-', alpha=0.7, linewidth=1.5, label='Actual')
    ax1.scatter(actual_xyz[:, 0], actual_xyz[:, 1], actual_xyz[:, 2],
               c='blue', s=30, alpha=0.6, marker='o')

    # Plot desired trajectory (green)
    ax1.plot(desired_xyz[:, 0], desired_xyz[:, 1], desired_xyz[:, 2],
            'g--', alpha=0.7, linewidth=1.5, label='Desired')
    ax1.scatter(desired_xyz[:, 0], desired_xyz[:, 1], desired_xyz[:, 2],
               c='green', s=30, alpha=0.6, marker='^')

    # Draw thin connecting lines between corresponding action and state indices (interval of 10)
    for i in range(0, n_min, 10):
        ax1.plot([desired_xyz[i, 0], actual_xyz[i, 0]],
                [desired_xyz[i, 1], actual_xyz[i, 1]],
                [desired_xyz[i, 2], actual_xyz[i, 2]],
                color='gray', alpha=0.3, linewidth=0.8, linestyle='-')

    # Draw error arrows
    if show_error_arrows:
        for i in range(0, n_min, max(1, frame_label_interval)):
            ax1.quiver(actual_xyz[i, 0], actual_xyz[i, 1], actual_xyz[i, 2],
                      (desired_xyz[i, 0] - actual_xyz[i, 0]) * arrow_scale,
                      (desired_xyz[i, 1] - actual_xyz[i, 1]) * arrow_scale,
                      (desired_xyz[i, 2] - actual_xyz[i, 2]) * arrow_scale,
                      color='red', alpha=0.5, arrow_length_ratio=0.3, linewidth=1.5)

    # Mark start and end points
    ax1.scatter(actual_xyz[0, 0], actual_xyz[0, 1], actual_xyz[0, 2],
               c='black', s=150, marker='o', label='Start', alpha=0.9)
    ax1.scatter(actual_xyz[-1, 0], actual_xyz[-1, 1], actual_xyz[-1, 2],
               c='red', s=150, marker='X', label='End', alpha=0.9)

    # Frame labels for both actual and desired trajectories
    if frame_label_interval > 0:
        for i in range(0, n_min, frame_label_interval):
            label_idx = original_action_indices[i]
            # Actual trajectory labels (blue) - show state index
            ax1.text(actual_xyz[i, 0], actual_xyz[i, 1], actual_xyz[i, 2],
                    f"{label_idx+1}", fontsize=8, fontweight="bold", color='blue')
            # Desired trajectory labels (green) - show action index
            ax1.text(desired_xyz[i, 0], desired_xyz[i, 1], desired_xyz[i, 2],
                    f"{label_idx}", fontsize=8, fontweight="bold", color='green')

    # Mark intervention points with red crosses (from original data)
    if n_interventions > 0:
        # Get intervention states from original actual_poses
        intervention_state_indices = np.where(interventions == 1)[0]
        if len(intervention_state_indices) > 0:
            intervention_states = actual_poses[intervention_state_indices, :3]
            ax1.scatter(
                intervention_states[:, 0],
                intervention_states[:, 1],
                intervention_states[:, 2],
                c='red', s=200, marker='x', linewidths=3,
                alpha=0.9, label=f'Intervention ({n_interventions})', zorder=10
            )

    # Configure 3D axis
    (xlim, ylim, zlim) = axis_limits
    ax1.set_xlim(*xlim)
    ax1.set_ylim(*ylim)
    ax1.set_zlim(*zlim)
    ax1.set_xlabel('X Position (m)', fontsize=10)
    ax1.set_ylabel('Y Position (m)', fontsize=10)
    ax1.set_zlabel('Z Position (m)', fontsize=10)
    ax1.set_title(title, fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper right', fontsize=9)

    # Calculate XYZ error components (needed for stats)
    xyz_errors = (actual_xyz - desired_xyz) * 1000  # Convert to mm

    # Only create error analysis plots if not traj_only
    if not traj_only:
        # Use original action indices for x-axis (shows where in the episode these errors occurred)
        x_indices = original_action_indices

        # Error over time plot
        ax2 = fig.add_subplot(222)
        ax2.plot(x_indices, errors * 1000, 'r-', linewidth=2, label='Tracking Error')
        ax2.axhline(mean_error * 1000, color='b', linestyle='--', linewidth=1.5,
                    label=f'Mean: {mean_error*1000:.2f}mm')
        ax2.fill_between(x_indices,
                         (mean_error - std_error) * 1000,
                         (mean_error + std_error) * 1000,
                         alpha=0.2, color='blue', label=f'±1σ')
        ax2.set_xlabel('Action Index', fontsize=10)
        ax2.set_ylabel('Tracking Error (mm)', fontsize=10)
        ax2.set_title('Tracking Error Over Time', fontsize=11, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.legend(fontsize=9)

        # XYZ error components
        ax3 = fig.add_subplot(223)
        ax3.plot(x_indices, xyz_errors[:, 0], 'r-', label='X error', linewidth=1.5, alpha=0.8)
        ax3.plot(x_indices, xyz_errors[:, 1], 'g-', label='Y error', linewidth=1.5, alpha=0.8)
        ax3.plot(x_indices, xyz_errors[:, 2], 'b-', label='Z error', linewidth=1.5, alpha=0.8)
        ax3.axhline(0, color='black', linestyle='-', linewidth=0.5)
        ax3.set_xlabel('Action Index', fontsize=10)
        ax3.set_ylabel('Position Error (mm)', fontsize=10)
        ax3.set_title('XYZ Error Components', fontsize=11, fontweight='bold')
        ax3.grid(True, alpha=0.3)
        ax3.legend(fontsize=9)

        # Error histogram
        ax4 = fig.add_subplot(224)
        ax4.hist(errors * 1000, bins=30, color='steelblue', alpha=0.7, edgecolor='black')
        ax4.axvline(mean_error * 1000, color='red', linestyle='--', linewidth=2,
                    label=f'Mean: {mean_error*1000:.2f}mm')
        ax4.set_xlabel('Tracking Error (mm)', fontsize=10)
        ax4.set_ylabel('Frequency', fontsize=10)
        ax4.set_title('Error Distribution', fontsize=11, fontweight='bold')
        ax4.grid(True, alpha=0.3, axis='y')
        ax4.legend(fontsize=9)

    plt.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(str(save_path), dpi=300, bbox_inches='tight')
        print(f"Comparison plot saved to: {save_path}")

    plt.show()

    stats = {
        'n_points': n_min,
        'n_interventions': n_interventions,
        'mean_error_m': mean_error,
        'max_error_m': max_error,
        'std_error_m': std_error,
        'mean_error_mm': mean_error * 1000,
        'max_error_mm': max_error * 1000,
        'std_error_mm': std_error * 1000,
        'errors': errors,
        'errors_all': errors_all,
        'xyz_errors': xyz_errors / 1000,  # Back to meters
        'original_action_indices': original_action_indices,
    }

    if traj_only:
        return fig, ax1, stats
    else:
        return fig, (ax1, ax2, ax3, ax4), stats



if __name__ == "__main__":
    # Example 1: Visualize actual trajectory from recorded episode
    poses = np.load('/home/mingxi/mingxi_ws/crisp/crisp_py/raw_datasets/episodes/episode_20251223_124900_335/state/pose_wrt_world.npy')
    interventions = np.load('/home/mingxi/mingxi_ws/crisp/crisp_py/raw_datasets/episodes/episode_20251223_124900_335/state/intervention.npy')
    # interventions = None
    visualize_3d_trajectory_positions(
        poses,
        title="Example Trajectory (pregrasp positions only)",
        frame_label_interval=5,
        interventions=interventions,
    )

    # Example 2: Compare actual vs desired trajectory
    # actual_poses = np.load('/home/mingxi/mingxi_ws/crisp/crisp_py/raw_datasets/episodes/episode_20251222_182313_198/state/pose_wrt_world.npy') # Move to
    # interventions = np.load('/home/mingxi/mingxi_ws/crisp/crisp_py/raw_datasets/episodes/episode_20251222_182313_198/state/intervention.npy') # Move to
    # desired_actions = np.load('/home/mingxi/mingxi_ws/crisp/crisp_py/debug_data/actions_20251222_182517.npy') # Move to
    actual_poses = np.load('/home/mingxi/mingxi_ws/crisp/crisp_py/raw_datasets/episodes/episode_20251222_184900_444/state/pose_wrt_world.npy') # Move to
    interventions = np.load('/home/mingxi/mingxi_ws/crisp/crisp_py/raw_datasets/episodes/episode_20251222_184900_444/state/intervention.npy') # Move to
    desired_actions = np.load('/home/mingxi/mingxi_ws/crisp/crisp_py/debug_data/actions_20251222_185142.npy') # Move to


    # actual_poses = np.load('/home/mingxi/mingxi_ws/crisp/crisp_py/raw_datasets/episodes/episode_20251222_182450_977/state/pose_wrt_world.npy') # Set Target
    # interventions = np.load('/home/mingxi/mingxi_ws/crisp/crisp_py/raw_datasets/episodes/episode_20251222_182450_977/state/intervention.npy') # Set Target
    # desired_actions = np.load('/home/mingxi/mingxi_ws/crisp/crisp_py/debug_data/actions_20251222_182409.npy') # Set Target
    # actual_poses = np.load('/home/mingxi/mingxi_ws/crisp/crisp_py/raw_datasets/episodes/episode_20251222_185313_763/state/pose_wrt_world.npy') # Set Target
    # interventions = np.load('/home/mingxi/mingxi_ws/crisp/crisp_py/raw_datasets/episodes/episode_20251222_185313_763/state/intervention.npy') # Set Target
    # desired_actions = np.load('/home/mingxi/mingxi_ws/crisp/crisp_py/debug_data/actions_20251222_185429.npy') # Set Target


    # Compare trajectories - Full analysis view
    # fig, axes, stats = compare_actual_vs_desired_trajectory(
    #     actual_poses,
    #     desired_actions,
    #     title="Actual vs Desired Trajectory Comparison",
    #     save_path="debug_plots/trajectory_comparison_move_to.png",
    #     frame_label_interval=5,
    #     show_error_arrows=False,
    #     arrow_scale=1.0,
    # )

    # Compare trajectories - Trajectory only (large plot)
    # fig, ax, stats = compare_actual_vs_desired_trajectory(
    #     actual_poses,
    #     desired_actions,
    #     title="Actual vs Desired Trajectory Comparison",
    #     save_path="debug_plots/trajectory_comparison_move_to_large.png",
    #     # save_path="debug_plots/trajectory_comparison_set_target_large.png",
    #     frame_label_interval=5,
    #     show_error_arrows=False,
    #     arrow_scale=1.0,
    #     traj_only=True,  # Show only the 3D trajectory plot
    #     interventions=interventions,  # Mark intervention points with red crosses
    # )

    # Print statistics
    # print(f"Mean error: {stats['mean_error_mm']:.2f} mm")
    # print(f"Max error: {stats['max_error_mm']:.2f} mm")

    # Example 3: Smoothed trajectory
    # smoothed_xyz = smooth_xyz_trajectory(poses[:, :3], method="savgol", savgol_window=11, savgol_polyorder=3)
    # poses_sm = np.array(poses, copy=True)
    # poses_sm[:, :3] = smoothed_xyz
    # ax = visualize_3d_trajectory_from_poses(poses_sm, title="Smoothed trajectory (Savitzky–Golay)")
    # plt.show()