"""Diagnose why identical observations have different inference times
in different execution contexts.

Key finding: Same obs → fast in standalone script, slow/variable in robot script
This suggests environmental/context differences, not data differences.
"""

import os
import sys
import time
import numpy as np
import torch
import dill
import hydra
import threading
import gc
import psutil

# Diffusion Imports
sys.path.append('/home/mingxi/mingxi_ws/handpi/diffusion_policy')
from diffusion_policy.workspace.base_workspace import BaseWorkspace
from diffusion_policy.policy.base_image_policy import BaseImagePolicy
from diffusion_policy.common.pytorch_util import dict_apply


def create_dummy_observation():
    """Create a dummy observation."""
    obs = {}
    obs['pcd'] = np.random.randn(1024, 6).astype(np.float32)
    obs['render_pcd'] = np.random.randn(1024, 6).astype(np.float32)
    obs['robot0_eye_in_hand_image'] = np.random.rand(3, 128, 128).astype(np.float32)
    obs['robot0_eef_pos'] = np.random.randn(3).astype(np.float32)
    obs['robot0_eef_quat'] = np.array([0, 0, 0, 1], dtype=np.float32)
    obs['robot0_gripper_qpos'] = np.array([0, 0], dtype=np.float32)
    return obs


def check_system_state(label):
    """Check current system state that might affect inference."""

    state = {
        'label': label,
        'timestamp': time.time(),
    }

    # CPU info
    state['cpu_percent'] = psutil.cpu_percent(interval=0.1)
    state['cpu_freq'] = psutil.cpu_freq().current if psutil.cpu_freq() else None

    # Memory info
    mem = psutil.virtual_memory()
    state['ram_percent'] = mem.percent
    state['ram_available_gb'] = mem.available / (1024**3)

    # GPU info
    if torch.cuda.is_available():
        state['gpu_memory_allocated_mb'] = torch.cuda.memory_allocated(0) / (1024**2)
        state['gpu_memory_reserved_mb'] = torch.cuda.memory_reserved(0) / (1024**2)
        state['gpu_memory_cached_mb'] = torch.cuda.memory_reserved(0) / (1024**2) - torch.cuda.memory_allocated(0) / (1024**2)

    # Thread info
    state['num_threads'] = threading.active_count()
    state['thread_names'] = [t.name for t in threading.enumerate()]

    # CUDA stream info
    if torch.cuda.is_available():
        state['cuda_device'] = torch.cuda.current_device()
        # Check if there are any pending operations
        torch.cuda.synchronize()

    return state


def measure_inference_in_clean_context(policy, obs_dict, device, num_runs=20):
    """Measure inference in a clean standalone context."""

    print("\n" + "="*80)
    print("CLEAN CONTEXT (Standalone Script)")
    print("="*80)

    # Check system state before
    state_before = check_system_state("Clean context - before")

    # Warmup
    print("\nWarming up...")
    for _ in range(10):
        with torch.no_grad():
            obs_gpu = dict_apply(obs_dict,
                lambda x: torch.from_numpy(x).unsqueeze(0).unsqueeze(1).to(device))
            result = policy.predict_action(obs_gpu)
            del result
            del obs_gpu

    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    gc.collect()

    # Measure
    print(f"Measuring {num_runs} runs...")
    times = []
    states = []

    for i in range(num_runs):
        state_during = check_system_state(f"Clean run {i}")

        torch.cuda.synchronize()
        t0 = time.time()

        with torch.no_grad():
            obs_gpu = dict_apply(obs_dict,
                lambda x: torch.from_numpy(x).unsqueeze(0).unsqueeze(1).to(device))
            result = policy.predict_action(obs_gpu)
            del result
            del obs_gpu

        torch.cuda.synchronize()
        t_elapsed = (time.time() - t0) * 1000
        times.append(t_elapsed)
        states.append(state_during)

        if (i + 1) % 5 == 0:
            print(f"  Run {i+1}/{num_runs}: {t_elapsed:.2f} ms")

    state_after = check_system_state("Clean context - after")

    return times, states, state_before, state_after


def measure_inference_with_background_activity(policy, obs_dict, device, num_runs=20):
    """Measure inference with simulated background activity (like ROS)."""

    print("\n" + "="*80)
    print("WITH BACKGROUND ACTIVITY (Simulating Robot Script)")
    print("="*80)

    # Start background threads to simulate ROS activity
    stop_flag = threading.Event()

    def background_worker(worker_id):
        """Simulate background processing like ROS callbacks."""
        while not stop_flag.is_set():
            # Simulate some CPU work
            _ = np.random.randn(100, 100) @ np.random.randn(100, 100)
            time.sleep(0.01)

    # Start background threads
    num_workers = 3
    threads = []
    for i in range(num_workers):
        t = threading.Thread(target=background_worker, args=(i,), daemon=True)
        t.start()
        threads.append(t)

    print(f"Started {num_workers} background threads simulating ROS activity...")
    time.sleep(0.5)  # Let threads stabilize

    # Check system state before
    state_before = check_system_state("Background context - before")

    # Warmup
    print("\nWarming up...")
    for _ in range(10):
        with torch.no_grad():
            obs_gpu = dict_apply(obs_dict,
                lambda x: torch.from_numpy(x).unsqueeze(0).unsqueeze(1).to(device))
            result = policy.predict_action(obs_gpu)
            del result
            del obs_gpu

    torch.cuda.synchronize()

    # Measure
    print(f"Measuring {num_runs} runs...")
    times = []
    states = []

    for i in range(num_runs):
        state_during = check_system_state(f"Background run {i}")

        torch.cuda.synchronize()
        t0 = time.time()

        with torch.no_grad():
            obs_gpu = dict_apply(obs_dict,
                lambda x: torch.from_numpy(x).unsqueeze(0).unsqueeze(1).to(device))
            result = policy.predict_action(obs_gpu)
            del result
            del obs_gpu

        torch.cuda.synchronize()
        t_elapsed = (time.time() - t0) * 1000
        times.append(t_elapsed)
        states.append(state_during)

        if (i + 1) % 5 == 0:
            print(f"  Run {i+1}/{num_runs}: {t_elapsed:.2f} ms")

    state_after = check_system_state("Background context - after")

    # Stop background threads
    stop_flag.set()
    for t in threads:
        t.join(timeout=1.0)

    return times, states, state_before, state_after


def analyze_context_differences(clean_times, clean_states, bg_times, bg_states):
    """Analyze differences between clean and background contexts."""

    print("\n" + "="*80)
    print("CONTEXT COMPARISON ANALYSIS")
    print("="*80)

    # Timing comparison
    print("\nTiming Statistics:")
    print(f"{'Metric':<20} {'Clean Context':<20} {'Background Context':<20} {'Difference':<15}")
    print("-" * 75)

    metrics = [
        ("Mean", np.mean(clean_times), np.mean(bg_times)),
        ("Median", np.median(clean_times), np.median(bg_times)),
        ("Std Dev", np.std(clean_times), np.std(bg_times)),
        ("Min", np.min(clean_times), np.min(bg_times)),
        ("Max", np.max(clean_times), np.max(bg_times)),
        ("Range", np.max(clean_times) - np.min(clean_times), np.max(bg_times) - np.min(bg_times)),
        ("CV %", np.std(clean_times)/np.mean(clean_times)*100, np.std(bg_times)/np.mean(bg_times)*100),
    ]

    for name, clean_val, bg_val in metrics:
        diff = bg_val - clean_val
        diff_pct = (diff / clean_val * 100) if clean_val > 0 else 0
        print(f"{name:<20} {clean_val:>8.2f} ms       {bg_val:>8.2f} ms       {diff:+8.2f} ms ({diff_pct:+.1f}%)")

    # System state comparison
    print("\n" + "="*80)
    print("SYSTEM STATE COMPARISON (averaged across runs)")
    print("="*80)

    # Average states
    clean_avg = {
        'cpu_percent': np.mean([s['cpu_percent'] for s in clean_states]),
        'num_threads': np.mean([s['num_threads'] for s in clean_states]),
        'gpu_memory_allocated_mb': np.mean([s['gpu_memory_allocated_mb'] for s in clean_states]),
        'gpu_memory_cached_mb': np.mean([s['gpu_memory_cached_mb'] for s in clean_states]),
    }

    bg_avg = {
        'cpu_percent': np.mean([s['cpu_percent'] for s in bg_states]),
        'num_threads': np.mean([s['num_threads'] for s in bg_states]),
        'gpu_memory_allocated_mb': np.mean([s['gpu_memory_allocated_mb'] for s in bg_states]),
        'gpu_memory_cached_mb': np.mean([s['gpu_memory_cached_mb'] for s in bg_states]),
    }

    print(f"\n{'Metric':<30} {'Clean':<15} {'Background':<15} {'Difference':<15}")
    print("-" * 75)
    print(f"{'CPU Usage (%)':<30} {clean_avg['cpu_percent']:>8.2f}%     {bg_avg['cpu_percent']:>8.2f}%     {bg_avg['cpu_percent']-clean_avg['cpu_percent']:+8.2f}%")
    print(f"{'Active Threads':<30} {clean_avg['num_threads']:>8.1f}       {bg_avg['num_threads']:>8.1f}       {bg_avg['num_threads']-clean_avg['num_threads']:+8.1f}")
    print(f"{'GPU Memory Allocated (MB)':<30} {clean_avg['gpu_memory_allocated_mb']:>8.2f}      {bg_avg['gpu_memory_allocated_mb']:>8.2f}      {bg_avg['gpu_memory_allocated_mb']-clean_avg['gpu_memory_allocated_mb']:+8.2f}")
    print(f"{'GPU Memory Cached (MB)':<30} {clean_avg['gpu_memory_cached_mb']:>8.2f}      {bg_avg['gpu_memory_cached_mb']:>8.2f}      {bg_avg['gpu_memory_cached_mb']-clean_avg['gpu_memory_cached_mb']:+8.2f}")

    # Look for correlations
    print("\n" + "="*80)
    print("CORRELATION ANALYSIS")
    print("="*80)

    from scipy.stats import pearsonr

    # For background context, check if system metrics correlate with timing
    bg_cpu = [s['cpu_percent'] for s in bg_states]
    bg_gpu_mem = [s['gpu_memory_allocated_mb'] for s in bg_states]

    corr_cpu, p_cpu = pearsonr(bg_cpu, bg_times)
    corr_mem, p_mem = pearsonr(bg_gpu_mem, bg_times)

    print(f"\nIn background context:")
    print(f"  CPU usage ↔ Inference time:      r = {corr_cpu:+.3f} (p = {p_cpu:.4f})")
    print(f"  GPU memory ↔ Inference time:     r = {corr_mem:+.3f} (p = {p_mem:.4f})")


def check_cuda_context_state():
    """Check CUDA context state that might affect performance."""

    print("\n" + "="*80)
    print("CUDA CONTEXT STATE")
    print("="*80)

    if not torch.cuda.is_available():
        print("CUDA not available")
        return

    print(f"\nCurrent device: {torch.cuda.current_device()}")
    print(f"Device name: {torch.cuda.get_device_name(0)}")

    # Check CUDA stream
    print(f"\nCUDA streams:")
    print(f"  Current stream: {torch.cuda.current_stream()}")
    print(f"  Default stream: {torch.cuda.default_stream()}")

    # Check cuDNN settings
    print(f"\ncuDNN settings:")
    print(f"  Enabled: {torch.backends.cudnn.enabled}")
    print(f"  Benchmark: {torch.backends.cudnn.benchmark}")
    print(f"  Deterministic: {torch.backends.cudnn.deterministic}")

    # Check memory management
    print(f"\nCUDA memory management:")
    print(f"  Allocated: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB")
    print(f"  Reserved:  {torch.cuda.memory_reserved(0) / 1024**2:.2f} MB")
    print(f"  Max allocated: {torch.cuda.max_memory_allocated(0) / 1024**2:.2f} MB")

    # Check for pending operations
    torch.cuda.synchronize()
    print(f"  Synchronized: OK")


def main():
    print("="*80)
    print("CONTEXT DIFFERENCE DIAGNOSTIC")
    print("="*80)
    print("\nInvestigating why identical observations have different inference times")
    print("in standalone vs robot script contexts.\n")

    # Load checkpoint
    ckpt_path = "/home/mingxi/Downloads/epoch=0040-val_loss=0.008.ckpt"
    print(f"Loading checkpoint: {ckpt_path}")

    payload = torch.load(open(ckpt_path, 'rb'), pickle_module=dill)
    cfg = payload['cfg']
    cfg.logging.resume = False
    cfg.logging.mode = 'offline'
    cfg.real_robot_eval = True

    cls = hydra.utils.get_class(cfg._target_)
    workspace = cls(cfg)
    workspace.load_payload(payload, exclude_keys=None, include_keys=None)

    # Setup policy
    policy: BaseImagePolicy = workspace.model
    device = torch.device('cuda')
    policy.eval()
    policy.to(device)
    policy.num_inference_steps = 20
    policy.n_action_steps = 8
    policy.reset()

    print(f"✓ Policy loaded")

    # Check CUDA context
    check_cuda_context_state()

    # Create observation (use same obs for both tests)
    obs_dict = create_dummy_observation()
    print(f"\n✓ Created observation")

    # Test 1: Clean context
    clean_times, clean_states, clean_before, clean_after = \
        measure_inference_in_clean_context(policy, obs_dict, device, num_runs=30)

    # Clear everything
    torch.cuda.empty_cache()
    gc.collect()
    time.sleep(1.0)

    # Reset policy
    policy.reset()

    # Test 2: With background activity
    bg_times, bg_states, bg_before, bg_after = \
        measure_inference_with_background_activity(policy, obs_dict, device, num_runs=30)

    # Analyze
    analyze_context_differences(clean_times, clean_states, bg_times, bg_states)

    # Additional diagnostics
    print("\n" + "="*80)
    print("POTENTIAL CAUSES")
    print("="*80)

    mean_diff = np.mean(bg_times) - np.mean(clean_times)
    std_diff = np.std(bg_times) - np.std(clean_times)

    causes = []

    if mean_diff > 5:
        causes.append(("Background CPU contention",
                      f"Background context is {mean_diff:.1f}ms slower on average"))

    if std_diff > 1:
        causes.append(("Increased variability from context switching",
                      f"Background context has {std_diff:.1f}ms more variance"))

    if bg_avg := np.mean([s['num_threads'] for s in bg_states]) > 5:
        causes.append(("Thread contention",
                      f"Background context has {bg_avg:.0f} active threads"))

    # Check if times are actually similar
    if abs(mean_diff) < 2 and abs(std_diff) < 1:
        print("\n✓ No significant difference found between contexts!")
        print("  This suggests the issue in your robot script is NOT from:")
        print("    - Background threads")
        print("    - System load")
        print("    - Thread contention")
        print("\n  The issue might be:")
        print("    - GPU frequency scaling (check with nvidia-smi)")
        print("    - Thermal throttling")
        print("    - Something specific to ROS2 or the real data pipeline")
    else:
        print("\n⚠ Context differences detected:\n")
        for i, (cause, evidence) in enumerate(causes, 1):
            print(f"{i}. {cause}")
            print(f"   {evidence}\n")

    print("\n" + "="*80)
    print("RECOMMENDATIONS")
    print("="*80)
    print("""
To diagnose the real script issue:

1. **Add timing inside your robot script**:
   In pcd_diff_eval_debug_inf_time.py, add:

   import threading
   print(f"Active threads: {threading.active_count()}")
   print(f"Thread names: {[t.name for t in threading.enumerate()]}")

2. **Check GPU state during inference**:

   import subprocess
   result = subprocess.run(['nvidia-smi', 'dmon', '-c', '1'],
                          capture_output=True, text=True)
   print(result.stdout)

3. **Test with policy.reset() before each inference**:
   See if calling policy.reset() between runs affects timing

4. **Check if ROS spinning affects GPU**:
   Try temporarily stopping the ROS spin thread during inference

5. **Monitor GPU clocks in real-time**:
   In another terminal:
   watch -n 0.1 nvidia-smi

   Look for GPU clock changes during inference
""")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
