"""Script to diagnose inference time variability in diffusion policy.

This script runs detailed profiling and monitoring to identify the cause of
inference time variations (0.2s to 5s) on identical inputs.
"""

import os
import sys
import time
import numpy as np
import torch
import dill
import hydra
from collections import defaultdict
import gc

# Diffusion Imports
sys.path.append('/home/mingxi/mingxi_ws/handpi/diffusion_policy')
from diffusion_policy.workspace.base_workspace import BaseWorkspace
from diffusion_policy.policy.base_image_policy import BaseImagePolicy
from diffusion_policy.common.pytorch_util import dict_apply


def create_dummy_observation(shape_meta):
    """Create a dummy observation matching the expected format."""
    obs = {}

    # Common observation components based on the main script
    obs['pcd'] = np.random.randn(1024, 6).astype(np.float32)
    obs['render_pcd'] = np.random.randn(1024, 6).astype(np.float32)
    obs['robot0_eye_in_hand_image'] = np.random.rand(3, 128, 128).astype(np.float32)
    obs['robot0_eef_pos'] = np.random.randn(3).astype(np.float32)
    obs['robot0_eef_quat'] = np.array([0, 0, 0, 1], dtype=np.float32)
    obs['robot0_gripper_qpos'] = np.array([0, 0], dtype=np.float32)

    return obs


def measure_gpu_utilization():
    """Measure current GPU utilization and memory."""
    try:
        import subprocess
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=utilization.gpu,memory.used,memory.free,temperature.gpu',
             '--format=csv,noheader,nounits'],
            capture_output=True, text=True, timeout=1
        )
        if result.returncode == 0:
            gpu_util, mem_used, mem_free, temp = result.stdout.strip().split(',')
            return {
                'gpu_util': float(gpu_util),
                'mem_used': float(mem_used),
                'mem_free': float(mem_free),
                'temp': float(temp)
            }
    except Exception as e:
        print(f"Warning: Could not get GPU stats: {e}")
    return None


def profile_inference_detailed(obs_dict, policy, device, num_runs=20):
    """Run detailed profiling of inference with timing breakdown."""

    print("\n" + "="*80)
    print("DETAILED INFERENCE PROFILING")
    print("="*80)

    timings = {
        'total': [],
        'dict_apply': [],
        'predict_action': [],
        'to_cpu': [],
        'gpu_util_before': [],
        'gpu_util_after': [],
        'mem_allocated': [],
        'mem_reserved': [],
    }

    for i in range(num_runs):
        # Clear cache before each run
        if i % 5 == 0:
            torch.cuda.empty_cache()
            gc.collect()

        # Measure GPU state before inference
        gpu_before = measure_gpu_utilization()
        mem_before = torch.cuda.memory_allocated(device)

        torch.cuda.synchronize()
        t_total_start = time.time()

        with torch.no_grad():
            # Time dict_apply (CPU->GPU transfer)
            t0 = time.time()
            obs_dict_gpu = dict_apply(obs_dict,
                lambda x: torch.from_numpy(x).unsqueeze(0).unsqueeze(1).to(device))
            torch.cuda.synchronize()
            t_dict_apply = time.time() - t0

            # Time predict_action
            t0 = time.time()
            result = policy.predict_action(obs_dict_gpu)
            torch.cuda.synchronize()
            t_predict = time.time() - t0

            # Time GPU->CPU transfer
            t0 = time.time()
            action = result['action'][0].detach().to('cpu').numpy()
            t_to_cpu = time.time() - t0

            del result
            del obs_dict_gpu

        torch.cuda.synchronize()
        t_total = time.time() - t_total_start

        # Measure GPU state after inference
        gpu_after = measure_gpu_utilization()
        mem_after = torch.cuda.memory_allocated(device)

        timings['total'].append(t_total * 1000)
        timings['dict_apply'].append(t_dict_apply * 1000)
        timings['predict_action'].append(t_predict * 1000)
        timings['to_cpu'].append(t_to_cpu * 1000)
        timings['mem_allocated'].append((mem_after - mem_before) / 1024**2)  # MB
        timings['mem_reserved'].append(torch.cuda.memory_reserved(device) / 1024**2)  # MB

        if gpu_before:
            timings['gpu_util_before'].append(gpu_before['gpu_util'])
        if gpu_after:
            timings['gpu_util_after'].append(gpu_after['gpu_util'])

        # Print progress
        if (i + 1) % 5 == 0:
            print(f"Run {i+1}/{num_runs}: {t_total*1000:6.1f} ms (predict: {t_predict*1000:6.1f} ms)")

    return timings


def analyze_timings(timings):
    """Analyze and print timing statistics."""

    print("\n" + "="*80)
    print("TIMING ANALYSIS")
    print("="*80)

    for key in ['total', 'dict_apply', 'predict_action', 'to_cpu']:
        times = timings[key]
        print(f"\n{key.upper()}:")
        print(f"  Mean:   {np.mean(times):8.2f} ms")
        print(f"  Median: {np.median(times):8.2f} ms")
        print(f"  Min:    {np.min(times):8.2f} ms")
        print(f"  Max:    {np.max(times):8.2f} ms")
        print(f"  Std:    {np.std(times):8.2f} ms")
        print(f"  P95:    {np.percentile(times, 95):8.2f} ms")
        print(f"  P99:    {np.percentile(times, 99):8.2f} ms")

        # Identify outliers (>2 std from mean)
        mean = np.mean(times)
        std = np.std(times)
        outliers = [i for i, t in enumerate(times) if abs(t - mean) > 2 * std]
        if outliers:
            print(f"  Outliers (>2σ): {len(outliers)} runs - indices: {outliers}")

    # Memory analysis
    if timings['mem_allocated']:
        print(f"\nMEMORY ALLOCATION (delta per run):")
        print(f"  Mean:   {np.mean(timings['mem_allocated']):8.2f} MB")
        print(f"  Max:    {np.max(timings['mem_allocated']):8.2f} MB")

    # GPU utilization analysis
    if timings['gpu_util_before']:
        print(f"\nGPU UTILIZATION BEFORE:")
        print(f"  Mean:   {np.mean(timings['gpu_util_before']):6.1f}%")
        print(f"  Range:  {np.min(timings['gpu_util_before']):6.1f}% - {np.max(timings['gpu_util_before']):6.1f}%")

    if timings['gpu_util_after']:
        print(f"\nGPU UTILIZATION AFTER:")
        print(f"  Mean:   {np.mean(timings['gpu_util_after']):6.1f}%")
        print(f"  Range:  {np.min(timings['gpu_util_after']):6.1f}% - {np.max(timings['gpu_util_after']):6.1f}%")


def test_cuda_profiler(obs_dict, policy, device):
    """Use PyTorch profiler to identify bottlenecks."""

    print("\n" + "="*80)
    print("CUDA PROFILER ANALYSIS")
    print("="*80)

    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        record_shapes=True,
        profile_memory=True,
        with_stack=True
    ) as prof:
        with torch.no_grad():
            for _ in range(5):
                obs_dict_gpu = dict_apply(obs_dict,
                    lambda x: torch.from_numpy(x).unsqueeze(0).unsqueeze(1).to(device))
                result = policy.predict_action(obs_dict_gpu)
                action = result['action'][0].detach().to('cpu').numpy()
                del result
                del obs_dict_gpu

    print("\nTop 10 operations by CUDA time:")
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))

    print("\nTop 10 operations by CPU time:")
    print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))


def test_warmup_effect(obs_dict, policy, device, num_warmup=5, num_test=20):
    """Test if warmup runs affect subsequent inference times."""

    print("\n" + "="*80)
    print("WARMUP EFFECT ANALYSIS")
    print("="*80)

    print(f"\nRunning {num_warmup} warmup iterations...")
    for i in range(num_warmup):
        with torch.no_grad():
            obs_dict_gpu = dict_apply(obs_dict,
                lambda x: torch.from_numpy(x).unsqueeze(0).unsqueeze(1).to(device))
            result = policy.predict_action(obs_dict_gpu)
            del result
            del obs_dict_gpu

    torch.cuda.synchronize()

    print(f"Running {num_test} timed iterations after warmup...")
    times = []
    for i in range(num_test):
        torch.cuda.synchronize()
        t0 = time.time()

        with torch.no_grad():
            obs_dict_gpu = dict_apply(obs_dict,
                lambda x: torch.from_numpy(x).unsqueeze(0).unsqueeze(1).to(device))
            result = policy.predict_action(obs_dict_gpu)
            action = result['action'][0].detach().to('cpu').numpy()
            del result
            del obs_dict_gpu

        torch.cuda.synchronize()
        times.append((time.time() - t0) * 1000)

    print(f"\nAfter warmup statistics:")
    print(f"  Mean:   {np.mean(times):8.2f} ms")
    print(f"  Min:    {np.min(times):8.2f} ms")
    print(f"  Max:    {np.max(times):8.2f} ms")
    print(f"  Std:    {np.std(times):8.2f} ms")


def test_cache_effects(obs_dict, policy, device):
    """Test if cache clearing affects timing."""

    print("\n" + "="*80)
    print("CACHE EFFECT ANALYSIS")
    print("="*80)

    scenarios = [
        ("No cache clear", False),
        ("Cache cleared before each", True),
    ]

    for scenario_name, clear_cache in scenarios:
        print(f"\n{scenario_name}:")
        times = []

        for i in range(10):
            if clear_cache:
                torch.cuda.empty_cache()
                gc.collect()

            torch.cuda.synchronize()
            t0 = time.time()

            with torch.no_grad():
                obs_dict_gpu = dict_apply(obs_dict,
                    lambda x: torch.from_numpy(x).unsqueeze(0).unsqueeze(1).to(device))
                result = policy.predict_action(obs_dict_gpu)
                action = result['action'][0].detach().to('cpu').numpy()
                del result
                del obs_dict_gpu

            torch.cuda.synchronize()
            times.append((time.time() - t0) * 1000)

        print(f"  Mean:   {np.mean(times):8.2f} ms")
        print(f"  Min:    {np.min(times):8.2f} ms")
        print(f"  Max:    {np.max(times):8.2f} ms")
        print(f"  Std:    {np.std(times):8.2f} ms")


def check_system_interference():
    """Check for potential system-level interference."""

    print("\n" + "="*80)
    print("SYSTEM INTERFERENCE CHECK")
    print("="*80)

    # Check GPU processes
    try:
        import subprocess
        result = subprocess.run(
            ['nvidia-smi', 'pmon', '-c', '1'],
            capture_output=True, text=True, timeout=2
        )
        print("\nGPU Processes:")
        print(result.stdout)
    except Exception as e:
        print(f"Could not check GPU processes: {e}")

    # Check CPU frequency scaling
    try:
        result = subprocess.run(
            ['cat', '/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor'],
            capture_output=True, text=True, timeout=1
        )
        print(f"\nCPU Governor: {result.stdout.strip()}")
    except Exception:
        pass

    # Check GPU power state
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=power.draw,power.limit', '--format=csv,noheader'],
            capture_output=True, text=True, timeout=1
        )
        print(f"GPU Power: {result.stdout.strip()}")
    except Exception:
        pass


def main():
    print("="*80)
    print("DIFFUSION POLICY INFERENCE TIME DIAGNOSTIC TOOL")
    print("="*80)

    # Load checkpoint
    ckpt_path = "/home/mingxi/Downloads/epoch=0040-val_loss=0.008.ckpt"
    print(f"\nLoading checkpoint: {ckpt_path}")

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

    print(f"Policy loaded. Inference steps: {policy.num_inference_steps}")
    print(f"Action steps: {policy.n_action_steps}")

    # Create dummy observation
    obs_dict = create_dummy_observation(cfg.task.shape_meta)
    print(f"\nObservation shapes:")
    for k, v in obs_dict.items():
        print(f"  {k}: {v.shape}")

    # Run diagnostics
    check_system_interference()

    # Test warmup effect
    test_warmup_effect(obs_dict, policy, device, num_warmup=10, num_test=30)

    # Test cache effects
    test_cache_effects(obs_dict, policy, device)

    # Detailed profiling
    timings = profile_inference_detailed(obs_dict, policy, device, num_runs=50)

    # Analyze timings
    analyze_timings(timings)

    # CUDA profiler (if not too slow)
    print("\nRun CUDA profiler? (This may take a while) [y/N]: ", end='')
    try:
        response = input().strip().lower()
        if response == 'y':
            test_cuda_profiler(obs_dict, policy, device)
    except:
        print("Skipping CUDA profiler")

    print("\n" + "="*80)
    print("DIAGNOSTIC COMPLETE")
    print("="*80)

    # Save timing data
    output_file = "inference_timing_analysis.npz"
    np.savez(output_file, **{k: np.array(v) for k, v in timings.items()})
    print(f"\nTiming data saved to: {output_file}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
