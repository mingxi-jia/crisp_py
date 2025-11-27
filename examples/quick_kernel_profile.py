"""Quick kernel-level profiling to identify slow operations.

This is a simpler version that focuses on:
1. Which CUDA kernels are running
2. How long each kernel takes
3. Identifying variance in kernel execution times
"""

import os
import sys
import time
import numpy as np
import torch
import dill
import hydra
from collections import defaultdict

# Diffusion Imports
sys.path.append('/home/mingxi/mingxi_ws/handpi/diffusion_policy')
from diffusion_policy.workspace.base_workspace import BaseWorkspace
from diffusion_policy.policy.base_image_policy import BaseImagePolicy
from diffusion_policy.common.pytorch_util import dict_apply


def create_dummy_observation():
    """Create a dummy observation matching the expected format."""
    obs = {}
    obs['pcd'] = np.random.randn(1024, 6).astype(np.float32)
    obs['render_pcd'] = np.random.randn(1024, 6).astype(np.float32)
    obs['robot0_eye_in_hand_image'] = np.random.rand(3, 128, 128).astype(np.float32)
    obs['robot0_eef_pos'] = np.random.randn(3).astype(np.float32)
    obs['robot0_eef_quat'] = np.array([0, 0, 0, 1], dtype=np.float32)
    obs['robot0_gripper_qpos'] = np.array([0, 0], dtype=np.float32)
    return obs


def main():
    print("="*80)
    print("QUICK KERNEL PROFILING")
    print("="*80)

    # Load checkpoint
    ckpt_path = "/home/mingxi/Downloads/epoch=0040-val_loss=0.008.ckpt"
    print(f"\nLoading checkpoint...")

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

    # Create dummy observation
    obs_dict = create_dummy_observation()

    # Warmup
    print("\nWarming up (10 iterations)...")
    for _ in range(10):
        with torch.no_grad():
            obs_dict_gpu = dict_apply(obs_dict,
                lambda x: torch.from_numpy(x).unsqueeze(0).unsqueeze(1).to(device))
            result = policy.predict_action(obs_dict_gpu)
            del result
            del obs_dict_gpu

    torch.cuda.synchronize()

    # Profile with PyTorch autograd profiler (simpler, faster)
    print("\nProfiling 20 inference runs...")
    print("(This will take about 1-2 minutes)\n")

    with torch.autograd.profiler.profile(
        use_cuda=True,
        record_shapes=False,
        with_stack=False,
    ) as prof:
        with torch.no_grad():
            for i in range(20):
                obs_dict_gpu = dict_apply(obs_dict,
                    lambda x: torch.from_numpy(x).unsqueeze(0).unsqueeze(1).to(device))
                result = policy.predict_action(obs_dict_gpu)
                del result
                del obs_dict_gpu

                if (i + 1) % 5 == 0:
                    print(f"  Completed {i+1}/20 runs")

    print("\n" + "="*80)
    print("PROFILING RESULTS")
    print("="*80)

    # Print top CUDA operations
    print("\nTop 25 CUDA operations by total time:\n")
    print(prof.key_averages().table(
        sort_by="cuda_time_total",
        row_limit=25
    ))

    # Print top CPU operations
    print("\n" + "="*80)
    print("Top 25 CPU operations by total time:\n")
    print(prof.key_averages().table(
        sort_by="cpu_time_total",
        row_limit=25
    ))

    # Analyze kernel variance
    print("\n" + "="*80)
    print("KERNEL VARIANCE ANALYSIS")
    print("="*80)

    # Get all events
    events_list = prof.function_events

    # Group by kernel name
    kernel_times = defaultdict(list)
    for event in events_list:
        if event.is_cuda and event.cuda_time_total > 0:
            kernel_times[event.key].append(event.cuda_time)

    # Find kernels with high variance
    kernel_stats = []
    for kernel, times in kernel_times.items():
        if len(times) > 5:  # Only consider kernels called multiple times
            mean_time = np.mean(times)
            std_time = np.std(times)
            cv = (std_time / mean_time * 100) if mean_time > 0 else 0
            kernel_stats.append({
                'kernel': kernel,
                'count': len(times),
                'mean': mean_time / 1000,  # Convert to ms
                'std': std_time / 1000,
                'cv': cv,
                'min': np.min(times) / 1000,
                'max': np.max(times) / 1000,
            })

    # Sort by coefficient of variation (relative variability)
    kernel_stats_sorted = sorted(kernel_stats, key=lambda x: x['cv'], reverse=True)

    print("\nKernels with HIGHEST variability (CV = coefficient of variation):")
    print(f"{'Kernel':<60} {'Count':<8} {'Mean (ms)':<12} {'Std (ms)':<12} {'CV %':<10}")
    print("-" * 105)

    for stat in kernel_stats_sorted[:15]:
        kernel_name = stat['kernel'][:58]
        print(f"{kernel_name:<60} {stat['count']:<8} {stat['mean']:<12.4f} {stat['std']:<12.4f} {stat['cv']:<10.2f}")

    # Find slowest kernels
    kernel_stats_by_mean = sorted(kernel_stats, key=lambda x: x['mean'], reverse=True)

    print("\n" + "="*80)
    print("SLOWEST kernels (by mean execution time):")
    print(f"{'Kernel':<60} {'Count':<8} {'Mean (ms)':<12} {'Min (ms)':<12} {'Max (ms)':<12}")
    print("-" * 117)

    for stat in kernel_stats_by_mean[:15]:
        kernel_name = stat['kernel'][:58]
        print(f"{kernel_name:<60} {stat['count']:<8} {stat['mean']:<12.4f} {stat['min']:<12.4f} {stat['max']:<12.4f}")

    # Look for synchronization patterns
    print("\n" + "="*80)
    print("SYNCHRONIZATION AND TRANSFER OPERATIONS")
    print("="*80)

    sync_ops = [e for e in events_list if 'synchronize' in e.key.lower() or 'stream' in e.key.lower()]
    memcpy_ops = [e for e in events_list if 'memcpy' in e.key.lower() or 'copy' in e.key.lower()]

    if sync_ops:
        total_sync_time = sum(e.cuda_time for e in sync_ops) / 1000
        print(f"\nSynchronization operations: {len(sync_ops)} calls, {total_sync_time:.2f} ms total")

    if memcpy_ops:
        total_memcpy_time = sum(e.cuda_time for e in memcpy_ops) / 1000
        print(f"Memory copy operations: {len(memcpy_ops)} calls, {total_memcpy_time:.2f} ms total")

    print("\n" + "="*80)
    print("KEY FINDINGS")
    print("="*80)

    # Calculate total time distribution
    total_cuda_time = sum(e.cuda_time_total for e in prof.key_averages()) / 1000

    # Identify main operation categories
    conv_time = sum(e.cuda_time_total for e in prof.key_averages() if 'conv' in e.key.lower()) / 1000
    matmul_time = sum(e.cuda_time_total for e in prof.key_averages()
                     if any(x in e.key.lower() for x in ['gemm', 'matmul', 'bmm'])) / 1000
    norm_time = sum(e.cuda_time_total for e in prof.key_averages()
                   if any(x in e.key.lower() for x in ['norm', 'batch_norm', 'layer_norm'])) / 1000
    attention_time = sum(e.cuda_time_total for e in prof.key_averages()
                        if any(x in e.key.lower() for x in ['attention', 'softmax'])) / 1000

    print(f"\nTime breakdown by operation type:")
    print(f"  Convolutions:         {conv_time:8.2f} ms ({conv_time/total_cuda_time*100:5.1f}%)")
    print(f"  Matrix multiplies:    {matmul_time:8.2f} ms ({matmul_time/total_cuda_time*100:5.1f}%)")
    print(f"  Normalizations:       {norm_time:8.2f} ms ({norm_time/total_cuda_time*100:5.1f}%)")
    print(f"  Attention/Softmax:    {attention_time:8.2f} ms ({attention_time/total_cuda_time*100:5.1f}%)")
    print(f"  Total:                {total_cuda_time:8.2f} ms")

    # Identify if there are a few dominant slow kernels
    top_3_time = sum(stat['mean'] for stat in kernel_stats_by_mean[:3])
    print(f"\nTop 3 slowest kernels account for: {top_3_time:.2f} ms ({top_3_time/total_cuda_time*20*100:.1f}% of total)")

    print("\n" + "="*80)
    print("COMPLETE")
    print("="*80)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
