"""Comprehensive GPU profiling for diffusion policy inference.

This script uses PyTorch profiler to analyze:
1. CUDA kernel execution times
2. Memory transfers (cudaMemcpy)
3. Synchronization overhead (cudaDeviceSynchronize)
4. CPU vs GPU time breakdown
"""

import os
import sys
import time
import numpy as np
import torch
import dill
import hydra
from datetime import datetime

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


def profile_with_pytorch_profiler(obs_dict, policy, device, num_runs=10):
    """Use PyTorch profiler to analyze inference."""

    print("\n" + "="*80)
    print("PYTORCH PROFILER - DETAILED TIMELINE")
    print("="*80)

    # Warmup
    print("\nWarming up (5 iterations)...")
    for _ in range(5):
        with torch.no_grad():
            obs_dict_gpu = dict_apply(obs_dict,
                lambda x: torch.from_numpy(x).unsqueeze(0).unsqueeze(1).to(device))
            result = policy.predict_action(obs_dict_gpu)
            del result
            del obs_dict_gpu

    torch.cuda.synchronize()

    # Profile with detailed settings
    print(f"\nProfiling {num_runs} inference runs...")

    activities = [
        torch.profiler.ProfilerActivity.CPU,
        torch.profiler.ProfilerActivity.CUDA,
    ]

    with torch.profiler.profile(
        activities=activities,
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
        with_flops=True,
        with_modules=True,
        # Schedule for multiple iterations
        schedule=torch.profiler.schedule(
            wait=1,
            warmup=1,
            active=num_runs-2,
            repeat=1
        ),
        on_trace_ready=torch.profiler.tensorboard_trace_handler('./profiler_logs')
    ) as prof:
        for i in range(num_runs):
            with torch.no_grad():
                obs_dict_gpu = dict_apply(obs_dict,
                    lambda x: torch.from_numpy(x).unsqueeze(0).unsqueeze(1).to(device))
                result = policy.predict_action(obs_dict_gpu)
                action = result['action'][0].detach().to('cpu').numpy()
                del result
                del obs_dict_gpu

            prof.step()

    # Print summary tables
    print("\n" + "="*80)
    print("TOP CUDA OPERATIONS (by total time)")
    print("="*80)
    print(prof.key_averages().table(
        sort_by="cuda_time_total",
        row_limit=20,
        max_name_column_width=60
    ))

    print("\n" + "="*80)
    print("TOP CPU OPERATIONS (by total time)")
    print("="*80)
    print(prof.key_averages().table(
        sort_by="cpu_time_total",
        row_limit=20,
        max_name_column_width=60
    ))

    print("\n" + "="*80)
    print("MEMORY OPERATIONS")
    print("="*80)
    print(prof.key_averages().table(
        sort_by="cuda_memory_usage",
        row_limit=15,
        max_name_column_width=60
    ))

    print("\n" + "="*80)
    print("OPERATIONS WITH HIGHEST SELF TIME (actual compute, not children)")
    print("="*80)
    print(prof.key_averages().table(
        sort_by="self_cuda_time_total",
        row_limit=20,
        max_name_column_width=60
    ))

    # Export trace for visualization
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    trace_file = f"./inference_trace_{timestamp}.json"
    prof.export_chrome_trace(trace_file)
    print(f"\n✓ Chrome trace exported to: {trace_file}")
    print(f"  View at: chrome://tracing")

    return prof


def profile_synchronization_overhead(obs_dict, policy, device, num_runs=20):
    """Measure synchronization overhead specifically."""

    print("\n" + "="*80)
    print("SYNCHRONIZATION OVERHEAD ANALYSIS")
    print("="*80)

    times_with_sync = []
    times_without_sync = []

    print("\nMeasuring WITH synchronization...")
    for i in range(num_runs):
        torch.cuda.synchronize()
        t0 = time.time()

        with torch.no_grad():
            obs_dict_gpu = dict_apply(obs_dict,
                lambda x: torch.from_numpy(x).unsqueeze(0).unsqueeze(1).to(device))
            torch.cuda.synchronize()
            t_after_transfer = time.time()

            result = policy.predict_action(obs_dict_gpu)
            torch.cuda.synchronize()
            t_after_predict = time.time()

            action = result['action'][0].detach().to('cpu')
            torch.cuda.synchronize()
            t_end = time.time()

            del result
            del obs_dict_gpu

        times_with_sync.append({
            'total': (t_end - t0) * 1000,
            'transfer': (t_after_transfer - t0) * 1000,
            'predict': (t_after_predict - t_after_transfer) * 1000,
            'to_cpu': (t_end - t_after_predict) * 1000
        })

    print("\nMeasuring WITHOUT intermediate synchronization...")
    for i in range(num_runs):
        t0 = time.time()

        with torch.no_grad():
            obs_dict_gpu = dict_apply(obs_dict,
                lambda x: torch.from_numpy(x).unsqueeze(0).unsqueeze(1).to(device))
            result = policy.predict_action(obs_dict_gpu)
            action = result['action'][0].detach().to('cpu')
            del result
            del obs_dict_gpu

        torch.cuda.synchronize()
        t_end = time.time()

        times_without_sync.append((t_end - t0) * 1000)

    # Analyze results
    print("\nWith synchronization (component breakdown):")
    print(f"  Transfer:  {np.mean([t['transfer'] for t in times_with_sync]):6.2f} ms ± {np.std([t['transfer'] for t in times_with_sync]):5.2f}")
    print(f"  Predict:   {np.mean([t['predict'] for t in times_with_sync]):6.2f} ms ± {np.std([t['predict'] for t in times_with_sync]):5.2f}")
    print(f"  To CPU:    {np.mean([t['to_cpu'] for t in times_with_sync]):6.2f} ms ± {np.std([t['to_cpu'] for t in times_with_sync]):5.2f}")
    print(f"  Total:     {np.mean([t['total'] for t in times_with_sync]):6.2f} ms ± {np.std([t['total'] for t in times_with_sync]):5.2f}")

    print("\nWithout intermediate synchronization:")
    print(f"  Total:     {np.mean(times_without_sync):6.2f} ms ± {np.std(times_without_sync):5.2f}")

    sync_overhead = np.mean([t['total'] for t in times_with_sync]) - np.mean(times_without_sync)
    print(f"\nEstimated sync overhead: {sync_overhead:6.2f} ms")


def profile_individual_kernels(obs_dict, policy, device):
    """Profile to identify specific slow kernels."""

    print("\n" + "="*80)
    print("INDIVIDUAL KERNEL ANALYSIS")
    print("="*80)

    # Use profiler with more detailed kernel info
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CUDA],
        record_shapes=True,
        with_stack=False,
    ) as prof:
        with torch.no_grad():
            for _ in range(5):
                obs_dict_gpu = dict_apply(obs_dict,
                    lambda x: torch.from_numpy(x).unsqueeze(0).unsqueeze(1).to(device))
                result = policy.predict_action(obs_dict_gpu)
                del result
                del obs_dict_gpu

    # Get kernel statistics
    events = prof.key_averages()

    # Filter for CUDA kernels only
    cuda_kernels = [e for e in events if e.device_type == torch.profiler.DeviceType.CUDA]

    # Sort by self time (actual kernel execution)
    cuda_kernels_sorted = sorted(cuda_kernels, key=lambda x: x.self_cuda_time_total, reverse=True)

    print("\nTop 20 CUDA kernels by self execution time:")
    print(f"{'Kernel Name':<60} {'Count':<8} {'Total (ms)':<12} {'Avg (ms)':<12}")
    print("-" * 95)

    for i, event in enumerate(cuda_kernels_sorted[:20]):
        kernel_name = event.key[:60]
        count = event.count
        total_time = event.self_cuda_time_total / 1000  # Convert to ms
        avg_time = total_time / count if count > 0 else 0
        print(f"{kernel_name:<60} {count:<8} {total_time:<12.2f} {avg_time:<12.4f}")

    # Look for specific patterns
    print("\n" + "="*80)
    print("KERNEL PATTERN ANALYSIS")
    print("="*80)

    # Group by operation type
    conv_ops = [e for e in cuda_kernels if 'conv' in e.key.lower()]
    matmul_ops = [e for e in cuda_kernels if any(x in e.key.lower() for x in ['gemm', 'matmul', 'bmm'])]
    norm_ops = [e for e in cuda_kernels if any(x in e.key.lower() for x in ['norm', 'batch_norm', 'layer_norm'])]
    attention_ops = [e for e in cuda_kernels if 'attention' in e.key.lower() or 'softmax' in e.key.lower()]

    def print_op_stats(ops, name):
        if ops:
            total = sum(e.self_cuda_time_total for e in ops) / 1000
            count = sum(e.count for e in ops)
            print(f"\n{name}:")
            print(f"  Total time: {total:8.2f} ms")
            print(f"  Total calls: {count}")
            print(f"  Avg per call: {total/count:8.4f} ms")

    print_op_stats(conv_ops, "Convolution operations")
    print_op_stats(matmul_ops, "Matrix multiplication operations")
    print_op_stats(norm_ops, "Normalization operations")
    print_op_stats(attention_ops, "Attention operations")


def check_memory_transfer_patterns(obs_dict, policy, device, num_runs=10):
    """Check for memory transfer bottlenecks."""

    print("\n" + "="*80)
    print("MEMORY TRANSFER PATTERN ANALYSIS")
    print("="*80)

    # Profile with memory recording
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        profile_memory=True,
        record_shapes=True,
    ) as prof:
        with torch.no_grad():
            for _ in range(num_runs):
                obs_dict_gpu = dict_apply(obs_dict,
                    lambda x: torch.from_numpy(x).unsqueeze(0).unsqueeze(1).to(device))
                result = policy.predict_action(obs_dict_gpu)
                action = result['action'][0].detach().to('cpu').numpy()
                del result
                del obs_dict_gpu

    # Look for memory copy operations
    events = prof.key_averages()
    memcpy_events = [e for e in events if 'memcpy' in e.key.lower() or 'copy' in e.key.lower()]

    if memcpy_events:
        print("\nMemory copy operations detected:")
        print(f"{'Operation':<60} {'Count':<8} {'Total (ms)':<12} {'Avg (ms)':<12}")
        print("-" * 95)

        for event in sorted(memcpy_events, key=lambda x: x.cuda_time_total, reverse=True)[:10]:
            print(f"{event.key[:60]:<60} {event.count:<8} {event.cuda_time_total/1000:<12.2f} {event.cuda_time_total/event.count/1000:<12.4f}")
    else:
        print("\nNo explicit memory copy operations found in profile.")


def main():
    print("="*80)
    print("GPU PROFILING ANALYSIS FOR DIFFUSION POLICY INFERENCE")
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

    print(f"✓ Policy loaded")
    print(f"  Inference steps: {policy.num_inference_steps}")
    print(f"  Action steps: {policy.n_action_steps}")

    # Create dummy observation
    obs_dict = create_dummy_observation()

    # Print GPU info
    print(f"\nGPU Information:")
    print(f"  Device: {torch.cuda.get_device_name(0)}")
    print(f"  Total memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    print(f"  Current allocated: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB")

    # Run profiling analyses
    try:
        # 1. PyTorch profiler with detailed timeline
        profile_with_pytorch_profiler(obs_dict, policy, device, num_runs=10)

        # 2. Synchronization overhead
        profile_synchronization_overhead(obs_dict, policy, device, num_runs=20)

        # 3. Individual kernel analysis
        profile_individual_kernels(obs_dict, policy, device)

        # 4. Memory transfer patterns
        check_memory_transfer_patterns(obs_dict, policy, device, num_runs=10)

    except Exception as e:
        print(f"\nError during profiling: {e}")
        import traceback
        traceback.print_exc()

    print("\n" + "="*80)
    print("PROFILING COMPLETE")
    print("="*80)

    print("""
Next steps:
1. Check the Chrome trace file (open chrome://tracing and load the .json file)
   - This shows a timeline of all operations
   - Look for gaps or long-running kernels

2. Check TensorBoard logs:
   tensorboard --logdir=./profiler_logs

3. For even more detailed analysis, run with NVIDIA Nsight Systems:
   nsys profile -o inference_profile python profile_inference.py

4. To analyze the nsys output:
   nsys-ui inference_profile.qdrep
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
