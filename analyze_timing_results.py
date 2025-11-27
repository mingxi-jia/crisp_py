"""Analyze the timing results from inference_timing_analysis.npz"""

import numpy as np
import matplotlib.pyplot as plt

# Load the data
data = np.load('inference_timing_analysis.npz')

print("="*80)
print("INFERENCE TIMING ANALYSIS RESULTS")
print("="*80)

# Extract timing arrays
total = data['total']
dict_apply = data['dict_apply']
predict_action = data['predict_action']
to_cpu = data['to_cpu']
gpu_util_before = data['gpu_util_before']
gpu_util_after = data['gpu_util_after']
mem_allocated = data['mem_allocated']
mem_reserved = data['mem_reserved']

print("\n" + "="*80)
print("SUMMARY STATISTICS (all times in milliseconds)")
print("="*80)

def print_stats(name, times):
    print(f"\n{name}:")
    print(f"  Mean:   {np.mean(times):8.2f} ms")
    print(f"  Median: {np.median(times):8.2f} ms")
    print(f"  Min:    {np.min(times):8.2f} ms")
    print(f"  Max:    {np.max(times):8.2f} ms")
    print(f"  Std:    {np.std(times):8.2f} ms")
    print(f"  Range:  {np.max(times) - np.min(times):8.2f} ms")
    print(f"  CV:     {np.std(times)/np.mean(times)*100:8.2f}% (coefficient of variation)")

    # Identify outliers
    mean = np.mean(times)
    std = np.std(times)
    outliers = [(i, t) for i, t in enumerate(times) if abs(t - mean) > 2 * std]
    if outliers:
        print(f"  Outliers (>2σ): {len(outliers)}")
        for idx, val in outliers[:5]:  # Show first 5
            print(f"    Run {idx}: {val:.2f} ms (Δ={val-mean:+.2f})")

print_stats("TOTAL TIME", total)
print_stats("dict_apply (CPU->GPU)", dict_apply)
print_stats("predict_action (INFERENCE)", predict_action)
print_stats("to_cpu (GPU->CPU)", to_cpu)

# Analyze the variance contribution
print("\n" + "="*80)
print("VARIANCE ANALYSIS")
print("="*80)

total_variance = np.var(total)
dict_apply_variance = np.var(dict_apply)
predict_variance = np.var(predict_action)
to_cpu_variance = np.var(to_cpu)

print(f"\nVariance contribution:")
print(f"  Total variance:        {total_variance:10.2f}")
print(f"  dict_apply variance:   {dict_apply_variance:10.2f} ({dict_apply_variance/total_variance*100:5.1f}%)")
print(f"  predict_action var:    {predict_variance:10.2f} ({predict_variance/total_variance*100:5.1f}%)")
print(f"  to_cpu variance:       {to_cpu_variance:10.2f} ({to_cpu_variance/total_variance*100:5.1f}%)")

# Find which component has the highest variability
components = {
    'dict_apply': dict_apply,
    'predict_action': predict_action,
    'to_cpu': to_cpu
}

cv_scores = {name: np.std(times)/np.mean(times)*100 for name, times in components.items()}
most_variable = max(cv_scores, key=cv_scores.get)

print(f"\n** MOST VARIABLE COMPONENT: {most_variable} (CV={cv_scores[most_variable]:.2f}%)")

# GPU utilization analysis
print("\n" + "="*80)
print("GPU UTILIZATION ANALYSIS")
print("="*80)

print(f"\nGPU Utilization BEFORE inference:")
print(f"  Mean:   {np.mean(gpu_util_before):6.1f}%")
print(f"  Range:  {np.min(gpu_util_before):6.1f}% - {np.max(gpu_util_before):6.1f}%")
print(f"  Std:    {np.std(gpu_util_before):6.1f}%")

print(f"\nGPU Utilization AFTER inference:")
print(f"  Mean:   {np.mean(gpu_util_after):6.1f}%")
print(f"  Range:  {np.min(gpu_util_after):6.1f}% - {np.max(gpu_util_after):6.1f}%")
print(f"  Std:    {np.std(gpu_util_after):6.1f}%")

# Check correlation between GPU util and inference time
from scipy.stats import pearsonr

try:
    corr_before, p_before = pearsonr(gpu_util_before, predict_action)
    corr_after, p_after = pearsonr(gpu_util_after, predict_action)

    print(f"\nCorrelation between GPU utilization and inference time:")
    print(f"  Before: r={corr_before:+.3f} (p={p_before:.4f})")
    print(f"  After:  r={corr_after:+.3f} (p={p_after:.4f})")
except:
    print("\nCould not compute correlation (scipy not available)")

# Memory analysis
print("\n" + "="*80)
print("MEMORY ANALYSIS")
print("="*80)

print(f"\nMemory allocated (delta per inference):")
print(f"  Mean:   {np.mean(mem_allocated):8.2f} MB")
print(f"  Max:    {np.max(mem_allocated):8.2f} MB")
print(f"  Min:    {np.min(mem_allocated):8.2f} MB")

print(f"\nMemory reserved (total):")
print(f"  Mean:   {np.mean(mem_reserved):8.2f} MB")
print(f"  Range:  {np.min(mem_reserved):8.2f} - {np.max(mem_reserved):8.2f} MB")

# Pattern analysis - check if there's a temporal pattern
print("\n" + "="*80)
print("TEMPORAL PATTERN ANALYSIS")
print("="*80)

# Split into first half and second half
n = len(predict_action)
first_half = predict_action[:n//2]
second_half = predict_action[n//2:]

print(f"\nFirst half ({len(first_half)} runs):")
print(f"  Mean: {np.mean(first_half):8.2f} ms")
print(f"  Std:  {np.std(first_half):8.2f} ms")

print(f"\nSecond half ({len(second_half)} runs):")
print(f"  Mean: {np.mean(second_half):8.2f} ms")
print(f"  Std:  {np.std(second_half):8.2f} ms")

# Check for runs with significant slowdowns
print("\n" + "="*80)
print("SLOWEST RUNS (predict_action component)")
print("="*80)

# Find slowest 10 runs
slowest_indices = np.argsort(predict_action)[-10:][::-1]
print(f"\nTop 10 slowest inference runs:")
print(f"{'Run':<5} {'Time (ms)':<12} {'GPU Before':<12} {'GPU After':<12}")
print("-" * 45)
for idx in slowest_indices:
    print(f"{idx:<5} {predict_action[idx]:8.2f} ms   {gpu_util_before[idx]:6.1f}%      {gpu_util_after[idx]:6.1f}%")

# Find fastest 10 runs
fastest_indices = np.argsort(predict_action)[:10]
print(f"\nTop 10 fastest inference runs:")
print(f"{'Run':<5} {'Time (ms)':<12} {'GPU Before':<12} {'GPU After':<12}")
print("-" * 45)
for idx in fastest_indices:
    print(f"{idx:<5} {predict_action[idx]:8.2f} ms   {gpu_util_before[idx]:6.1f}%      {gpu_util_after[idx]:6.1f}%")

# ROOT CAUSE ANALYSIS
print("\n" + "="*80)
print("ROOT CAUSE ANALYSIS")
print("="*80)

# Calculate time range
time_range = np.max(predict_action) - np.min(predict_action)
mean_time = np.mean(predict_action)

print(f"\nInference time varies from {np.min(predict_action):.0f}ms to {np.max(predict_action):.0f}ms")
print(f"Range: {time_range:.0f}ms ({time_range/mean_time*100:.1f}% of mean)")

# Check if GPU utilization varies
gpu_util_range = np.max(gpu_util_before) - np.min(gpu_util_before)

print(f"\nGPU utilization varies from {np.min(gpu_util_before):.0f}% to {np.max(gpu_util_before):.0f}%")
print(f"Range: {gpu_util_range:.0f}%")

print("\n" + "="*80)
print("LIKELY CAUSES (ranked by evidence):")
print("="*80)

causes = []

# Check GPU utilization variation
if gpu_util_range > 20:
    causes.append(("GPU frequency scaling / power management",
                   f"GPU utilization varies by {gpu_util_range:.0f}%, suggesting GPU is scaling clock speeds"))

# Check predict_action variance
if cv_scores['predict_action'] > 20:
    causes.append(("Inference kernel variability",
                   f"predict_action has CV={cv_scores['predict_action']:.1f}%, much higher than data transfer"))

# Check memory allocation
if np.any(mem_allocated != 0):
    causes.append(("Memory allocation overhead",
                   "Some runs show memory allocation, suggesting dynamic allocation during inference"))

# Check temporal pattern
if abs(np.mean(first_half) - np.mean(second_half)) > 50:
    causes.append(("Warmup effect",
                   f"First half mean: {np.mean(first_half):.0f}ms, Second half: {np.mean(second_half):.0f}ms"))

# Check GPU utilization bins
unique_utils = np.unique(gpu_util_before)
if len(unique_utils) > 3:
    causes.append(("Multiple GPU performance states",
                   f"GPU operating at {len(unique_utils)} distinct utilization levels: {sorted(unique_utils)}%"))

for i, (cause, evidence) in enumerate(causes, 1):
    print(f"\n{i}. {cause}")
    print(f"   Evidence: {evidence}")

# Recommendations
print("\n" + "="*80)
print("RECOMMENDATIONS TO FIX THE ISSUE:")
print("="*80)

print("""
1. **Lock GPU to maximum performance mode:**
   sudo nvidia-smi -pm 1
   sudo nvidia-smi -lgc <max_clock>

2. **Disable GPU power management:**
   sudo nvidia-smi -pl <max_power_limit>

3. **Set persistence mode:**
   sudo nvidia-smi -pm ENABLED

4. **In your code, warm up the policy properly:**
   - Run 10-20 warmup iterations before timing
   - Use torch.cuda.synchronize() before/after timing

5. **Use torch.backends.cudnn.benchmark = False:**
   - This disables auto-tuning which can cause variability

6. **Check for other GPU processes:**
   nvidia-smi

7. **Pin CPU cores and set CPU governor to performance:**
   sudo cpupower frequency-set -g performance
""")

print("\n" + "="*80)
print("END OF ANALYSIS")
print("="*80)
