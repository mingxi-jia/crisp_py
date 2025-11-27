"""Visualize profiling results to understand inference time variability."""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# Load existing timing data
try:
    data = np.load('inference_timing_analysis.npz')
    has_data = True
except FileNotFoundError:
    print("No timing data found. Run debug_inference_timing.py first.")
    has_data = False
    exit(1)

# Extract data
total = data['total']
dict_apply = data['dict_apply']
predict_action = data['predict_action']
to_cpu = data['to_cpu']
gpu_util_before = data['gpu_util_before']
mem_reserved = data['mem_reserved']

# Create comprehensive visualization
fig = plt.figure(figsize=(16, 12))
gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)

# 1. Inference time over iterations
ax1 = fig.add_subplot(gs[0, :2])
iterations = np.arange(len(predict_action))
ax1.plot(iterations, predict_action, 'o-', linewidth=2, markersize=6, label='predict_action', alpha=0.7)
ax1.axhline(np.mean(predict_action), color='red', linestyle='--', label=f'Mean: {np.mean(predict_action):.2f}ms', linewidth=2)
ax1.axhline(np.mean(predict_action) + 2*np.std(predict_action), color='orange', linestyle=':', label='±2σ', linewidth=1.5)
ax1.axhline(np.mean(predict_action) - 2*np.std(predict_action), color='orange', linestyle=':', linewidth=1.5)
ax1.set_xlabel('Iteration', fontsize=12)
ax1.set_ylabel('Inference Time (ms)', fontsize=12)
ax1.set_title('Inference Time Across Runs\n(Looking for variance and patterns)', fontsize=14, fontweight='bold')
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3)

# 2. GPU utilization over iterations
ax2 = fig.add_subplot(gs[0, 2])
ax2.plot(iterations, gpu_util_before, 'o-', color='green', linewidth=2, markersize=5, alpha=0.7)
ax2.axhline(np.mean(gpu_util_before), color='red', linestyle='--', linewidth=2)
ax2.set_xlabel('Iteration', fontsize=12)
ax2.set_ylabel('GPU Util (%)', fontsize=12)
ax2.set_title('GPU Utilization\nBefore Inference', fontsize=12, fontweight='bold')
ax2.grid(True, alpha=0.3)
ax2.set_ylim([0, 100])

# 3. Distribution of inference times
ax3 = fig.add_subplot(gs[1, 0])
ax3.hist(predict_action, bins=20, edgecolor='black', alpha=0.7, color='steelblue')
ax3.axvline(np.mean(predict_action), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(predict_action):.2f}ms')
ax3.axvline(np.median(predict_action), color='orange', linestyle='--', linewidth=2, label=f'Median: {np.median(predict_action):.2f}ms')
ax3.set_xlabel('Inference Time (ms)', fontsize=12)
ax3.set_ylabel('Frequency', fontsize=12)
ax3.set_title('Distribution of\nInference Times', fontsize=12, fontweight='bold')
ax3.legend(fontsize=9)
ax3.grid(True, alpha=0.3, axis='y')

# 4. Component breakdown
ax4 = fig.add_subplot(gs[1, 1])
components = ['dict_apply\n(CPU→GPU)', 'predict_action\n(Inference)', 'to_cpu\n(GPU→CPU)']
means = [np.mean(dict_apply), np.mean(predict_action), np.mean(to_cpu)]
stds = [np.std(dict_apply), np.std(predict_action), np.std(to_cpu)]
colors = ['lightblue', 'steelblue', 'lightcoral']

bars = ax4.bar(components, means, yerr=stds, capsize=5, color=colors, edgecolor='black', linewidth=1.5, alpha=0.8)
ax4.set_ylabel('Time (ms)', fontsize=12)
ax4.set_title('Time Breakdown\nby Component', fontsize=12, fontweight='bold')
ax4.grid(True, alpha=0.3, axis='y')

# Add value labels on bars
for bar, mean in zip(bars, means):
    height = bar.get_height()
    ax4.text(bar.get_x() + bar.get_width()/2., height,
             f'{mean:.2f}ms',
             ha='center', va='bottom', fontsize=10, fontweight='bold')

# 5. GPU util vs inference time scatter
ax5 = fig.add_subplot(gs[1, 2])
scatter = ax5.scatter(gpu_util_before, predict_action, c=iterations, cmap='viridis',
                     s=100, alpha=0.6, edgecolor='black', linewidth=0.5)
ax5.set_xlabel('GPU Utilization Before (%)', fontsize=12)
ax5.set_ylabel('Inference Time (ms)', fontsize=12)
ax5.set_title('GPU Util vs\nInference Time', fontsize=12, fontweight='bold')
ax5.grid(True, alpha=0.3)

# Add correlation coefficient
from scipy.stats import pearsonr
corr, p_value = pearsonr(gpu_util_before, predict_action)
ax5.text(0.05, 0.95, f'r = {corr:+.3f}\np = {p_value:.4f}',
         transform=ax5.transAxes, fontsize=10, verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.colorbar(scatter, ax=ax5, label='Iteration')

# 6. Variance contribution pie chart
ax6 = fig.add_subplot(gs[2, 0])
variances = [np.var(dict_apply), np.var(predict_action), np.var(to_cpu)]
total_var = sum(variances)
percentages = [v/total_var*100 for v in variances]
labels = [f'{comp}\n({pct:.1f}%)' for comp, pct in zip(['dict_apply', 'predict_action', 'to_cpu'], percentages)]

ax6.pie(percentages, labels=labels, autopct='', colors=colors, startangle=90,
        explode=(0.05, 0.1, 0.05), shadow=True)
ax6.set_title('Variance Contribution\n(Where does variability come from?)', fontsize=12, fontweight='bold')

# 7. Statistics summary
ax7 = fig.add_subplot(gs[2, 1:])
ax7.axis('off')

stats_text = f"""
SUMMARY STATISTICS

Inference Time (predict_action):
  Mean:    {np.mean(predict_action):6.2f} ms
  Median:  {np.median(predict_action):6.2f} ms
  Std Dev: {np.std(predict_action):6.2f} ms
  Min:     {np.min(predict_action):6.2f} ms
  Max:     {np.max(predict_action):6.2f} ms
  Range:   {np.max(predict_action) - np.min(predict_action):6.2f} ms
  CV:      {np.std(predict_action)/np.mean(predict_action)*100:6.2f}%

GPU Utilization (before inference):
  Mean:    {np.mean(gpu_util_before):6.1f}%
  Range:   {np.min(gpu_util_before):6.1f}% - {np.max(gpu_util_before):6.1f}%
  Std Dev: {np.std(gpu_util_before):6.1f}%

Correlation:
  GPU Util ↔ Inference Time: r = {corr:+.3f} (p = {p_value:.4f})

KEY FINDING:
  • {percentages[1]:.1f}% of variance comes from predict_action
  • GPU utilization varies by {np.max(gpu_util_before) - np.min(gpu_util_before):.0f}%
  • Likely cause: GPU frequency scaling
"""

ax7.text(0.1, 0.9, stats_text, transform=ax7.transAxes,
         fontsize=11, verticalalignment='top', family='monospace',
         bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))

# Main title
fig.suptitle('Diffusion Policy Inference Time Analysis\n(Diagnosing 0.2s to 5s Variability)',
             fontsize=16, fontweight='bold', y=0.995)

# Save figure
output_file = 'profiling_visualization.png'
plt.savefig(output_file, dpi=150, bbox_inches='tight')
print(f"\n✓ Visualization saved to: {output_file}")

# Show plot
try:
    plt.show()
except:
    print("  (Display not available, but image saved)")

print("\n" + "="*80)
print("INTERPRETATION GUIDE")
print("="*80)
print("""
What to look for in the visualization:

1. TOP LEFT: Inference Time Across Runs
   - Look for: Patterns, trends, or random variation
   - If random with high variance → GPU frequency scaling
   - If increasing/decreasing → thermal or memory issues
   - If periodic → system interference

2. TOP MIDDLE: Component Breakdown
   - Shows where time is spent
   - predict_action should dominate (you're seeing ~73ms)
   - If dict_apply is high → memory transfer bottleneck

3. TOP RIGHT: GPU Utilization
   - Should be consistently high (95-99%)
   - If varying widely (yours: 14-57%) → GPU clock scaling
   - Low utilization → GPU not being fully used

4. MIDDLE LEFT: Distribution
   - Should be tight bell curve
   - Wide distribution → high variance (your case)
   - Bimodal → two performance states

5. MIDDLE RIGHT: GPU Util vs Inference Time
   - Positive correlation → GPU scaling affects performance
   - Your r=+0.155 suggests weak correlation
   - But GPU util range (43%) is the issue

6. BOTTOM LEFT: Variance Contribution
   - Shows which component has most variability
   - 97.8% from predict_action → problem is in GPU kernels

YOUR CASE:
✗ High variance in inference time (13ms range)
✗ Wide GPU utilization range (14-57%)
✗ 97.8% variance from predict_action
→ DIAGNOSIS: GPU frequency scaling

SOLUTION:
1. Lock GPU to max performance (fix_gpu_performance.sh)
2. Re-run diagnostic
3. Should see: tight distribution, consistent GPU util 95-99%
""")

print("\n" + "="*80)
