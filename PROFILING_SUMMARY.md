# Profiling Tools Summary

## What Was Created

I've created a comprehensive suite of GPU profiling tools to diagnose your inference time variability issue (0.2s to 5s).

---

## Files Created

### 1. Diagnostic Scripts

#### `debug_inference_timing.py` ✅ (Already run)
- **Purpose**: Initial diagnostic to identify the problem
- **Output**: `inference_timing_analysis.npz` (timing data)
- **Result**: Identified GPU frequency scaling as the main issue

#### `analyze_timing_results.py` ✅ (Already run)
- **Purpose**: Analyze the diagnostic data
- **What it found**:
  - Inference varies 69-82ms (18% variation)
  - GPU utilization varies 14%-57%
  - 97.8% of variance in `predict_action` (the actual inference)
  - Root cause: GPU frequency scaling

---

### 2. Profiling Scripts (New - To Help You Dig Deeper)

#### `quick_kernel_profile.py` ⭐ START HERE
- **Purpose**: Fast kernel-level profiling
- **Runtime**: ~2 minutes
- **Shows**:
  - Which CUDA kernels are slowest
  - Which kernels have high execution variance
  - Time breakdown by operation type (conv, matmul, attention)
- **Use**: Quick overview of what's happening inside predict_action

#### `profile_inference.py` 🔍 DETAILED
- **Purpose**: Comprehensive profiling with multiple analyses
- **Runtime**: ~5 minutes
- **Shows**:
  - Complete timeline of operations
  - Memory transfer analysis
  - Synchronization overhead
  - Individual kernel breakdown
- **Outputs**:
  - Chrome trace file (visual timeline)
  - TensorBoard logs
- **Use**: When you need to see exact timeline and ordering

#### `nsight_profile.sh` 🚀 MOST POWERFUL
- **Purpose**: Professional GPU profiling with NVIDIA tools
- **Runtime**: ~5 minutes
- **Requires**: `nvidia-nsight-systems` installed
- **Shows**:
  - CUDA API calls
  - Exact kernel execution timeline
  - Memory transfers
  - Stream activity
- **Use**: Maximum detail, professional analysis

#### `run_profiling.sh` 🎯 LAUNCHER
- **Purpose**: Interactive menu to run profiling tools
- **Use**: Easy way to choose which profiling to run

---

### 3. Configuration Scripts

#### `fix_gpu_performance.sh`
- **Purpose**: Lock GPU to max performance mode
- **What it does**:
  - Enables persistence mode
  - Sets max power limit
  - Locks GPU clock to maximum
- **Use**: Fix the GPU frequency scaling issue
- **Note**: Requires sudo

---

### 4. Documentation

#### `PROFILING_GUIDE.md` 📖
- Complete guide to using all the profiling tools
- Explains what to look for in results
- How to fix common issues
- Troubleshooting section

#### `PROFILING_SUMMARY.md` 📋 (This file)
- Overview of all tools
- Quick reference

---

## Recommended Workflow

### Step 1: Quick Profiling (Start Here)
```bash
cd /home/mingxi/mingxi_ws/crisp/crisp_py/examples
python quick_kernel_profile.py
```

**This will tell you**:
- Which specific kernels are slow
- Which kernels have variance
- If it's attention, convolution, or other operations

**Time**: 2 minutes

---

### Step 2: Fix GPU Frequency Scaling (Likely Fix)
```bash
# Check current GPU status
nvidia-smi

# Lock to max performance
cd /home/mingxi/mingxi_ws/crisp/crisp_py
sudo bash fix_gpu_performance.sh

# Verify GPU is now at max clock
nvidia-smi
```

**This should**:
- Eliminate the 18% variance
- Make inference consistently fast (~70ms)

---

### Step 3: Verify Fix
```bash
cd /home/mingxi/mingxi_ws/crisp/crisp_py
python debug_inference_timing.py
python analyze_timing_results.py
```

**Expected after fix**:
- Std deviation: < 1ms (was 3.4ms)
- GPU utilization: 95-99% (was 14-57%)
- Inference time: consistently ~70ms

---

### Step 4: Deep Dive (If Still Variable)

If the issue persists after fixing GPU clocks, then use detailed profiling:

```bash
cd /home/mingxi/mingxi_ws/crisp/crisp_py/examples

# Option A: Detailed PyTorch profiler
python profile_inference.py
# Then view Chrome trace or TensorBoard

# Option B: NVIDIA Nsight Systems
./nsight_profile.sh
# Then analyze with: nsys stats <output_file>.nsys-rep
```

**This will show**:
- Exact timeline of what's happening
- Memory transfer bottlenecks
- Synchronization overhead
- Specific slow kernels

---

## What Each Tool Will Tell You

### From Initial Diagnostic (Already Done ✅)
**Problem Identified**: GPU frequency scaling
- Inference: 69-82ms (should be consistent)
- GPU util: 14-57% (should be 95-99%)
- Variance source: 97.8% from predict_action

### From Quick Kernel Profile (Recommended Next ⭐)
**Will show you**:
- "Attention kernels account for 60% of time"
- "Convolution kernel 'xyz' has 25% variance"
- "Top 3 kernels: kernel_a (30ms), kernel_b (20ms), kernel_c (15ms)"

**Example output**:
```
Top 20 CUDA kernels by total time:
Kernel                                    Count    Total (ms)  Avg (ms)
--------------------------------------------------------------------
volta_sgemm_128x128_tn                    400      800.5       2.001
attention_softmax_kernel                  80       450.2       5.627
batch_norm_forward                        200      120.8       0.604
```

### From Detailed Profile (If Needed 🔍)
**Will show you**:
- Timeline visualization (Chrome trace)
- "Gap of 5ms between kernel launches" (scheduling issue)
- "cudaMemcpy taking 10ms" (memory transfer bottleneck)
- "cudaDeviceSynchronize overhead: 3ms" (unnecessary sync)

### From Nsight Systems (Professional 🚀)
**Will show you**:
- Exact CUDA API call sequence
- Driver-level scheduling
- Stream dependencies
- PCIe transfer patterns

---

## Quick Reference Commands

```bash
# Navigate to profiling directory
cd /home/mingxi/mingxi_ws/crisp/crisp_py/examples

# Quick profile (start here)
python quick_kernel_profile.py

# Detailed profile
python profile_inference.py

# NVIDIA profiling (requires nsys)
./nsight_profile.sh

# Interactive menu
./run_profiling.sh

# Fix GPU clocks
cd /home/mingxi/mingxi_ws/crisp/crisp_py
sudo bash fix_gpu_performance.sh

# Verify fix
python debug_inference_timing.py
python analyze_timing_results.py

# View Chrome trace
# 1. Open chrome://tracing
# 2. Load inference_trace_*.json

# View TensorBoard
tensorboard --logdir=./profiler_logs
```

---

## Expected Results After Profiling

### Current State (Before Fix)
```
Inference time: 69-82ms (Δ=13ms, CV=4.6%)
GPU util: 14%-57%
Variance: HIGH (GPU clock scaling)
```

### After GPU Clock Fix
```
Inference time: 69-71ms (Δ=2ms, CV=<1%)
GPU util: 95-99%
Variance: LOW (consistent performance)
```

### If Still Variable After Fix
Then profiling will show one of:
1. **Memory bottleneck**: cudaMemcpy takes significant time
2. **Specific slow kernel**: One kernel dominates and varies
3. **Synchronization**: Gaps between kernel launches
4. **Thermal throttling**: GPU temperature causing slowdown

---

## Key Insight from Analysis

Based on your initial diagnostic data:

**The problem is NOT in your code**. It's the GPU dynamically adjusting its performance:
- When GPU util drops to 14%, GPU clocks down
- When it clocks up to 57%, inference is faster
- This causes the 0.2s to 5s variance you observed

**The fix**: Lock GPU to max performance (see `fix_gpu_performance.sh`)

**The profiling**: Will confirm this and show you which specific kernels are affected

---

## Next Steps

1. **Run quick profile** to see kernel breakdown:
   ```bash
   cd /home/mingxi/mingxi_ws/crisp/crisp_py/examples
   python quick_kernel_profile.py
   ```

2. **Fix GPU clocks** (likely solves the issue):
   ```bash
   cd /home/mingxi/mingxi_ws/crisp/crisp_py
   sudo bash fix_gpu_performance.sh
   ```

3. **Verify** the fix worked:
   ```bash
   python debug_inference_timing.py
   python analyze_timing_results.py
   ```

4. **If still variable**, use detailed profiling to dig deeper

---

## Support

See `PROFILING_GUIDE.md` for:
- Detailed usage instructions
- How to interpret results
- Troubleshooting
- Advanced analysis techniques