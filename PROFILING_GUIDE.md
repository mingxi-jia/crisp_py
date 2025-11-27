# GPU Profiling Guide for Diffusion Policy Inference

This guide helps you diagnose inference time variability (0.2s to 5s) in your diffusion policy.

## Quick Start

```bash
cd /home/mingxi/mingxi_ws/crisp/crisp_py/examples
./run_profiling.sh
```

Then select option 1 (Quick kernel profile) to get started.

---

## Problem Summary

Based on the initial diagnostic (`debug_inference_timing.py`), we found:

- **Inference time varies**: 69ms to 82ms (~18% variation)
- **Root cause**: GPU frequency scaling and power management
- **97.8% of variance** comes from the `predict_action` kernel execution
- **GPU utilization varies**: 14% to 57% (suggesting dynamic clock scaling)

---

## Profiling Tools

### 1. Quick Kernel Profile (Recommended First Step)

**File**: `quick_kernel_profile.py`

**Run**:
```bash
python quick_kernel_profile.py
```

**What it does**:
- Shows which CUDA kernels are slowest
- Identifies kernels with high execution time variance
- Analyzes operation types (convolution, matmul, attention, etc.)
- **Runtime**: ~2 minutes

**Use when**: You want a fast overview of what's slow

---

### 2. Detailed PyTorch Profiler

**File**: `profile_inference.py`

**Run**:
```bash
python profile_inference.py
```

**What it does**:
- Complete timeline analysis of CPU and GPU operations
- Memory transfer analysis (cudaMemcpy)
- Synchronization overhead measurement
- Exports Chrome trace for visualization
- Exports TensorBoard logs

**Outputs**:
- `inference_trace_<timestamp>.json` - Load in Chrome at `chrome://tracing`
- `./profiler_logs/` - View with TensorBoard

**View results**:
```bash
# Chrome trace (visual timeline)
# 1. Open Chrome browser
# 2. Go to: chrome://tracing
# 3. Load the .json file
# 4. Use WASD to navigate, mouse wheel to zoom

# TensorBoard
tensorboard --logdir=./profiler_logs
# Then open http://localhost:6006 in browser
```

**Runtime**: ~5 minutes

**Use when**: You need detailed timeline and want to see exact ordering of operations

---

### 3. NVIDIA Nsight Systems (Most Detailed)

**File**: `nsight_profile.sh`

**Prerequisites**:
```bash
sudo apt install nvidia-nsight-systems
```

**Run**:
```bash
./nsight_profile.sh
```

**What it does**:
- Professional-grade GPU profiling
- Shows exact CUDA API calls, kernel launches, memory transfers
- Stream activity and dependencies
- CPU-GPU synchronization points
- Can identify scheduling issues

**Outputs**:
- `./nsight_profiles/inference_profile_<timestamp>.nsys-rep`

**View results**:
```bash
# Generate text report
nsys stats inference_profile_<timestamp>.nsys-rep

# Kernel summary
nsys stats --report cuda_gpu_kern_sum inference_profile_<timestamp>.nsys-rep

# Memory operations summary
nsys stats --report cuda_gpu_mem_time_sum inference_profile_<timestamp>.nsys-rep

# GUI (if available)
nsys-ui inference_profile_<timestamp>.nsys-rep
```

**Runtime**: ~5 minutes

**Use when**: You need the absolute most detailed analysis or suspect driver/scheduling issues

---

## What to Look For

### 1. Kernel Execution Variance

**Look for**:
- Kernels with high coefficient of variation (CV > 20%)
- Same kernel taking different amounts of time across runs

**Indicates**:
- GPU frequency scaling (most likely)
- Cache effects
- Thermal throttling

**Solution**: Lock GPU to max performance (see below)

---

### 2. Memory Transfer Bottlenecks

**Look for**:
- `cudaMemcpy` operations taking significant time
- H2D (Host-to-Device) or D2H (Device-to-Host) transfers

**Indicates**:
- PCIe bottleneck
- Inefficient data transfer patterns

**Solution**:
- Pin memory: `torch.cuda.empty_cache()` before runs
- Use pinned memory for transfers
- Batch transfers

---

### 3. Synchronization Overhead

**Look for**:
- `cudaDeviceSynchronize` calls
- Stream synchronization
- Large gaps between kernel launches

**Indicates**:
- CPU waiting for GPU
- Poor kernel launch scheduling
- Unnecessary synchronization

**Solution**:
- Remove unnecessary `torch.cuda.synchronize()` calls
- Use CUDA streams for parallelism
- Overlap compute with data transfer

---

### 4. Specific Slow Kernels

**Look for**:
- A few dominant kernels taking most of the time
- Kernels related to specific operations (e.g., attention, convolution)

**Indicates**:
- Algorithmic bottleneck
- Inefficient implementation

**Solution**:
- Optimize those specific operations
- Use fused kernels (e.g., FlashAttention for attention)
- Consider model architecture changes

---

## Fixing GPU Frequency Scaling (Primary Issue)

Based on our initial analysis, GPU frequency scaling is the main culprit.

### Method 1: Lock GPU to Max Performance (Recommended)

```bash
# Check current GPU status
nvidia-smi

# Enable persistence mode (keeps GPU initialized)
sudo nvidia-smi -pm 1

# Set power limit to maximum
sudo nvidia-smi -pl <max_power_limit>
# Example: sudo nvidia-smi -pl 250

# Lock GPU clock to maximum frequency
sudo nvidia-smi -lgc <max_clock>
# Example: sudo nvidia-smi -lgc 2100

# Verify
nvidia-smi
```

### Method 2: Disable GPU Power Management

Create `/etc/modprobe.d/nvidia-power.conf`:
```
options nvidia NVreg_RegistryDwords="PowerMizerEnable=0x1;PerfLevelSrc=0x2222;PowerMizerDefault=0x1;PowerMizerDefaultAC=0x1"
```

Then:
```bash
sudo update-initramfs -u
sudo reboot
```

### Method 3: Set in Code (Less Reliable)

```python
import torch

# Set CUDA to high performance mode
torch.backends.cudnn.benchmark = False  # Disable auto-tuning
torch.backends.cudnn.deterministic = True  # Ensure consistent execution

# Warm up thoroughly
for _ in range(20):
    # Run inference
    pass

# Then start actual inference
```

---

## Verification

After applying fixes, re-run the diagnostic:

```bash
python debug_inference_timing.py
```

**Look for**:
- GPU utilization variance should be < 5%
- Inference time std deviation should be < 2ms
- All runs should cluster around the same time

---

## Expected Results

### Before Fix
- Mean: 73.5ms, Std: 3.4ms, Range: 69-82ms
- GPU util: 14%-57% (43% variation)
- CV: 4.6%

### After Fix (Expected)
- Mean: ~70ms, Std: <1ms, Range: 69-71ms
- GPU util: 95%-99% (consistent)
- CV: <1%

---

## Troubleshooting

### Issue: nsys not found
```bash
sudo apt update
sudo apt install nvidia-nsight-systems
```

### Issue: Permission denied for nvidia-smi
```bash
# Add yourself to video group
sudo usermod -a -G video $USER
# Log out and back in
```

### Issue: GPU clock won't lock
```bash
# Check if persistence mode is enabled
nvidia-smi -q | grep "Persistence Mode"

# Some GPUs don't support clock locking
# Use power limit only:
sudo nvidia-smi -pm 1
sudo nvidia-smi -pl <max_power>
```

### Issue: TensorBoard won't start
```bash
pip install tensorboard
tensorboard --logdir=./profiler_logs --port=6006
```

---

## Advanced: Analyzing Chrome Trace

1. Run: `python profile_inference.py`
2. Open Chrome: `chrome://tracing`
3. Load the `inference_trace_*.json` file

**Navigation**:
- `W/S`: Zoom in/out
- `A/D`: Pan left/right
- Click and drag: Select time range
- `?`: Help

**What to look for**:
- **Gaps between kernels**: Indicates synchronization or scheduling overhead
- **Kernel duration variance**: Same kernel taking different times
- **Memory operations**: Look for `cudaMemcpy` bars
- **Overlap**: Good execution has overlapping operations

---

## Summary

1. **Start with**: `python quick_kernel_profile.py` (fastest)
2. **If you need more detail**: `python profile_inference.py`
3. **For maximum detail**: `./nsight_profile.sh`
4. **Fix the main issue**: Lock GPU to max performance
5. **Verify**: Re-run `python debug_inference_timing.py`

The profiling will show you exactly which operations are slow and variable, allowing you to pinpoint the issue beyond just "GPU frequency scaling" to specific kernels or patterns.
