#!/bin/bash

# Script to profile diffusion policy inference using NVIDIA Nsight Systems
# This provides the most detailed GPU profiling including:
# - CUDA API calls
# - Kernel execution timeline
# - Memory transfers
# - CPU-GPU synchronization points
# - CUDA stream activity

echo "=================================="
echo "NVIDIA Nsight Systems Profiling"
echo "=================================="

# Check if nsys is available
if ! command -v nsys &> /dev/null; then
    echo "Error: nsys (NVIDIA Nsight Systems) not found!"
    echo ""
    echo "Install with:"
    echo "  sudo apt install nvidia-nsight-systems"
    echo "  or download from: https://developer.nvidia.com/nsight-systems"
    exit 1
fi

# Create output directory
OUTPUT_DIR="./nsight_profiles"
mkdir -p $OUTPUT_DIR

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
OUTPUT_FILE="${OUTPUT_DIR}/inference_profile_${TIMESTAMP}"

echo ""
echo "Running profiler..."
echo "Output will be saved to: ${OUTPUT_FILE}.nsys-rep"
echo ""

# Run nsight systems profiler
# Options:
#   -o: output file
#   --trace: what to trace (cuda, nvtx, osrt, cudnn, cublas)
#   --cuda-memory-usage: track CUDA memory allocations
#   --gpu-metrics-device: which GPU to profile
#   --duration: how long to profile (in seconds)
#   --delay: delay before starting profiling
#   --sample: CPU sampling mode

nsys profile \
    -o ${OUTPUT_FILE} \
    --trace=cuda,nvtx,osrt,cudnn,cublas \
    --cuda-memory-usage=true \
    --gpu-metrics-device=0 \
    --force-overwrite=true \
    --capture-range=cudaProfilerApi \
    --capture-range-end=stop \
    python profile_inference.py

echo ""
echo "=================================="
echo "Profiling complete!"
echo "=================================="

echo ""
echo "Output files:"
ls -lh ${OUTPUT_FILE}.*

echo ""
echo "To analyze the results:"
echo ""
echo "1. Generate a report (text):"
echo "   nsys stats ${OUTPUT_FILE}.nsys-rep"
echo ""
echo "2. View in GUI (if available):"
echo "   nsys-ui ${OUTPUT_FILE}.nsys-rep"
echo ""
echo "3. Export to SQLite for custom analysis:"
echo "   nsys export --type=sqlite ${OUTPUT_FILE}.nsys-rep"
echo ""
echo "4. Generate a summary:"
echo "   nsys stats --report cuda_gpu_kern_sum ${OUTPUT_FILE}.nsys-rep"
echo "   nsys stats --report cuda_gpu_mem_time_sum ${OUTPUT_FILE}.nsys-rep"
echo "   nsys stats --report cuda_api_sum ${OUTPUT_FILE}.nsys-rep"
echo ""
