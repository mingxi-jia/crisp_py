#!/bin/bash

# Helper script to run various profiling analyses

echo "=================================="
echo "GPU Profiling Tool Launcher"
echo "=================================="
echo ""
echo "This script helps you profile the diffusion policy inference"
echo "to identify the cause of timing variability (0.2s to 5s)."
echo ""
echo "Available profiling methods:"
echo ""
echo "  1) Quick kernel profile (fastest, ~2 min)"
echo "     - Shows which kernels are slow"
echo "     - Identifies kernels with high variance"
echo "     - Good starting point"
echo ""
echo "  2) Detailed PyTorch profiler (~5 min)"
echo "     - Comprehensive timeline analysis"
echo "     - Exports Chrome trace for visualization"
echo "     - Shows memory operations"
echo ""
echo "  3) NVIDIA Nsight Systems (most detailed, ~5 min)"
echo "     - Professional GPU profiling"
echo "     - Shows exact timeline of all GPU operations"
echo "     - Requires nvidia-nsight-systems installed"
echo ""
echo "  4) All profiling methods (runs all 3)"
echo ""
echo "  5) Exit"
echo ""

read -p "Select option (1-5): " choice

case $choice in
    1)
        echo ""
        echo "Running quick kernel profiler..."
        python quick_kernel_profile.py
        ;;
    2)
        echo ""
        echo "Running detailed PyTorch profiler..."
        python profile_inference.py
        echo ""
        echo "To view TensorBoard logs:"
        echo "  tensorboard --logdir=./profiler_logs"
        ;;
    3)
        echo ""
        echo "Running NVIDIA Nsight Systems profiler..."

        # Check if nsys is available
        if ! command -v nsys &> /dev/null; then
            echo ""
            echo "Error: nsys (NVIDIA Nsight Systems) not found!"
            echo ""
            echo "Install with:"
            echo "  sudo apt install nvidia-nsight-systems"
            echo ""
            exit 1
        fi

        chmod +x nsight_profile.sh
        ./nsight_profile.sh
        ;;
    4)
        echo ""
        echo "Running all profiling methods..."
        echo ""

        echo "=== STEP 1/3: Quick kernel profile ==="
        python quick_kernel_profile.py
        echo ""
        echo "Press Enter to continue to next profiling method..."
        read

        echo "=== STEP 2/3: Detailed PyTorch profiler ==="
        python profile_inference.py
        echo ""
        echo "Press Enter to continue to next profiling method..."
        read

        if command -v nsys &> /dev/null; then
            echo "=== STEP 3/3: NVIDIA Nsight Systems ==="
            chmod +x nsight_profile.sh
            ./nsight_profile.sh
        else
            echo "=== STEP 3/3: NVIDIA Nsight Systems ==="
            echo "Skipping (nsys not found)"
        fi
        ;;
    5)
        echo "Exiting..."
        exit 0
        ;;
    *)
        echo "Invalid option"
        exit 1
        ;;
esac

echo ""
echo "=================================="
echo "Profiling complete!"
echo "=================================="
