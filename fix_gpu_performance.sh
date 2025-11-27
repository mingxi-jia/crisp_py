#!/bin/bash

# Script to lock GPU to maximum performance mode
# This eliminates inference time variability caused by GPU frequency scaling

echo "=================================="
echo "GPU Performance Lock Script"
echo "=================================="

# Check if running as root
if [ "$EUID" -ne 0 ]; then
    echo "This script needs sudo privileges. Running with sudo..."
    sudo "$0" "$@"
    exit $?
fi

echo ""
echo "Current GPU status:"
nvidia-smi --query-gpu=name,power.draw,power.limit,clocks.current.graphics,clocks.max.graphics --format=csv

echo ""
echo "Setting GPU to maximum performance mode..."

# Enable persistence mode (keeps GPU initialized between runs)
nvidia-smi -pm 1
echo "✓ Persistence mode enabled"

# Set power limit to maximum (adjust if needed)
# Get max power limit first
MAX_POWER=$(nvidia-smi --query-gpu=power.max_limit --format=csv,noheader,nounits | head -1)
nvidia-smi -pl $MAX_POWER
echo "✓ Power limit set to maximum: ${MAX_POWER}W"

# Lock GPU clocks to maximum (prevents downclocking)
# Get max graphics clock
MAX_CLOCK=$(nvidia-smi --query-gpu=clocks.max.graphics --format=csv,noheader,nounits | head -1)
nvidia-smi -lgc $MAX_CLOCK
echo "✓ Graphics clock locked to: ${MAX_CLOCK}MHz"

# Set compute mode to default (allows multiple processes)
nvidia-smi -c 0
echo "✓ Compute mode set to default"

echo ""
echo "New GPU status:"
nvidia-smi --query-gpu=name,power.draw,power.limit,clocks.current.graphics,clocks.max.graphics --format=csv

echo ""
echo "=================================="
echo "GPU is now locked to max performance!"
echo "=================================="
echo ""
echo "To revert to auto mode later, run:"
echo "  sudo nvidia-smi -pm 0"
echo "  sudo nvidia-smi -rgc"
echo ""
