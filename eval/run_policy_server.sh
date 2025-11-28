#!/bin/bash

# Script to start the policy inference server
# Usage: ./run_policy_server.sh [checkpoint_path] [port]

# CKPT_PATH="${1:-/home/mingxi/Downloads/epoch=0040-val_loss=0.008.ckpt}"
# CKPT_PATH="${1:-/home/mingxi/Downloads/epoch=0100-val_loss=0.011.ckpt}"
# CKPT_PATH="${1:-/home/mingxi/Downloads/epoch=0040-val_loss=0.021.ckpt}"
CKPT_PATH="${1:-/home/mingxi/Downloads/epoch=0020-val_loss=0.024.ckpt}"
# CKPT_PATH="${1:-/home/mingxi/Downloads/epoch=0040-val_loss=0.020.ckpt}"
CKPT_PATH="${1:-/home/mingxi/Downloads/epoch=0080-val_loss=0.016.ckpt}"
CKPT_PATH="${1:-/home/mingxi/Downloads/epoch=0040-val_loss=0.008.ckpt}" # Old Diffusion Policy
PORT="${2:-5000}"

HYDRA_FULL_ERROR=1

echo "Starting policy server..."
echo "Checkpoint: $CKPT_PATH"
echo "Port: $PORT"
echo ""

python eval/server_diff_policy.py --ckpt_path "$CKPT_PATH" --port "$PORT"
