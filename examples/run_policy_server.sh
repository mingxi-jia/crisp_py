#!/bin/bash

# Script to start the policy inference server
# Usage: ./run_policy_server.sh [checkpoint_path] [port]

CKPT_PATH="${1:-/home/mingxi/Downloads/epoch=0040-val_loss=0.008.ckpt}"
PORT="${2:-5000}"

echo "Starting policy server..."
echo "Checkpoint: $CKPT_PATH"
echo "Port: $PORT"
echo ""

python examples/policy_server.py --ckpt_path "$CKPT_PATH" --port "$PORT"
