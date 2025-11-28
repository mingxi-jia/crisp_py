#!/bin/bash

# Script to start the point cloud processing server
# Usage: ./run_pcd_server.sh [port]

PORT="${1:-5001}"

echo "Starting PCD processing server..."
echo "Port: $PORT"
echo ""

python eval/server_pcd_processing.py --port "$PORT"
