#!/bin/bash

# Script to start the policy inference server
# Usage: ./run_policy_server.sh [checkpoint_path] [port]

# CKPT_PATH="${1:-/home/mingxi/Downloads/epoch=0040-val_loss=0.008.ckpt}"
# CKPT_PATH="${1:-/home/mingxi/Downloads/epoch=0100-val_loss=0.011.ckpt}"
# CKPT_PATH="${1:-/home/mingxi/Downloads/epoch=0040-val_loss=0.021.ckpt}"
CKPT_PATH="${1:-/home/mingxi/Downloads/epoch=0020-val_loss=0.024.ckpt}"
# CKPT_PATH="${1:-/home/mingxi/Downloads/epoch=0040-val_loss=0.020.ckpt}"
CKPT_PATH="${1:-/home/mingxi/Downloads/epoch=0080-val_loss=0.016.ckpt}"
# CKPT_PATH="${1:-/home/mingxi/Downloads/epoch=0040-val_loss=0.008.ckpt}" # Old Diffusion Policy Hand Sphere Data
CKPT_PATH="${1:-/home/mingxi/Downloads/epoch=0100-val_loss=0.009.ckpt}" # Old Diffusion Policy Hand Gripper Data
# CKPT_PATH="${1:-/home/mingxi/Downloads/epoch=0180-val_loss=0.014.ckpt}" # Old Diffusion Policy Hand Gripper Data
CKPT_PATH="${1:-/home/mingxi/Downloads/epoch=0200-val_loss=0.011.ckpt}" # New Diffusion Policy Hand Gripper Data
# CKPT_PATH="${1:-/home/mingxi/Downloads/epoch=0060-val_loss=0.009.ckpt}" # Old Diffusion Policy Hand Gripper Data Finetune
# CKPT_PATH="${1:-/home/mingxi/Downloads/epoch=0020-val_loss=0.012.ckpt}" # Old Diffusion Policy Hand Gripper Data Finetune
# CKPT_PATH="${1:-/home/mingxi/Downloads/epoch=0100-val_loss=0.006.ckpt}" # Old Diffusion Policy Hand Gripper Data Finetune
CKPT_PATH="${1:-/home/mingxi/Downloads/epoch=0080-val_loss=0.008.ckpt}" # New Diffusion Policy Hand Gripper Data Finetune
CKPT_PATH="${1:-/home/mingxi/Downloads/epoch=0160-val_loss=0.007.ckpt}" # New Diffusion Policy Hand Gripper Data Finetune
CKPT_PATH="${1:-/home/mingxi/Downloads/epoch=0080-val_loss=0.006.ckpt}" # New Diffusion Policy Hand Gripper Data Finetune Pointnet 
CKPT_PATH="${1:-/home/mingxi/Downloads/epoch=0080-val_loss=0.010.ckpt}" # New Diffusion Policy Hand Gripper Data Mix-Finetune Pointnet 
# CKPT_PATH="${1:-/home/mingxi/Downloads/epoch=0120-val_loss=0.009.ckpt}" # New Diffusion Policy Hand Gripper Data Mix-Finetune Pointnet 
CKPT_PATH="${1:-/home/mingxi/Downloads/epoch=0200-val_loss=0.009.ckpt}" # New Diffusion Policy Hand Gripper Data Mix-Finetune Pointnet 
CKPT_PATH="${1:-/home/mingxi/Downloads/epoch=0080-val_loss=0.009.ckpt}" # New Diffusion Policy Hand Gripper Data Mix-Finetune Pointnet WirstAlwaysOn MixRender 
CKPT_PATH="${1:-/home/mingxi/Downloads/epoch=0080-val_loss=0.050.ckpt}" # New Diffusion Policy Robot Only Pointnet WirstAlwaysOn MixRender 
CKPT_PATH="${1:-/home/mingxi/Downloads/epoch=0060-val_loss=0.018.ckpt}" # New Diffusion Policy Robot Only Pointnet WirstAlwaysOn MixRender 20 demos
CKPT_PATH="${1:-/home/mingxi/Downloads/epoch=0100-val_loss=0.024.ckpt}" # New Diffusion Policy Robot Only Pointnet WirstAlwaysOn MixRender 20 demos
# CKPT_PATH="${1:-/home/mingxi/Downloads/epoch=0120-val_loss=0.017.ckpt}" # New Diffusion Policy Robot Only Pointnet WirstAlwaysOn MixRender 20 demos
CKPT_PATH="${1:-/home/mingxi/Downloads/epoch=0020-val_loss=0.008.ckpt}" # New Diffusion Policy MixTraining Pointnet WirstAlwaysOn MixRender 20 demos
CKPT_PATH="${1:-/home/mingxi/Downloads/epoch=0080-val_loss=0.008.ckpt}" # New Diffusion Policy MixTraining Pointnet WirstAlwaysOn MixRender 20 demos
CKPT_PATH="${1:-/home/mingxi/Downloads/epoch=0100-val_loss=0.008.ckpt}" # New Diffusion Policy MixTraining Pointnet WirstAlwaysOn MixRender 8 demos
CKPT_PATH="${1:-/home/mingxi/Downloads/epoch=0020-val_loss=0.010.ckpt}" # New New Diffusion Policy MixTraining Pointnet WirstAlwaysOn MixRender 8 demos
CKPT_PATH="${1:-/home/mingxi/Downloads/epoch=0280-val_loss=0.008.ckpt}" # New New Diffusion Policy MixTraining Pointnet Blind MixRender 8 demos
CKPT_PATH="${1:-/home/mingxi/Downloads/epoch=0280-val_loss=0.034.ckpt}" # New New Diffusion Policy Robot Only Pointnet Blind MixRender 8 demos

CKPT_PATH="${1:-/home/mingxi/Downloads/epoch=0060-val_loss=0.019.ckpt}" # Screw Hand Only
# CKPT_PATH="${1:-/home/mingxi/Downloads/epoch=0020-val_loss=0.021.ckpt}" # Screw Pretrain + MixTrain
# CKPT_PATH="${1:-/home/mingxi/Downloads/epoch=0040-val_loss=0.020.ckpt}" # Screw Pretrain + MixTrain
# CKPT_PATH="${1:-/home/mingxi/Downloads/epoch=0020-val_loss=0.012.ckpt}" # Screw Hand Only Interpolated 0.6
CKPT_PATH="${1:-/home/mingxi/Downloads/epoch=0100-val_loss=0.013.ckpt}" # Screw Hand Only Interpolated 0.5
CKPT_PATH="${1:-/home/mingxi/Downloads/epoch=0020-val_loss=0.020.ckpt}" # Screw Hand Only Fixed normalization pointnet-voxel
CKPT_PATH="${1:-/home/mingxi/Downloads/epoch=0020-val_loss=0.020-pointnet.ckpt}" # Screw Hand Only Fixed normalization pointnet
CKPT_PATH="${1:-/home/mingxi/Downloads/epoch=0060-val_loss=0.022.ckpt}" # Screw Hand Only Fixed normalization pointnet

# CKPT_PATH="${1:-/home/mingxi/Downloads/epoch=0060-val_loss=0.021.ckpt}" # Screw Pretrain + MixTrain Interpolated 0.5
# CKPT_PATH="${1:-/home/mingxi/Downloads/epoch=0120-val_loss=0.025.ckpt}" # Screw Pretrain + MixTrain Interpolated 0.5
# CKPT_PATH="${1:-/home/mingxi/Downloads/epoch=0060-val_loss=0.021 (1).ckpt}" # Screw Pretrain + MixTrain Interpolated 0.5 + raw_pcd wrist gating
# CKPT_PATH="${1:-/home/mingxi/Downloads/epoch=0040-val_loss=0.034.ckpt}" # Screw Pretrain + MixTrain Interpolated 0.5 + raw_pcd wrist gating + no blind + fix control feature
# CKPT_PATH="${1:-/home/mingxi/Downloads/epoch=0160-val_loss=0.012.ckpt}" # Screw Pretrain + MixTrain Interpolated 0.5 + raw_pcd wrist gating + no blind + fix control feature
# CKPT_PATH="${1:-/home/mingxi/Downloads/epoch=0020-val_loss=0.014.ckpt}" # Screw Pretrain + MixTrain Interpolated 0.5 + raw_pcd wrist gating + no blind + fix control feature


PORT="${2:-5000}"

HYDRA_FULL_ERROR=1

export PYTHONPATH=/home/mingxi/mingxi_ws/handpi/robot-vision-toolbox:$PYTHONPATH

echo "Starting policy server..."
echo "Checkpoint: $CKPT_PATH"
echo "Port: $PORT"
echo ""

which python
python eval/server_diff_policy.py --ckpt_path "$CKPT_PATH" --port "$PORT"
