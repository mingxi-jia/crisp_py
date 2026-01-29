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
# CKPT_PATH="${1:-/home/mingxi/Downloads/epoch=0060-val_loss=0.022-pointvoxel.ckpt}" # Screw Hand Only Fixed normalization pointvoxel
# CKPT_PATH="${1:-/home/mingxi/Downloads/epoch=0060-val_loss=0.020.ckpt}" # Screw Hand Only Fixed normalization pointnet
# CKPT_PATH="${1:-/home/mingxi/Downloads/latest.ckpt}" # Screw Hand Only Fixed normalization pointnet Updated

# CKPT_PATH="${1:-/home/mingxi/Downloads/epoch=0020-val_loss=0.036.ckpt}" # Screw Hand + Intervention fine tune Fixed normalization pointnet
# CKPT_PATH="${1:-/home/mingxi/Downloads/epoch=0040-val_loss=0.047.ckpt}" # Screw Hand + Intervention fine tune Fixed normalization pointnet


# CKPT_PATH="${1:-/home/mingxi/Downloads/epoch=0100-val_loss=0.030.ckpt}" # Screw Hand Intervention Mix Training
CKPT_PATH="${1:-/home/mingxi/Downloads/epoch=0060-val_loss=0.024.ckpt}" # Screw Hand Only
CKPT_PATH="${1:-/home/mingxi/Downloads/epoch=0040-val_loss=0.018.ckpt}" # Screw Hand Only
# CKPT_PATH="${1:-/home/mingxi/Downloads/epoch=0100-val_loss=0.026.ckpt}" # Screw Hand Intervention Mix Training Include Ground
# CKPT_PATH="${1:-/home/mingxi/Downloads/epoch=0020-val_loss=0.022.ckpt}" # Screw Hand Only Augmentation-Max
# CKPT_PATH="${1:-/home/mingxi/Downloads/epoch=0040-val_loss=0.020.ckpt}" # Screw Hand Only Augmentation-Max
# CKPT_PATH="${1:-/home/mingxi/Downloads/epoch=0020-val_loss=0.019.ckpt}" # Screw Hand Intervention T2I top left corner
# CKPT_PATH="${1:-/home/mingxi/Downloads/epoch=0060-val_loss=0.022.ckpt}" # Intervention Finetune top left corner
# CKPT_PATH="${1:-/home/mingxi/Downloads/epoch=0180-val_loss=0.017.ckpt}" # Intervention T2I top left corner
CKPT_PATH="${1:-/home/mingxi/Downloads/epoch=0140-val_loss=0.018.ckpt}" # Screw Hand Only
# CKPT_PATH="${1:-/home/mingxi/Downloads/epoch=0120-val_loss=0.017.ckpt}" # Screw Hand Only
CKPT_PATH="${1:-/home/mingxi/Downloads/epoch=0080-val_loss=0.018.ckpt}" # Screw Hand Only
# CKPT_PATH="${1:-/home/mingxi/Downloads/epoch=0040-val_loss=0.014.ckpt}" # Screw Hand Only + Grasp Finetune
# CKPT_PATH="${1:-/home/mingxi/Downloads/epoch=0080-val_loss=0.018-ctrl-net.ckpt}" # Screw Hand Only ControlNet
# CKPT_PATH="${1:-/home/mingxi/Downloads/epoch=0140-val_loss=0.018-nobias.ckpt}" # Screw Hand + ControlNet Soft 
# CKPT_PATH="${1:-/home/mingxi/Downloads/epoch=0080-val_loss=0.018-1.ckpt}" # Screw Hand + ControlNet Soft  
# CKPT_PATH="${1:-/home/mingxi/Downloads/epoch=0060-val_loss=0.017.ckpt}" # Screw Hand + ControlNet Soft  
# CKPT_PATH="${1:-/home/mingxi/Downloads/epoch=0020-val_loss=0.020-1.ckpt}" # Screw Hand + ControlNet Soft  
# CKPT_PATH="${1:-/home/mingxi/Downloads/epoch=0060-val_loss=0.018.ckpt}" # Screw Hand + ControlNet Soft  
# CKPT_PATH="${1:-/home/mingxi/Downloads/epoch=0160-val_loss=0.016.ckpt}" # Screw Hand + ControlNet Soft  
# CKPT_PATH="${1:-/home/mingxi/Downloads/epoch=0160-val_loss=0.018.ckpt}" # Screw Hand + ControlNet Soft  
# CKPT_PATH="${1:-/home/mingxi/Downloads/epoch=0100-val_loss=0.043.ckpt}" # Screw Hand + ControlNet Soft  
# CKPT_PATH="${1:-/home/mingxi/Downloads/epoch=0080-val_loss=0.018-2.ckpt}" # Screw Hand Only
# CKPT_PATH="${1:-/home/mingxi/Downloads/epoch=0060-val_loss=0.043.ckpt}" # Screw Hand Only Good No Aug Relative
# CKPT_PATH="${1:-/home/mingxi/Downloads/latest_1226.ckpt}" # Screw Hand Only without rot6d
# CKPT_PATH="${1:-/home/mingxi/Downloads/epoch=0060-val_loss=0.020-1.ckpt}" # Screw Hand Only without rot6d
CKPT_PATH="${1:-/home/mingxi/Downloads/epoch=0080-val_loss=0.019.ckpt}" # Screw Hand Only with rot6d (good enough) -> v1 pretrain
# CKPT_PATH="${1:-/home/mingxi/Downloads/epoch=0240-val_loss=0.012.ckpt}" # Screw Hand finetune Relative -> v1 finetune
CKPT_PATH="${1:-/home/mingxi/Downloads/epoch=0240-val_loss=0.007.ckpt}" # Screw Hand finetune Relative -> v1 finetune New Dataset
# CKPT_PATH="${1:-/home/mingxi/Downloads/epoch=0100-train_action_mse_error=0.014.ckpt}" # Screw Hand finetune Relative -> v1 finetune New Dataset Filtered


CKPT_PATH="${1:-/home/mingxi/data/realworld/data/outputs/2026.01.23/18.01.03_train_diffusion_unet_image_pod_grasp_realworld_10/checkpoints/epoch=0100-train_action_mse_error=0.005.ckpt}" # Coffee Pod Lifting
CKPT_PATH="${1:-/home/mingxi/data/realworld/data/outputs/2026.01.23/18.29.16_train_diffusion_unet_image_pod_grasp_realworld_10/checkpoints/epoch=0100-train_action_mse_error=0.005.ckpt}" # Coffee Pod Lifting
CKPT_PATH="${1:-/home/mingxi/data/realworld/data/outputs/2026.01.23/21.47.38_train_diffusion_unet_image_pod_grasp_realworld_23/checkpoints/latest.ckpt}"

CKPT_PATH="${1:-/home/mingxi/data/realworld/data/outputs/2026.01.24/17.02.28_train_diffusion_unet_image_pod_grasp_realworld_23/checkpoints/latest.ckpt}"

CKPT_PATH="${1:-/media/mingxi/T7/XEMB_Experiment/pod_lifting/data/outputs/2026.01.25/16.30.59_train_diffusion_unet_image_pod_grasp_realworld_w_offset_23/checkpoints/latest.ckpt}"


CKPT_PATH="${1:-/media/mingxi/T7/XEMB_Experiment/pod_lifting/data/outputs/2026.01.25/16.30.59_train_diffusion_unet_image_pod_grasp_realworld_w_offset_23/checkpoints/latest.ckpt}"



# CKPT_PATH="${1:-/media/mingxi/T7/XEMB_Experiment/pod_lifting/data/outputs/2026.01.27/10.42.58_train_diffusion_unet_image_pod_grasp_realworld_wo_offset_23/checkpoints/latest.ckpt}"
# # CKPT_PATH="${1:-/media/mingxi/T7/XEMB_Experiment/pod_lifting/data/outputs/2026.01.27/10.54.56_train_diffusion_unet_image_pod_grasp_realworld_wo_offset_slow_23/checkpoints/latest.ckpt}"

# CKPT_PATH="${1:-/media/mingxi/T7/XEMB_Experiment/pod_lifting/data/outputs/2026.01.27/11.19.25_train_diffusion_unet_image_pod_grasp_realworld_wo_offset_slow_23/checkpoints/latest.ckpt}"

CKPT_PATH="${1:-/media/mingxi/T7/XEMB_Experiment/nutella_sort/nutella_sort_D1_epoch=0200-train_action_mse_error=0.000.ckpt}"

CKPT_PATH="${1:-/media/mingxi/T7/XEMB_Experiment/nutella_sort/nutella_sort_D1_epoch=0500-train_action_mse_error=0.000.ckpt}"

CKPT_PATH="${1:-/media/mingxi/T7/nutella_dp_2d/episodes/data/outputs/2026.01.29/02.01.47_train_diffusion_unet_image_nutella_realworld_20/checkpoints/latest.ckpt}"

CKPT_PATH="${1:-/media/mingxi/T7/nutella_dp_2d/data/outputs/2026.01.29/02.39.14_train_diffusion_unet_image_nutella_realworld_3/checkpoints/epoch=0140-train_action_mse_error=0.001.ckpt}"

LD_LIBRARY_PATH="$CONDA_PREFIX/lib"

CLASSIFIER_CKPT_PATH="${2:-/home/mingxi/mingxi_ws/crisp/crisp_py/eval_policy/best_intervention_classifier_left_side.pth}"

PORT="${3:-6666}"

HYDRA_FULL_ERROR=1

echo "Starting policy server..."
echo "Checkpoint: $CKPT_PATH"
echo "Classifier Checkpoint: $CLASSIFIER_CKPT_PATH"
echo "Port: $PORT"
echo ""

which python
python eval_policy/server_diff_policy.py --ckpt_path "$CKPT_PATH" --classifier_ckpt_path "$CLASSIFIER_CKPT_PATH" --port "$PORT"