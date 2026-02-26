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

CKPT_PATH="${1:-/media/mingxi/T7/XEMB_Experiment/nutella_sort/nutella_d0/epoch=0180-train_action_mse_error=0.001.ckpt}" # No intervention Finetune Checkpoint

# CKPT_PATH="${1:-/media/mingxi/T7/XEMB_Experiment/nutella_sort/nutella_d0/correction_d0_15_epoch=0040-train_action_mse_error=0.024.ckpt}"

# CKPT_PATH="${1:-/media/mingxi/T7/XEMB_Experiment/nutella_sort/nutella_d0/correction_new_d0_15_epoch=0020-train_action_mse_error=0.023.ckpt}"
# CKPT_PATH="${1:-/media/mingxi/T7/XEMB_Experiment/nutella_sort/nutella_d0/correction_new_d0_15_latest.ckpt}"

# CKPT_PATH="${1:-/media/mingxi/T7/XEMB_Experiment/nutella_sort/nutella_d0/data/outputs/diff_voxel_nutella_correction_precise_d0_30_realworld_15_20260202184505/checkpoints/epoch=0020-train_action_mse_error=0.032.ckpt}"
 
# CKPT_PATH="${1:-/media/mingxi/T7/XEMB_Experiment/nutella_sort/nutella_d0/data/outputs/diff_voxel_nutella_correction_precise_early_d0_15_modified_all_realworld_15_20260202204638/checkpoints/epoch=0080-train_action_mse_error=0.015.ckpt}"
# CKPT_PATH="${1:-/media/mingxi/T7/XEMB_Experiment/nutella_sort/nutella_d0/data/outputs/diff_voxel_nutella_correction_precise_early_d0_15_modified_all_realworld_15_20260202204638/checkpoints/latest.ckpt}"

# CKPT_PATH="${1:-/media/mingxi/T7/XEMB_Experiment/nutella_sort/nutella_d0/data/outputs/diff_voxel_nutella_correction_precise_early_d0_15_modified_all_realworld_15_20260202214209/checkpoints/latest.ckpt}"
CKPT_PATH="${1:-/media/mingxi/T7/XEMB_Experiment/desk_clean_up/desk_clean_up_d0_71_epoch=0200_pretrain.ckpt}"
# CKPT_PATH="${1:-/media/mingxi/T7/XEMB_Experiment/nutella_sort/nutella_d0/precise_early_15x2_10_pad_epoch=0180.ckpt}"
# CKPT_PATH="${1:-/media/mingxi/T7/XEMB_Experiment/nutella_sort/nutella_d0/precise_early_15x3_10_pad_epoch=0200.ckpt}"

# CKPT_PATH="${1:-/media/mingxi/T7/XEMB_Experiment/nutella_sort/nutella_d0/data/outputs/2026.02.05/14.08.12_diff_voxel_nutella_sort_fixed_offset_realworld_15_None/checkpoints/epoch=0180-train_action_mse_error=0.009.ckpt}"

# CKPT_PATH="${1:-/media/mingxi/T7/XEMB_Experiment/nutella_sort/nutella_d0/data/outputs/2026.02.05/14.37.02_diff_voxel_nutella_sort_fixed_offset_realworld_15x2_10_pad_30_None/checkpoints/epoch=0180-train_action_mse_error=0.009.ckpt}"


# # Coffee Making D0 80
# CKPT_PATH="${1:-/media/mingxi/T7/XEMB_Experiment/coffee_making/data/outputs/2026.02.06/10.02.07_diff_voxel_coffee_making_d0_80_realworld_80_None/checkpoints/epoch=0180-train_action_mse_error=0.000.ckpt}"

# CKPT_PATH="${1:-/media/mingxi/T7/XEMB_Experiment/party_host/party_host_d0_144_epoch=0180.ckpt}"

# CKPT_PATH="${1:-/media/mingxi/T7/XEMB_Experiment/coffee/raw_datasets/data/outputs/2026.02.06/16.31.27_diff_voxel_mug_d0_9_realworld_9_None/checkpoints/epoch=0180-train_action_mse_error=0.002.ckpt}"

# CKPT_PATH="${1:-/media/mingxi/T7/XEMB_Experiment/coffee_making/data/outputs/2026.02.07/13.13.50_diff_voxel_coffee_making_d1_33_realworld_33_None/checkpoints/epoch=0200-train_action_mse_error=0.002.ckpt}"
# CKPT_PATH="${1:-/media/mingxi/T7/XEMB_Experiment/coffee_making/coffee_making_d1_24_epoch=0180.ckpt}"

# CKPT_PATH="${1:-/media/mingxi/T7/XEMB_Experiment/coffee_prep/data/outputs/2026.02.06/10.57.37_diff_voxel_coffee_prep_d0_57_realworld_57_None/checkpoints/epoch=0200-train_action_mse_error=0.001.ckpt}"

CKPT_PATH="${1:-/media/mingxi/T7/XEMB_Experiment/coffee_making/coffee_making_d1_24_epoch=0360.ckpt}"

CKPT_PATH="${1:-/media/mingxi/T7/XEMB_Experiment/desk_clean_up/data/outputs/2026.02.08/11.51.56_diff_voxel_desk_clean_up_d1_40_trimmed_0150_1000_08_04_realworld_40_None/checkpoints/epoch=0200-train_action_mse_error=0.000.ckpt}"

CKPT_PATH="${1:-/media/mingxi/T7/XEMB_Experiment/desk_clean_up/data/outputs/2026.02.08/11.51.56_diff_voxel_desk_clean_up_d1_40_trimmed_0150_1000_08_04_realworld_40_None/checkpoints/epoch=0380-train_action_mse_error=0.000.ckpt}"

CKPT_PATH="${1:-/media/mingxi/T7/XEMB_Experiment/desk_clean_up/desk_clean_up_d1_42_filtered_epoch=0220.ckpt}"

CKPT_PATH="${1:-/media/mingxi/T7/XEMB_Experiment/desk_clean_up/desk_clean_up_d1_42_filtered_epoch=0340.ckpt}"

CKPT_PATH="${1:-/media/mingxi/T7/XEMB_Experiment/coffee_making/data/outputs/2026.02.09/01.18.14_diff_voxel_coffee_making_d1_filtered_realworld_21_None/checkpoints/epoch=0380-train_action_mse_error=0.000.ckpt}"

CKPT_PATH="${1:-/media/mingxi/T7/XEMB_Experiment/coffee_making/data/outputs/2026.02.09/19.30.50_diff_voxel_coffee_making_d2_filtered_smoothed_realworld_28_None/checkpoints/epoch=0700-train_action_mse_error=0.000.ckpt}"

CKPT_PATH="${1:-/media/mingxi/T7/XEMB_Experiment/coffee_making/data/outputs/2026.02.09/19.30.50_diff_voxel_coffee_making_d2_filtered_smoothed_realworld_28_None/checkpoints/latest.ckpt}" # 400 + 400 epochs

# CKPT_PATH="${1:-/media/mingxi/T7/XEMB_Experiment/coffee_making/coffee_making_d2_epoch=0760.ckpt}"

# CKPT_PATH="${1:-/media/mingxi/T7/XEMB_Experiment/coffee_making/coffee_making_d2_epoch=0800.ckpt}"

CKPT_PATH="${1:-/media/mingxi/T7/XEMB_Experiment/coffee_making/coffee_making_d2_s1_epoch=0260.ckpt}"

CKPT_PATH="${1:-/media/mingxi/T7/XEMB_Experiment/desk_clean_up/d1_finetune_0200.ckpt}"

CKPT_PATH="${1:-/media/mingxi/T7/XEMB_Experiment/coffee_making/data/outputs/2026.02.11/16.52.10_diff_voxel_coffee_making_d2_s12_54_filtered_smoothed_realworld_54_None/checkpoints/epoch=0200-train_action_mse_error=0.000.ckpt}"

# Desk Clean up 
CKPT_PATH="${1:-/media/mingxi/T7/XEMB_Experiment/desk_clean_up/d1_pretrain_0400.ckpt}"

# CKPT_PATH="${1:-/media/mingxi/T7/XEMB_Experiment/coffee_making/d2_s12_54_0400.ckpt}"

CKPT_PATH="${1:-/media/mingxi/T7/XEMB_Experiment/party_host/data/outputs/2026.02.12/15.15.15_diff_voxel_party_host_d0_all_199_realworld_199_None/checkpoints/epoch=0400-train_action_mse_error=0.000.ckpt}"

# # CKPT_PATH="${1:-/media/mingxi/T7/XEMB_Experiment/desk_clean_up/data/outputs/2026.02.12/23.40.13_diff_voxel_desk_clean_up_d1_15x2_10_pad_realworld_30_None/checkpoints/epoch=0380-train_action_mse_error=0.004.ckpt}"

# # CKPT_PATH="${1:-/media/mingxi/T7/XEMB_Experiment/coffee_making/data/outputs/2026.02.13/02.13.42_diff_voxel_coffee_making_d2_s12_54_filtered_smoothed_realworld_54_None/checkpoints/epoch=0240-train_action_mse_error=0.000.ckpt}"

# CKPT_PATH="${1:-/media/mingxi/T7/XEMB_Experiment/coffee_making/data/outputs/2026.02.13/11.55.13_diff_voxel_coffee_making_d2_s12_54_filtered_smoothed_realworld_54_None/checkpoints/epoch=0400-train_action_mse_error=0.000.ckpt}"

# CKPT_PATH="${1:-/media/mingxi/T7/XEMB_Experiment/coffee_prep/data/outputs/2026.02.12/01.56.33_diff_voxel_coffee_prep_d0_57_realworld_57_None/checkpoints/epoch=0380-train_action_mse_error=0.000.ckpt}"

# CKPT_PATH="${1:-/media/mingxi/T7/XEMB_Experiment/desk_clean_up/data/outputs/2026.02.13/22.04.44_diff_voxel_desk_clean_up_d1_21x2_10_pad_realworld_42_None/checkpoints/latest.ckpt}"

CKPT_PATH="${1:-/media/mingxi/T7/XEMB_Experiment/desk_clean_up/epoch=0400-train_action_mse_error=0.006.ckpt}"

CKPT_PATH="${1:-/media/mingxi/T7/XEMB_Experiment/desk_clean_up/data/outputs/2026.02.14/21.31.02_diff_voxel_desk_clean_up_d1_21x2_10_pad_realworld_42_None/checkpoints/epoch=0580-train_action_mse_error=0.005.ckpt}"

# CKPT_PATH="${1:-/media/mingxi/T7/XEMB_Experiment/coffee_making/d23_86_pretrain_0400.ckpt}"

CKPT_PATH="${1:-/media/mingxi/T7/XEMB_Experiment/party_host/party_host_d1_144_0360.ckpt}"

CKPT_PATH="${1:-/media/mingxi/T7/XEMB_Experiment/coffee_making/coffee_making_d3_86_pretrain_0400.ckpt}"

CKPT_PATH="${1:-/media/mingxi/T7/XEMB_Experiment/desk_clean_up/desk_clean_up_85_pretrain_0400.ckpt}"

# CKPT_PATH="${1:-/media/mingxi/T7/XEMB_Experiment/party_host/data/outputs/2026.02.17/04.34.57_diff_voxel_party_host_d1_110_all_realworld_110_None/checkpoints/epoch=0400-train_action_mse_error=0.000.ckpt}"

# CKPT_PATH="${1:-/media/mingxi/T7/XEMB_Experiment/coffee_making/data/outputs/2026.02.17/13.16.36_diff_voxel_coffee_making_d3_intv_realworld_15_None/checkpoints/epoch=0360-train_action_mse_error=0.004.ckpt}"

# CKPT_PATH="${1:-/media/mingxi/T7/XEMB_Experiment/desk_clean_up/desk_clean_up_d2_intv_17_0400.ckpt}"

CKPT_PATH="${1:-/media/mingxi/T7/XEMB_Experiment/coffee_making/d3_intv_29_0400.ckpt}"

CKPT_PATH="${1:-/media/mingxi/T7/XEMB_Experiment/coffee_making/d4_117_pretrain_0400.ckpt}"

CKPT_PATH="${1:-/media/mingxi/T7/XEMB_Experiment/desk_clean_up/d2_74_pretrain_0400.ckpt}"

CKPT_PATH="${1:-/media/mingxi/T7/XEMB_Experiment/party_host/d2_78_pretrain_0400.ckpt}"

CKPT_PATH="${1:-/media/mingxi/T7/XEMB_Experiment/desk_clean_up/d3_51_pretrain_0400.ckpt}"

# CKPT_PATH="${1:-/media/mingxi/T7/XEMB_Experiment/party_host/d2_143_f2_pretrain_0400.ckpt}"

CKPT_PATH="${1:-/media/mingxi/T7/XEMB_Experiment/coffee_making/epoch=0260-train_action_mse_error=0.004.ckpt}"

CKPT_PATH="${1:-/media/mingxi/T7/XEMB_Experiment/coffee_making/d5_101_f2_pretrain_0400.ckpt}"

# CKPT_PATH="${1:-/media/mingxi/T7/XEMB_Experiment/coffee_making/data/outputs/2026.02.24/02.18.07_diff_voxel_coffee_making_d5_intv_realworld_26_None/checkpoints/latest.ckpt}"

CKPT_PATH="${1:-/media/mingxi/T7/XEMB_Experiment/party_host/d3_106_f1_pretrain.ckpt}"

CKPT_PATH="${1:-/media/mingxi/T7/XEMB_Experiment/coffee_making/d5_101_f1_pretrain_0400.ckpt}"

# CKPT_PATH="${1:-/media/mingxi/T7/XEMB_Experiment/coffee_making/d5_101_f1_pretrain_test.ckpt}"

CKPT_PATH="${1:-/media/mingxi/T7/XEMB_Experiment/coffee_making/d6_all_106_pretrain_0400.ckpt}"

# CKPT_PATH="${1:-/media/mingxi/T7/XEMB_Experiment/party_host/d3_106_pretrain_0400.ckpt}"

CKPT_PATH="${1:-/media/mingxi/T7/XEMB_Experiment/desk_clean_up/d4_53_pretrain_0400.ckpt}"

CKPT_PATH="${1:-/media/mingxi/Elements/XEmbodimentManipulation/coffee_making/d6_107_pretrain_0400.ckpt}"

LD_LIBRARY_PATH="$CONDA_PREFIX/lib"

PORT="${2:-6666}"

HYDRA_FULL_ERROR=1

echo "Starting policy server..."
echo "Checkpoint: $CKPT_PATH"
echo "Port: $PORT"
echo ""

which python
python eval_policy/server_diff_policy.py --ckpt_path "$CKPT_PATH" --port "$PORT"