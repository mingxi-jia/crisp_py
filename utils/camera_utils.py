import os
import yaml

def load_camera_info(yaml_path):
    with open(yaml_path, 'r') as f:
        cam_info = yaml.safe_load(f)
    return cam_info