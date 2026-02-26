"""Policy inference server that runs independently."""

import sys
import time
import torch
import numpy as np
import dill
import hydra
from flask import Flask, request, jsonify
import base64

dp_path = '/home/mingxi/mingxi_ws/handpi/diffusion_policy'
robotool_path = dp_path + '/robotool'
sys.path.append(robotool_path)
sys.path.append(dp_path)
from diffusion_policy.workspace.base_workspace import BaseWorkspace
from diffusion_policy.policy.base_image_policy import BaseImagePolicy
from diffusion_policy.common.pytorch_util import dict_apply

app = Flask(__name__)

# Global policy object
policy = None
device = None

def initialize_policy(ckpt_path: str):
    """Initialize the diffusion policy model."""
    global policy, device

    print(f"Loading policy from {ckpt_path}")
    payload = torch.load(open(ckpt_path, 'rb'), pickle_module=dill)
    cfg = payload['cfg']
    cfg.logging.resume = False
    cfg.logging.mode = 'offline'
    if hasattr(cfg, 'real_robot_eval'):
        cfg.real_robot_eval = True
        print('set real_robot_eval to True')

    if hasattr(cfg, 'policy') :
        if hasattr(cfg.policy, 'predict_contact'):
            delattr(cfg.policy, 'predict_contact')
        if hasattr(cfg.policy, 'control_mode'):
            delattr(cfg.policy, 'control_mode')
    # if cfg.is_hand_pretrain == False:
    #     cfg.load_pretrain_folder = "/media/mingxi/T7/XEMB_Experiment/desk_clean_up/pretrained_ckpts/42_filtered_epoch_0340"
    print(f"cfg.se2_augmentation: {cfg.se2_augmentation}")

    print(f"cfg.policy: {cfg.policy}")
    
    cls = hydra.utils.get_class(cfg._target_)
    workspace = cls(cfg, real_robot_eval=True)

    # Filter out zero_conv parameters if they exist in checkpoint but not in model
    keys_to_remove = ['enc.enc_ih_zero_conv.weight', 'enc.enc_ih_zero_conv.bias']
    for state_dict_key in payload['state_dicts']:
        state_dict = payload['state_dicts'][state_dict_key]
        for param_key in keys_to_remove:
            if param_key in state_dict:
                print(f"Removing {param_key} from {state_dict_key}")
                del state_dict[param_key]

    # Exclude optimizer from loading (not needed for inference and has parameter group mismatch)
    workspace.load_payload(payload, exclude_keys=['optimizer'], include_keys=None)

    policy = workspace.ema_model
    import inspect
    print(inspect.getfile(policy.__class__))
    device = torch.device('cuda:0')
    print(device)
    policy.eval()
    policy.to(device)
    policy.num_inference_steps = 20
    policy.n_action_steps = 16
    policy.reset()

    print("Policy initialized successfully")
    return cfg

@app.route('/predict_intv', methods=['POST'])
def predict_intv():
    """Endpoint for intervention prediction using the policy."""
    global policy, device

    if policy is None:
        return jsonify({'error': 'Policy not initialized'}), 500

    try:
        # Receive observation dictionary
        data = request.get_json()

        # Decode numpy arrays from base64
        obs_dict = {}
        for key, value in data.items():
            array_bytes = base64.b64decode(value['data'])
            array = np.frombuffer(array_bytes, dtype=value['dtype']).reshape(value['shape'])
            obs_dict[key] = array

        # Run intervention prediction
        with torch.no_grad():
            t0 = time.time()
            # Convert to torch tensors (add batch and time dims)
            obs_dict_torch = dict_apply(obs_dict,
                lambda x: torch.from_numpy(x).unsqueeze(0).unsqueeze(1).to(device))
            predicted_label = policy.predict_intervention(obs_dict_torch)
            t_predict = time.time() - t0

        print(f"Intervention prediction: {predicted_label}")

        response = {
            'predicted_label': int(predicted_label),
            'timing': {
                'predict_intv': t_predict * 1000
            }
        }

        return jsonify(response)

    except Exception as e:
        return jsonify({'error': str(e)}), 500
    

@app.route('/predict', methods=['POST'])
def predict():
    """Endpoint for action prediction."""
    global policy, device

    if policy is None:
        return jsonify({'error': 'Policy not initialized'}), 500

    try:
        # Receive observation dictionary
        data = request.get_json()

        # Check for img_policy flag
        img_policy = data.pop('_img_policy', False)

        # Decode numpy arrays from base64
        obs_dict = {}
        for key, value in data.items():
            array_bytes = base64.b64decode(value['data'])
            array = np.frombuffer(array_bytes, dtype=value['dtype']).reshape(value['shape'])
            obs_dict[key] = array

        # Run inference
        with torch.no_grad():
            t0 = time.time()
            print(obs_dict.keys())
            for k, v in obs_dict.items():
                print(f"  {k}: shape={v.shape}, dtype={v.dtype}")
            if img_policy:
                # img_policy: frames already stacked with time dim, only add batch dim
                obs_dict_torch = dict_apply(obs_dict,
                    lambda x: torch.from_numpy(x).unsqueeze(0).to(device))
            else:
                obs_dict_torch = dict_apply(obs_dict,
                    lambda x: torch.from_numpy(x).unsqueeze(0).unsqueeze(1).to(device))
            torch.cuda.synchronize()
            t_to_gpu = time.time() - t0

            t0 = time.time()
            result = policy.predict_action(obs_dict_torch)
            torch.cuda.synchronize()
            t_predict = time.time() - t0

            t0 = time.time()
            action = result['action'][0].detach().to('cpu').numpy()
            t_to_cpu = time.time() - t0

        # Encode action as base64
        action_bytes = action.tobytes()
        action_b64 = base64.b64encode(action_bytes).decode('utf-8')
        print("done encoding action")
        response = {
            'action': {
                'data': action_b64,
                'dtype': str(action.dtype),
                'shape': action.shape
            },
            'timing': {
                'to_gpu': t_to_gpu * 1000,
                'predict': t_predict * 1000,
                'to_cpu': t_to_cpu * 1000
            }
        }

        return jsonify(response)

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/reset', methods=['POST'])
def reset():
    """Reset the policy."""
    global policy

    if policy is None:
        return jsonify({'error': 'Policy not initialized'}), 500

    policy.reset()
    return jsonify({'status': 'success'})

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint."""
    # Check if policy has predict_contact attribute (indicates it can predict intervention)
    predict_contact = getattr(policy, 'predict_contact', False) if policy is not None else False
    return jsonify({
        'status': 'healthy',
        'policy_loaded': policy is not None,
        'predict_contact': predict_contact
    })

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt_path', type=str, required=True,
                       help='Path to policy checkpoint')
    parser.add_argument('--port', type=int, default=5000,
                       help='Port to run server on')
    args = parser.parse_args()

    # Initialize policy before starting server
    initialize_policy(args.ckpt_path)

    # Run server
    print(f"Starting policy server on port {args.port}")
    app.run(host='localhost', port=args.port, threaded=False)
