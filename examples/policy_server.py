"""Policy inference server that runs independently."""

import sys
import time
import torch
import numpy as np
import dill
import hydra
from flask import Flask, request, jsonify
import base64

sys.path.append('/home/mingxi/mingxi_ws/handpi/diffusion_policy')
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
    cfg.real_robot_eval = True

    cls = hydra.utils.get_class(cfg._target_)
    workspace = cls(cfg)
    workspace.load_payload(payload, exclude_keys=None, include_keys=None)

    policy = workspace.model
    device = torch.device('cuda')
    policy.eval()
    policy.to(device)
    policy.num_inference_steps = 20
    policy.n_action_steps = 16
    policy.reset()

    print("Policy initialized successfully")
    return cfg

@app.route('/predict', methods=['POST'])
def predict():
    """Endpoint for action prediction."""
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

        # Run inference
        with torch.no_grad():
            t0 = time.time()
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
    return jsonify({
        'status': 'healthy',
        'policy_loaded': policy is not None
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
