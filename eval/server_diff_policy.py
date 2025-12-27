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

from diff_eval_utils.classifier_network import ResNetClassifier
app = Flask(__name__)

# Global policy and classifier objects
policy = None
classifier = None
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
    import inspect
    print(inspect.getfile(policy.__class__))
    device = torch.device('cuda')
    policy.eval()
    policy.to(device)
    policy.num_inference_steps = 20
    policy.n_action_steps = 16
    policy.reset()

    print("Policy initialized successfully")
    return cfg

def initialize_classifier(ckpt_path: str):
    """Initialize the intervention classifier model."""
    global classifier, device

    if device is None:
        device = torch.device('cuda')

    # Create model instance
    model = ResNetClassifier(num_classes=2, pretrained=False)

    # Load weights
    model.load_state_dict(torch.load(ckpt_path, map_location=device))

    # model.

    # Move to device and set to eval mode
    classifier = model.to(device)
    classifier.eval()

    print(f"Classifier loaded from {ckpt_path}")
    

@app.route('/predict', methods=['POST'])
def predict():
    """Endpoint for action prediction."""
    global policy, classifier, device

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

        # Run classifier inference if classifier is loaded
        if classifier is not None:
            t0 = time.time()
            # Get in-hand image - it's in (C, H, W) format with values in [0, 1]
            inhand_image = obs_dict['robot0_eye_in_hand_image']
            print(f"In-hand image shape: {inhand_image.shape}, dtype: {inhand_image.dtype}, min: {inhand_image.min()}, max: {inhand_image.max()}")

            predicted_label, confidence, prob_dict = classifier.predict_image(inhand_image, device=device)

            # Add is_contact to obs_dict (1 for contact/intervention, 0 for no contact)
            obs_dict['is_contact'] = np.array([predicted_label], dtype=np.float32)
            # obs_dict['is_contact'] = np.array([0], dtype=np.float32)  # --- IGNORE ---
            print(f"Classifier prediction: {predicted_label} (confidence: {confidence:.3f})")

            t_classifier = time.time() - t0
        else:
            t_classifier = 0.0

        # Run inference
        with torch.no_grad():
            t0 = time.time()
            print(obs_dict.keys())
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
                'classifier': t_classifier * 1000,
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
    parser.add_argument('--classifier_ckpt_path', type=str, default=None,
                       help='Path to intervention classifier checkpoint (optional)')
    parser.add_argument('--port', type=int, default=5000,
                       help='Port to run server on')
    args = parser.parse_args()

    # Initialize policy before starting server
    initialize_policy(args.ckpt_path)

    # Initialize classifier if path is provided
    if args.classifier_ckpt_path is not None:
        initialize_classifier(args.classifier_ckpt_path)
        print("Classifier enabled - will add 'in_contact' to observations")
    else:
        print("No classifier specified - running without intervention detection")

    # Run server
    print(f"Starting policy server on port {args.port}")
    app.run(host='localhost', port=args.port, threaded=False)
