"""Point cloud processing server that runs independently."""

import sys
import time
import numpy as np
import base64
from flask import Flask, request, jsonify

sys.path.append('/home/mingxi/mingxi_ws/handpi/robot-vision-toolbox')
from hand.trajectory_loader import ObservationProcessor

app = Flask(__name__)

# Global processor object
pcd_processor = None

def initialize_processor():
    """Initialize the point cloud processor."""
    global pcd_processor

    print("Initializing point cloud processor...")
    pcd_processor = ObservationProcessor()
    print("Point cloud processor initialized successfully")

@app.route('/process_pcd', methods=['POST'])
def process_pcd():
    """Endpoint for point cloud processing."""
    global pcd_processor

    if pcd_processor is None:
        return jsonify({'error': 'Processor not initialized'}), 500

    try:
        # Receive data
        data = request.get_json()

        # Decode numpy arrays from base64
        pcd_bytes = base64.b64decode(data['pcd']['data'])
        pcd = np.frombuffer(pcd_bytes, dtype=data['pcd']['dtype']).reshape(data['pcd']['shape'])

        eef_pose_bytes = base64.b64decode(data['eef_pose']['data'])
        eef_pose = np.frombuffer(eef_pose_bytes, dtype=data['eef_pose']['dtype']).reshape(data['eef_pose']['shape'])

        joint_state_bytes = base64.b64decode(data['joint_state']['data'])
        joint_state = np.frombuffer(joint_state_bytes, dtype=data['joint_state']['dtype']).reshape(data['joint_state']['shape'])

        # Process point cloud
        t0 = time.time()
        processed_pcd, render_pcd = pcd_processor.get_policy_obs(pcd, eef_pose, joint_state)
        t_process = time.time() - t0

        # Encode results as base64
        processed_pcd_bytes = processed_pcd.tobytes()
        processed_pcd_b64 = base64.b64encode(processed_pcd_bytes).decode('utf-8')

        render_pcd_bytes = render_pcd.tobytes()
        render_pcd_b64 = base64.b64encode(render_pcd_bytes).decode('utf-8')

        response = {
            'processed_pcd': {
                'data': processed_pcd_b64,
                'dtype': str(processed_pcd.dtype),
                'shape': processed_pcd.shape
            },
            'render_pcd': {
                'data': render_pcd_b64,
                'dtype': str(render_pcd.dtype),
                'shape': render_pcd.shape
            },
            'timing': {
                'process': t_process * 1000
            }
        }

        return jsonify(response)

    except Exception as e:
        import traceback
        return jsonify({
            'error': str(e),
            'traceback': traceback.format_exc()
        }), 500

@app.route('/process_images', methods=['POST'])
def process_images():
    """Endpoint for image processing."""
    global pcd_processor

    if pcd_processor is None:
        return jsonify({'error': 'Processor not initialized'}), 500

    try:
        # Receive data
        data = request.get_json()

        # Decode images from base64
        rgb_dict = {}
        depth_dict = {}

        for cam_name, img_data in data['rgb_dict'].items():
            img_bytes = base64.b64decode(img_data['data'])
            rgb_dict[cam_name] = np.frombuffer(img_bytes, dtype=img_data['dtype']).reshape(img_data['shape'])

        for cam_name, img_data in data['depth_dict'].items():
            img_bytes = base64.b64decode(img_data['data'])
            depth_dict[cam_name] = np.frombuffer(img_bytes, dtype=img_data['dtype']).reshape(img_data['shape'])

        # Process images
        t0 = time.time()
        processed_rgb, processed_depth, is_contact = pcd_processor.get_policy_images(rgb_dict, depth_dict)
        # processed_rgb, processed_depth = pcd_processor.get_policy_images(rgb_dict, depth_dict)

        t_process = time.time() - t0

        # Encode results as base64
        result_rgb = {}
        result_depth = {}

        for cam_name, img in processed_rgb.items():
            result_rgb[cam_name] = {
                'data': base64.b64encode(img.tobytes()).decode('utf-8'),
                'dtype': str(img.dtype),
                'shape': img.shape
            }

        for cam_name, img in processed_depth.items():
            result_depth[cam_name] = {
                'data': base64.b64encode(img.tobytes()).decode('utf-8'),
                'dtype': str(img.dtype),
                'shape': img.shape
            }

        response = {
            'rgb_dict': result_rgb,
            'depth_dict': result_depth,
            'timing': {
                'process': t_process * 1000
            }
        }

        return jsonify(response)

    except Exception as e:
        import traceback
        return jsonify({
            'error': str(e),
            'traceback': traceback.format_exc()
        }), 500

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'processor_loaded': pcd_processor is not None
    })

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=5001,
                       help='Port to run server on')
    args = parser.parse_args()

    # Initialize processor before starting server
    initialize_processor()

    # Run server
    print(f"Starting point cloud processing server on port {args.port}")
    app.run(host='localhost', port=args.port, threaded=False)
