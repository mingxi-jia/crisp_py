"""A ROS2-based point cloud processing example."""

# %%
import time
from pathlib import Path
import rclpy
import open3d as o3d
import threading
import os

from crisp_py.camera.pointcloud import PointCloudManager
from crisp_py.robot import Robot
from crisp_py.gripper.gripper import Gripper, GripperConfig

from diff_eval_utils.ros_utils import JointStateSubscriber
from diff_eval_utils.diffusion_transforms import get_pose_from_robot
from geometry_msgs.msg import PoseStamped
from scipy.spatial.transform import Rotation as R
from crisp_py.robot import Pose
import sys
import numpy as np
toolbox_path = '/home/mingxi/mingxi_ws/handpi/diffusion_policy/robotool'
sys.path.append(toolbox_path)
from robot_filter.arm_segmentor import RobotArmSegmentation
from hand_tool.trajectory_loader import ObservationProcessor
# %%

NO_ROBOT = True


class PoseSubscriber:
    """A simple ROS2 subscriber to get end-effector pose from /current_pose topic."""

    def __init__(self, node, topic: str = "/current_pose"):
        """Initialize the pose subscriber.

        Args:
            node: ROS2 node to attach the subscription to.
            topic: Topic name to subscribe to.
        """
        self._node = node
        self._received = False
        self._pose = None  # [x, y, z, qx, qy, qz, qw]

        self._subscription = node.create_subscription(
            PoseStamped,
            topic,
            self._callback,
            10
        )

    def _callback(self, msg: PoseStamped):
        """Update pose from the message."""
        pos = msg.pose.position
        ori = msg.pose.orientation
        self._pose = np.array([
            pos.x, pos.y, pos.z,
            ori.x, ori.y, ori.z, ori.w
        ], dtype=np.float32)
        self._received = True

    @property
    def pose(self) -> np.ndarray:
        """Get pose as [x, y, z, qx, qy, qz, qw]."""
        return self._pose

    @property
    def is_ready(self) -> bool:
        """Check if at least one message has been received."""
        return self._received
def main():
    """Test the PointCloudManager."""
    rclpy.init()


    # Get config path
    config_path = "/home/mingxi/mingxi_ws/handpi/diffusion_policy/robotool/robot_configs/camera_info.yaml"
    print(f"debug: config_path = {config_path}")
    
    # Create point cloud manager
    manager = PointCloudManager(str(config_path))
    if not NO_ROBOT:
        joint_state_subscriber = JointStateSubscriber(manager, topic="/joint_states")
        pose_subscriber = PoseSubscriber(manager, topic="/current_pose")

        # Initialize robot filter
        robot_seg = RobotArmSegmentation()
        robot_seg.load_urdf(os.path.join(toolbox_path, "robot_filter/panda_description/urdf/panda_arm_hand.urdf"))

    obs_processor = ObservationProcessor()

    spin_thread = threading.Thread(target=rclpy.spin, args=(manager,), daemon=True)
    spin_thread.start()

    # Create non-blocking visualizer
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="Real-time Point Cloud", width=800, height=600)
    pcd_o3d = o3d.geometry.PointCloud()
    vis.add_geometry(pcd_o3d)
    vis.get_render_option().point_size = 15.0 

    add_ground = True
    if add_ground:
        x_range = [0.3, 0.9]
        y_range = [-0.3, 0.3]
        z_ground = 0.01
        
        # Create a mesh for the ground
        ground_mesh = o3d.geometry.TriangleMesh()
        vertices = np.array([
            [x_range[0], y_range[0], z_ground],  # corner 1
            [x_range[1], y_range[0], z_ground],  # corner 2
            [x_range[1], y_range[1], z_ground],  # corner 3
            [x_range[0], y_range[1], z_ground],  # corner 4
        ])
        triangles = np.array([
            [0, 1, 2],  # first triangle
            [0, 2, 3],  # second triangle
        ])
        ground_mesh.vertices = o3d.utility.Vector3dVector(vertices)
        ground_mesh.triangles = o3d.utility.Vector3iVector(triangles)
        
        # Color the ground (e.g., light gray)
        ground_mesh.paint_uniform_color([0.9, 0.2, 0.3])
        
        # Add to visualizer
        vis.add_geometry(ground_mesh)

    # Initialize flag for first update
    geometry_added = False
    collect_hand_data = False
    policy_obs_mode = True  # New mode for policy observation visualization

    # Wait and visualize
    print("Waiting for point clouds...")
    print("Press Q or close the window to exit")

    try:
        while True:
            time.sleep(0.01)  # Faster update rate for real-time visualization
            pcd = manager.get_latest_pointcloud()

            # Process and visualize point cloud
            if pcd is not None:
                # Filter and render point cloud
                obs_processor.robot_filter._init_pre_samples()

                if policy_obs_mode and not NO_ROBOT:
                    # Policy observation mode: uses get_policy_obs for exact policy input visualization
                    if pose_subscriber.is_ready and joint_state_subscriber.is_ready:
                        raw_pose = pose_subscriber.pose
                        # Convert robot EE pose to fingertip frame
                        robot_pose = Pose(
                            position=raw_pose[:3],
                            orientation=R.from_quat(raw_pose[3:])
                        )
                        pose = get_pose_from_robot(robot_pose)

                        # Combine gripper state with joint values: [gripper_state, joint1, ..., joint7]
                        gripper_state = joint_state_subscriber.gripper_state[0] if joint_state_subscriber.gripper_state is not None else 0.0
                        joint = np.concatenate([[gripper_state], joint_state_subscriber.joint_values])

                        # Call get_policy_obs which matches trajectory_loader.py exactly
                        render_pcd, _ = obs_processor.get_policy_obs(pcd, pose, joint)
                        print(f"render_pcd.shape: {render_pcd.shape}")
                        pcd_o3d.points = o3d.utility.Vector3dVector(render_pcd[:, :3])
                        pcd_o3d.colors = o3d.utility.Vector3dVector(render_pcd[:, 3:])
                    else:
                        # Not ready yet, show filtered point cloud
                        pcd_filtered = obs_processor.filter_pcd_by_workspace(pcd)
                        pcd_o3d.points = o3d.utility.Vector3dVector(pcd_filtered[:, :3])
                        pcd_o3d.colors = o3d.utility.Vector3dVector(pcd_filtered[:, 3:])
                elif collect_hand_data or NO_ROBOT:
                    pcd = obs_processor.filter_pcd_by_workspace(pcd)
                    robo_pcd = obs_processor.get_render_pcd(pcd, np.array([0.5, 0.0, 0.25, 0, np.pi/12, 0, 0]), 1, render_type='gripper')
                    robo_colors = robo_pcd[:, 3:]
                    pcd_o3d.points = o3d.utility.Vector3dVector(robo_pcd[:, :3])
                    pcd_o3d.colors = o3d.utility.Vector3dVector(robo_colors)
                else:
                    pcd = obs_processor.filter_pcd_by_workspace(pcd)
                    # print(f"joint_state_subscriber.joint_values {joint_state_subscriber.joint_values}")
                    robo_pcd = obs_processor.robot_filter.get_robot_pcd(joint_state_subscriber.joint_values)
                    # print(joint_state_subscriber.joint_values)
                    robo_colors = np.ones((robo_pcd.shape[0], 3)) * [1.0, 0.5, 0.0]  # Orange
                    pcd_o3d.points = o3d.utility.Vector3dVector(np.concatenate([pcd[:, :3], robo_pcd], axis=0))
                    pcd_o3d.colors = o3d.utility.Vector3dVector(np.concatenate([pcd[:, 3:], robo_colors], axis=0))


                # Update visualization
                vis.update_geometry(pcd_o3d)

                if not geometry_added:
                    vis.reset_view_point(True)
                    geometry_added = True

            # Poll events and update renderer (non-blocking)
            if not vis.poll_events():
                break
            vis.update_renderer()

    except KeyboardInterrupt:
        print("\nStopping visualization...")
    except Exception as e:
        print(f"\nError occurred: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Cleanup
        try:
            vis.destroy_window()
        except:
            pass
        try:
            manager.destroy_node()
        except:
            pass
        try:
            if rclpy.ok():
                rclpy.shutdown()
        except:
            pass


if __name__ == "__main__":
    main()
