"""A ROS2-based point cloud processing example."""

# %%
import time
from pathlib import Path
import rclpy
import open3d as o3d
import os

from crisp_py.camera.pointcloud import PointCloudManager
from crisp_py.robot import Robot
from crisp_py.gripper.gripper import Gripper, GripperConfig

import sys
toolbox_path = '/home/mingxi/mingxi_ws/handpi/robot-vision-toolbox'
sys.path.append(toolbox_path)
from robot_filter.arm_segmentor import RobotArmSegmentation
# %%
def main():
    """Test the PointCloudManager."""
    rclpy.init()

    # Initialize robot
    robot = Robot(namespace="")
    robot.wait_until_ready()
    print(f"Robot ready. Current joint values: {robot.joint_values}")

    # Initialize gripper
    gripper_config = GripperConfig.from_yaml("./config/gripper_right.yaml")
    gripper = Gripper(gripper_config=gripper_config, namespace="/right/gripper")
    gripper.wait_until_ready()
    print("Gripper ready")

    # Get config path
    config_path = Path(__file__).parent.parent / "config" / "camera_info.yaml"
    print(f"debug: config_path = {config_path}")
    
    # Create point cloud manager
    manager = PointCloudManager(str(config_path))

    # Initialize robot filter
    robot_seg = RobotArmSegmentation()
    robot_seg.load_urdf(os.path.join(toolbox_path, "robot_filter/panda_description/urdf/panda_arm_hand.urdf"))

    # Spin in background thread to receive messages
    import threading
    # spin_thread = threading.Thread(target=rclpy.spin, args=(manager,), daemon=True)
    spin_thread = threading.Thread(target=rclpy.spin, args=(manager,))
    spin_thread.start()

    # Wait and visualize
    print("Waiting for point clouds...")
    for i in range(10):
        time.sleep(0.1)
        pcd = manager.get_latest_pointcloud()


        # Visualize filtered point cloud
        if pcd is not None:

            # crop pcd using workspace
            

            # Get current joint state from robot
            joint_state = robot.joint_values
            filtered_pcd = robot_seg.segment(pcd, joint_state)

            print(f"Iteration {i+1}: Filtered PCD has {len(filtered_pcd)} points")
            filtered_pcd_o3d = o3d.geometry.PointCloud()
            filtered_pcd_o3d.points = o3d.utility.Vector3dVector(filtered_pcd[:, :3])
            filtered_pcd_o3d.colors = o3d.utility.Vector3dVector(filtered_pcd[:, 3:])
            o3d.visualization.draw_geometries(
                [filtered_pcd_o3d],
                window_name="Filtered Point Cloud",
                width=800,
                height=600
            )
        elif pcd is not None:
            print(f"Iteration {i+1}: No filtering applied, showing original PCD")

    # Cleanup
    robot.shutdown()
    manager.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
