"""Utility functions for Pink IK solver."""

import os
from dataclasses import dataclass
from typing import Optional

import pinocchio as pin


@dataclass
class Robot:
    """Container for robot model and data."""
    model: pin.Model
    data: pin.Data
    collision_model: pin.GeometryModel
    visual_model: pin.GeometryModel


def load_robot_from_urdf(
    urdf_path: str,
    mesh_package_dirs: Optional[list[str]] = None
) -> Robot:
    """Load robot model from local URDF file.

    Args:
        urdf_path: Path to the URDF file
        mesh_package_dirs: List of directories to search for mesh packages.
            If None, common ROS package locations will be searched.

    Returns:
        Robot object with model, data, collision_model, visual_model
    """
    if not os.path.exists(urdf_path):
        raise FileNotFoundError(f"URDF file not found: {urdf_path}")

    # Default package directories for ROS meshes
    if mesh_package_dirs is None:
        mesh_package_dirs = []

        # Common ROS package locations
        potential_dirs = [
            "/opt/ros/humble/share",
            "/opt/ros/galactic/share",
            "/opt/ros/foxy/share",
            "/opt/ros/noetic/share",
            os.path.expanduser("~/ros2_ws/install"),
            os.path.expanduser("~/catkin_ws/install"),
            "/home/ros/ros2_ws/install",
        ]

        # Also check for franka_description in common locations
        franka_dirs = [
            "/opt/ros/humble/share/franka_description",
            "/home/ros/ros2_ws/install/franka_description/share",
        ]

        for d in potential_dirs:
            if os.path.isdir(d):
                mesh_package_dirs.append(d)

        # Add parent dirs of franka_description if found
        for d in franka_dirs:
            if os.path.isdir(d):
                parent = os.path.dirname(os.path.dirname(d))
                if parent not in mesh_package_dirs:
                    mesh_package_dirs.append(parent)

    print(f"Loading URDF from: {urdf_path}")
    print(f"Mesh package directories: {mesh_package_dirs}")

    # Build models from URDF
    if mesh_package_dirs:
        model, collision_model, visual_model = pin.buildModelsFromUrdf(
            urdf_path,
            package_dirs=mesh_package_dirs,
            root_joint=None
        )
    else:
        model, collision_model, visual_model = pin.buildModelsFromUrdf(
            urdf_path,
            root_joint=None
        )

    data = model.createData()

    return Robot(
        model=model,
        data=data,
        collision_model=collision_model,
        visual_model=visual_model
    )
