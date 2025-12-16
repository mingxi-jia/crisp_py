from abc import ABC, abstractmethod
import time
import numpy as np
from scipy.spatial.transform import Rotation as R

from crisp_py.robot import Robot, Pose
from crisp_py.gripper.gripper import Gripper, GripperConfig
from .control_command import ControlCommand, ControlMode


class TeleopController(ABC):
    """
    Abstract base class for teleoperation controllers.

    Handles common robot setup, gripper control, and main control loop.
    Subclasses implement device-specific input handling.
    """

    def __init__(self, ctrl_freq: float = 10.0):
        """
        Initialize teleop controller.

        Args:
            ctrl_freq: Control loop frequency in Hz
        """
        self.ctrl_freq = ctrl_freq
        self.robot = None
        self.gripper = None
        self.home_pose = None
        self.deviation_threshold = 0.05
        self.gripper_closed = False
        self.prev_gripper_toggle = False

    @abstractmethod
    def setup_input(self):
        """Initialize input device (keyboard, spacemouse, etc.)"""
        pass

    @abstractmethod
    def get_control_command(self) -> ControlCommand:
        """
        Get control command from input device.

        Returns:
            ControlCommand with delta_pose/delta_joints and control signals
        """
        pass

    @abstractmethod
    def cleanup_input(self):
        """Cleanup input device resources"""
        pass

    @abstractmethod
    def print_controls(self):
        """Print device-specific control instructions"""
        pass

    def setup_robot(self, namespace: str = ""):
        """
        Initialize robot and move to home pose.

        Args:
            namespace: Robot ROS namespace
        """
        self.robot = Robot(namespace=namespace)
        self.robot.wait_until_ready()
        print(f"Robot ready. Current joint values: {self.robot.joint_values}")
        print(self.robot.end_effector_pose)
        print(self.robot.joint_values)
        print("Going to home position...")
        self.robot.home()

        self.robot.controller_switcher_client.switch_controller("cartesian_impedance_controller")
        self.robot.cartesian_controller_parameters_client.load_param_config(
            file_path="config/control/spacemouse_cartesian_impedance.yaml"
        )
        time.sleep(2.0)

        print("Going to start position...")
        self.home_pose = Pose(
            position=np.array([0.6, 0., 0.35]),
            orientation=R.from_euler('XYZ', [np.pi, 0, 0])
        )
        self.robot.move_to(pose=self.home_pose, speed=0.15)

    def setup_gripper(self, config_path: str = "./config/gripper_right.yaml", namespace: str = "/right/gripper"):
        """
        Initialize gripper.

        Args:
            config_path: Path to gripper config YAML
            namespace: Gripper ROS namespace
        """
        gripper_config = GripperConfig.from_yaml(config_path)
        self.gripper = Gripper(gripper_config=gripper_config, namespace=namespace)
        self.gripper.wait_until_ready()
        self.gripper.set_target(1.0)
        self.gripper_closed = False
        print("Gripper ready")

    def handle_reset(self, target_pose, start_orientation):
        """
        Reset robot to home position.

        Args:
            target_pose: Current target pose
            start_orientation: Initial orientation to reset to

        Returns:
            Tuple of (target_pose, target_xyz, target_orientation) after reset
        """
        print("\n[RESET] Resetting to start position...")
        self.gripper.set_target(1.0)
        self.gripper_closed = False
        time.sleep(1.0)

        self.robot.move_to(pose=self.home_pose, speed=0.15)

        target_pose = self.robot.end_effector_pose
        target_xyz = np.array(target_pose.position)
        target_orientation = start_orientation.copy()
        print("Reset complete!\n")

        return target_pose, target_xyz, target_orientation

    def handle_gripper(self, gripper_rate):
        """
        Toggle gripper state.

        Args:
            gripper_rate: Rate object for gripper control
        """
        self.gripper_closed = not self.gripper_closed
        gripper_target = 0.0 if self.gripper_closed else 1.0
        print(f"Gripper {'closing' if self.gripper_closed else 'opening'}...")
        self.gripper.set_target(gripper_target)
        gripper_rate.sleep()
        time.sleep(0.5)

    def _apply_delta_pose(self, command: ControlCommand, target_xyz, target_orientation):
        """
        Apply delta pose control (Cartesian space).

        Args:
            command: Control command with delta_pose
            target_xyz: Current target position [x, y, z]
            target_orientation: Current target orientation [roll, pitch, yaw]

        Returns:
            Tuple of (new_xyz, new_orientation)
        """
        dx, dy, dz, droll, dpitch, dyaw = command.delta_pose

        curr_x, curr_y, curr_z = target_xyz
        curr_roll, curr_pitch, curr_yaw = target_orientation

        x = curr_x + dx
        y = curr_y + dy
        z = curr_z + dz
        roll = curr_roll + droll
        pitch = curr_pitch + dpitch
        yaw = curr_yaw + dyaw

        new_xyz = np.array([x, y, z])
        new_orientation = np.array([roll, pitch, yaw])

        # Only print if there's movement
        if dx != 0 or dy != 0 or dz != 0 or droll != 0 or dpitch != 0 or dyaw != 0:
            print(f"x={x:.4f}\ty={y:.4f}\tz={z:.4f}\troll={roll:.4f}\tpitch={pitch:.4f}\tyaw={yaw:.4f}")

        return new_xyz, new_orientation

    def _apply_joint_delta(self, command: ControlCommand):
        """
        Apply joint delta control (Joint space).

        Reserved for future implementation.

        Args:
            command: Control command with delta_joints

        Returns:
            New joint positions
        """
        # TODO: Implement joint-space control
        # Current joint values: self.robot.joint_values
        # Apply delta: new_joints = current_joints + command.delta_joints
        # Send command: self.robot.set_joint_target(new_joints)
        raise NotImplementedError("Joint control not yet implemented")

    def apply_control(self, command: ControlCommand, target_xyz, target_orientation):
        """
        Apply control command based on mode.

        Args:
            command: Control command
            target_xyz: Current target position
            target_orientation: Current target orientation

        Returns:
            Updated (target_xyz, target_orientation) for DELTA_POSE mode

        Raises:
            ValueError: If control mode is unsupported
        """
        if command.mode == ControlMode.DELTA_POSE:
            return self._apply_delta_pose(command, target_xyz, target_orientation)
        elif command.mode == ControlMode.JOINT_DELTA:
            return self._apply_joint_delta(command)
        else:
            raise ValueError(f"Unsupported control mode: {command.mode}")

    def run(self):
        """Main control loop."""
        try:
            # Setup
            self.setup_input()
            self.setup_robot()
            self.setup_gripper()
            self.print_controls()

            # Initialize state
            target_pose = self.robot.end_effector_pose
            target_xyz = np.array(target_pose.position)
            target_orientation = np.array(target_pose.orientation.as_euler('XYZ'))
            start_orientation = target_orientation.copy()

            arm_rate = self.robot.node.create_rate(self.ctrl_freq)
            gripper_rate = self.gripper.node.create_rate(self.ctrl_freq)

            # Main loop
            while True:
                command = self.get_control_command()

                # Handle reset
                if command.reset_requested:
                    target_pose, target_xyz, target_orientation = self.handle_reset(
                        target_pose, start_orientation
                    )
                    continue

                # Check for large deviation
                if np.linalg.norm(target_pose.position - self.robot.end_effector_pose.position) > self.deviation_threshold:
                    arm_rate.sleep()
                    continue

                # Apply control based on mode
                target_xyz, target_orientation = self.apply_control(
                    command, target_xyz, target_orientation
                )

                # Update target pose
                target_pose.position = target_xyz
                target_pose.orientation = R.from_euler('XYZ', target_orientation)
                self.robot.set_target(pose=target_pose)

                # Handle gripper toggle
                if command.gripper_toggle and not self.prev_gripper_toggle:
                    self.handle_gripper(gripper_rate)
                self.prev_gripper_toggle = command.gripper_toggle

                arm_rate.sleep()

        finally:
            # Cleanup
            self.cleanup_input()
            if self.robot:
                self.robot.home()
                self.robot.shutdown()
