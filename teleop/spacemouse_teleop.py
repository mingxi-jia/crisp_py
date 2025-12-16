import threading
import numpy as np
from pynput import keyboard
from std_msgs.msg import Int32MultiArray

from crisp_py.spacemouse import Spacemouse
from .base_teleop import TeleopController
from .control_command import ControlCommand, ControlMode


class SpacemouseTeleopController(TeleopController):
    """
    Spacemouse-based teleoperation controller.

    Controls:
        Spacemouse - 6DOF control (position + rotation)
        Button 0 - Toggle gripper open/close
        R - Reset to start position (keyboard)
    """

    def __init__(self, ctrl_freq: float = 10.0, action_scale: float = 0.01, deadzone: float = 0.1):
        """
        Initialize spacemouse teleop controller.

        Args:
            ctrl_freq: Control loop frequency in Hz
            action_scale: Action scaling factor
            deadzone: Deadzone for spacemouse input
        """
        super().__init__(ctrl_freq)
        self.action_scale = action_scale
        self.deadzone = deadzone
        self.deviation_threshold = 0.02

        # Shared state for keyboard reset trigger
        self.reset_requested = {'flag': False}
        self.lock = threading.Lock()
        self.listener = None
        self.spacemouse = None
        self.spacemouse_pub = None

    def on_press(self, key):
        """Handle key press events for reset."""
        with self.lock:
            try:
                if key.char == 'r':
                    self.reset_requested['flag'] = True
            except AttributeError:
                pass

    def on_release(self, key):
        """Handle key release events."""
        pass

    def setup_input(self):
        """Initialize spacemouse and keyboard listener."""
        self.spacemouse = Spacemouse(deadzone=self.deadzone)
        self.spacemouse.__enter__()

        # Start keyboard listener for reset
        self.listener = keyboard.Listener(on_press=self.on_press, on_release=self.on_release)
        self.listener.start()

    def get_control_command(self) -> ControlCommand:
        """
        Get control command from spacemouse input.

        Returns:
            ControlCommand with delta_pose and control signals
        """
        with self.lock:
            reset_flag = self.reset_requested['flag']
            if reset_flag:
                self.reset_requested['flag'] = False

        # Get action from spacemouse
        spacemouse_eef_action = self.spacemouse.get_motion_state_transformed()
        button_pressed = self.spacemouse.is_button_pressed(0)

        # Scale actions
        dx, dy, dz, droll, dpitch, dyaw = spacemouse_eef_action * self.action_scale

        # Apply custom yaw scaling (from original spacemouse_example.py line 148)
        dyaw = -dyaw * 2

        # Publish spacemouse signals
        if self.spacemouse_pub:
            spacemouse_msg = Int32MultiArray()
            gripper_toggle = 1 if button_pressed else 0
            # Convert float deltas to int (sign only)
            dx_int = (1 if dx > 0 else -1 if dx < 0 else 0)
            dy_int = (1 if dy > 0 else -1 if dy < 0 else 0)
            dz_int = (1 if dz > 0 else -1 if dz < 0 else 0)
            droll_int = (1 if droll > 0 else -1 if droll < 0 else 0)
            dpitch_int = (1 if dpitch > 0 else -1 if dpitch < 0 else 0)
            dyaw_int = (1 if dyaw > 0 else -1 if dyaw < 0 else 0)
            spacemouse_msg.data = [dx_int, dy_int, dz_int, droll_int, dpitch_int, dyaw_int, gripper_toggle, 0]
            self.spacemouse_pub.publish(spacemouse_msg)

        delta_pose = np.array([dx, dy, dz, droll, dpitch, dyaw])

        return ControlCommand(
            delta_pose=delta_pose,
            gripper_toggle=button_pressed,
            reset_requested=reset_flag,
            mode=ControlMode.DELTA_POSE
        )

    def cleanup_input(self):
        """Cleanup spacemouse and keyboard listener."""
        if self.spacemouse:
            self.spacemouse.__exit__(None, None, None)
        if self.listener:
            self.listener.stop()

    def print_controls(self):
        """Print spacemouse control instructions."""
        print("\n" + "=" * 50)
        print("Spacemouse Control Active")
        print("=" * 50)
        print("\nControls:")
        print("  Spacemouse - 6DOF control (position + rotation)")
        print("  Button 0 - Toggle gripper open/close")
        print("  R - Reset to start position")
        print("  Ctrl+C - Exit")
        print("=" * 50 + "\n")

    def setup_robot(self, namespace: str = ""):
        """
        Initialize robot and setup spacemouse signal publisher.

        Args:
            namespace: Robot ROS namespace
        """
        super().setup_robot(namespace)
        # Create spacemouse signal publisher
        # Format: [dx, dy, dz, droll, dpitch, dyaw, gripper_toggle, reset]
        self.spacemouse_pub = self.robot.node.create_publisher(Int32MultiArray, '/teleop/signals', 30)

    def _apply_delta_pose(self, command: ControlCommand, target_xyz, target_orientation):
        """
        Apply delta pose control with z-clipping for spacemouse.

        Args:
            command: Control command with delta_pose
            target_xyz: Current target position [x, y, z]
            target_orientation: Current target orientation [roll, pitch, yaw]

        Returns:
            Tuple of (new_xyz, new_orientation)
        """
        # Call parent implementation
        new_xyz, new_orientation = super()._apply_delta_pose(command, target_xyz, target_orientation)

        # Clip z to be above table height (from spacemouse_example.py line 151)
        new_xyz[2] = max(new_xyz[2], 0.02)

        return new_xyz, new_orientation
