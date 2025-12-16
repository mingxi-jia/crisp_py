import threading
import numpy as np
from pynput import keyboard
from std_msgs.msg import Int32MultiArray

from .base_teleop import TeleopController
from .control_command import ControlCommand, ControlMode


class KeyboardTeleopController(TeleopController):
    """
    Keyboard-based teleoperation controller.

    Position Controls:
        W/S - Forward/Backward (X axis)
        A/D - Left/Right (Y axis)
        Arrow Up/Down - Up/Down (Z axis)

    Rotation Controls:
        I/K - Roll +/-
        J/L - Pitch +/-
        U/O - Yaw +/-

    Other Controls:
        SPACE - Toggle gripper open/close
        R - Reset to start position
    """

    def __init__(self, ctrl_freq: float = 10.0, position_scale: float = 0.01, rotation_scale: float = 0.05):
        """
        Initialize keyboard teleop controller.

        Args:
            ctrl_freq: Control loop frequency in Hz
            position_scale: Position delta per key press (meters)
            rotation_scale: Rotation delta per key press (radians)
        """
        super().__init__(ctrl_freq)
        self.position_scale = position_scale
        self.rotation_scale = rotation_scale
        self.deviation_threshold = 0.05

        # Shared state for keyboard input
        self.key_state = {
            # Position controls (WASD + QE)
            'w': False,  # +X (forward)
            's': False,  # -X (backward)
            'a': False,  # +Y (left)
            'd': False,  # -Y (right)
            # Z-axis controls (Arrow Up/Down)
            'up': False,    # +Z (up)
            'down': False,  # -Z (down)
            # Rotation controls (IJKL + UO)
            'i': False,  # +roll
            'k': False,  # -roll
            'j': False,  # +pitch
            'l': False,  # -pitch
            'u': False,  # +yaw
            'o': False,  # -yaw
            # Gripper control
            'space': False,  # toggle gripper
            # Reset
            'r': False,  # reset to start position
        }
        self.lock = threading.Lock()
        self.listener = None
        self.keyboard_pub = None

    def on_press(self, key):
        """Handle key press events."""
        with self.lock:
            try:
                if hasattr(key, 'char') and key.char in self.key_state:
                    self.key_state[key.char] = True
            except AttributeError:
                pass
            if key == keyboard.Key.space:
                self.key_state['space'] = True
            if key == keyboard.Key.up:
                self.key_state['up'] = True
            if key == keyboard.Key.down:
                self.key_state['down'] = True

    def on_release(self, key):
        """Handle key release events."""
        with self.lock:
            try:
                if hasattr(key, 'char') and key.char in self.key_state:
                    self.key_state[key.char] = False
            except AttributeError:
                pass
            if key == keyboard.Key.space:
                self.key_state['space'] = False
            if key == keyboard.Key.up:
                self.key_state['up'] = False
            if key == keyboard.Key.down:
                self.key_state['down'] = False

    def setup_input(self):
        """Initialize keyboard listener."""
        self.listener = keyboard.Listener(on_press=self.on_press, on_release=self.on_release)
        self.listener.start()

    def get_control_command(self) -> ControlCommand:
        """
        Get control command from keyboard input.

        Returns:
            ControlCommand with delta_pose and control signals
        """
        with self.lock:
            # Calculate position delta from WASD + Arrow keys
            dx = (1 if self.key_state['s'] else 0) - (1 if self.key_state['w'] else 0)
            dy = (1 if self.key_state['d'] else 0) - (1 if self.key_state['a'] else 0)
            dz = (1 if self.key_state['up'] else 0) - (1 if self.key_state['down'] else 0)

            # Calculate rotation delta from IJKL + UO
            droll = (1 if self.key_state['i'] else 0) - (1 if self.key_state['k'] else 0)
            dpitch = (1 if self.key_state['j'] else 0) - (1 if self.key_state['l'] else 0)
            dyaw = (1 if self.key_state['u'] else 0) - (1 if self.key_state['o'] else 0)

            # Handle gripper toggle and reset
            space_pressed = self.key_state['space']
            reset_pressed = self.key_state['r']
            if reset_pressed:
                self.key_state['r'] = False

        # Publish keyboard signals
        if self.keyboard_pub:
            keyboard_msg = Int32MultiArray()
            gripper_toggle = 1 if space_pressed else 0
            keyboard_msg.data = [dx, dy, dz, droll, dpitch, dyaw, gripper_toggle, int(reset_pressed)]
            self.keyboard_pub.publish(keyboard_msg)

        # Scale deltas
        delta_pose = np.array([
            dx * self.position_scale,
            dy * self.position_scale,
            dz * self.position_scale,
            droll * self.rotation_scale,
            dpitch * self.rotation_scale,
            dyaw * self.rotation_scale
        ])

        return ControlCommand(
            delta_pose=delta_pose,
            gripper_toggle=space_pressed,
            reset_requested=reset_pressed,
            mode=ControlMode.DELTA_POSE
        )

    def cleanup_input(self):
        """Cleanup keyboard listener."""
        if self.listener:
            self.listener.stop()

    def print_controls(self):
        """Print keyboard control instructions."""
        print("\n" + "=" * 50)
        print("Keyboard Control Active")
        print("=" * 50)
        print("\nPosition Controls:")
        print("  W/S - Forward/Backward (X axis)")
        print("  A/D - Left/Right (Y axis)")
        print("  Arrow Up/Down - Up/Down (Z axis)")
        print("\nRotation Controls:")
        print("  I/K - Roll +/-")
        print("  J/L - Pitch +/-")
        print("  U/O - Yaw +/-")
        print("\nOther Controls:")
        print("  SPACE - Toggle gripper open/close")
        print("  R - Reset to start position")
        print("  Ctrl+C - Exit")
        print("=" * 50 + "\n")

    def setup_robot(self, namespace: str = ""):
        """
        Initialize robot and setup keyboard signal publisher.

        Args:
            namespace: Robot ROS namespace
        """
        super().setup_robot(namespace)
        # Create keyboard signal publisher
        # Format: [dx, dy, dz, droll, dpitch, dyaw, gripper_toggle, reset]
        self.keyboard_pub = self.robot.node.create_publisher(Int32MultiArray, '/teleop/signals', 30)
