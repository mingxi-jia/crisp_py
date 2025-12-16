# Teleoperation Package

Unified teleoperation interface for robot control with support for multiple input devices.

## Architecture

```
teleop/
├── control_command.py       # ControlCommand and ControlMode definitions
├── base_teleop.py          # TeleopController abstract base class
├── keyboard_teleop.py      # Keyboard controller implementation
├── spacemouse_teleop.py    # Spacemouse controller implementation
├── keyboard_runner.py      # Keyboard teleop entry point
├── spacemouse_runner.py    # Spacemouse teleop entry point
└── README.md              # This file
```

## Features

- **Unified Interface**: Common base class for all input devices
- **Multiple Input Devices**: Keyboard, Spacemouse (easily extensible)
- **Control Modes**:
  - `DELTA_POSE`: Cartesian space control (implemented)
  - `JOINT_DELTA`: Joint space control (reserved for future)
- **Shared Functionality**: Robot setup, gripper control, reset handling

## Usage

### Keyboard Control

```bash
python -m teleop.keyboard_runner
```

**Controls:**
- **Position**: W/S (X), A/D (Y), Arrow Up/Down (Z)
- **Rotation**: I/K (Roll), J/L (Pitch), U/O (Yaw)
- **Gripper**: SPACE (toggle)
- **Reset**: R
- **Exit**: Ctrl+C

### Spacemouse Control

```bash
python -m teleop.spacemouse_runner
```

**Controls:**
- **6DOF**: Spacemouse (position + rotation)
- **Gripper**: Button 0 (toggle)
- **Reset**: R (keyboard)
- **Exit**: Ctrl+C

## Extending with New Devices

To add a new input device, create a subclass of `TeleopController`:

```python
from teleop.base_teleop import TeleopController
from teleop.control_command import ControlCommand, ControlMode
import numpy as np

class GamepadTeleopController(TeleopController):
    def __init__(self):
        super().__init__(ctrl_freq=10.0)
        self.gamepad = None

    def setup_input(self):
        """Initialize gamepad"""
        # Your gamepad initialization code
        pass

    def get_control_command(self) -> ControlCommand:
        """Read gamepad state and return control command"""
        # Read gamepad inputs
        # Convert to delta_pose or delta_joints
        delta_pose = np.array([dx, dy, dz, droll, dpitch, dyaw])

        return ControlCommand(
            delta_pose=delta_pose,
            gripper_toggle=button_a_pressed,
            reset_requested=button_start_pressed,
            mode=ControlMode.DELTA_POSE
        )

    def cleanup_input(self):
        """Cleanup gamepad"""
        pass

    def print_controls(self):
        """Print gamepad controls"""
        print("Gamepad Control Active")
        # Print control mapping
```

## Control Modes

### DELTA_POSE (Implemented)
- Cartesian space incremental control
- Delta pose: `[dx, dy, dz, droll, dpitch, dyaw]`
- Applied to current end-effector pose

### JOINT_DELTA (Reserved)
- Joint space incremental control
- Delta joints: `[dq1, dq2, ..., dq7]`
- To be implemented in `base_teleop.py::_apply_joint_delta()`

Example future usage:
```python
# In your custom controller
def get_control_command(self):
    if mode_switch_button_pressed:
        # Switch to joint control
        delta_joints = np.array([...])  # 7 joint deltas
        return ControlCommand(
            delta_joints=delta_joints,
            mode=ControlMode.JOINT_DELTA
        )
```

## Configuration

Controllers can be configured via constructor parameters:

```python
# Keyboard with custom scaling
controller = KeyboardTeleopController(
    ctrl_freq=20.0,          # 20 Hz control rate
    position_scale=0.005,    # 5mm per key press
    rotation_scale=0.1       # 0.1 rad per key press
)

# Spacemouse with custom deadzone
controller = SpacemouseTeleopController(
    ctrl_freq=15.0,
    action_scale=0.02,       # 2cm action scale
    deadzone=0.05            # Smaller deadzone
)
```

## Design Principles

1. **DRY (Don't Repeat Yourself)**: Common functionality in base class
2. **Open/Closed Principle**: Easy to extend with new devices, no need to modify existing code
3. **Single Responsibility**: Each controller handles only its input device
4. **Extensibility**: Reserved space for joint control and future features
