"""ROS2 utility classes for joint state subscription."""

import numpy as np
from sensor_msgs.msg import JointState


class JointStateSubscriber:
    """A simple ROS2 subscriber to get joint states from /joint_states topic."""

    def __init__(self, node, joint_names: list[str], topic: str = "/joint_states"):
        """Initialize the joint state subscriber.

        Args:
            node: ROS2 node to attach the subscription to.
            joint_names: List of joint names to track (in desired order).
            topic: Topic name to subscribe to.
        """
        self._node = node
        self._received = False
        self.joint_array = None

        self._subscription = node.create_subscription(
            JointState,
            topic,
            self._callback,
            10
        )

    def _callback(self, msg: JointState):
        """Update joint positions from the message."""
        joint_names = msg.name
        joint_positions = msg.position
        sorted_indices = sorted(range(len(joint_names)), key=lambda i: joint_names[i])
        sorted_positions = [joint_positions[i] for i in sorted_indices]
        self.joint_array = np.array(sorted_positions, dtype=np.float32)
        self._received = True

    @property
    def joint_values(self) -> np.ndarray:
        """Get joint values, skipping world joint (first element) for Franka."""
        return self.joint_array[1:] if self.joint_array is not None else None

    @property
    def is_ready(self) -> bool:
        """Check if at least one message has been received."""
        return self._received
