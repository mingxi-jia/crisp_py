"""ROS2 utility classes for joint state subscription."""

import numpy as np
from sensor_msgs.msg import JointState


class JointStateSubscriber:
    """A simple ROS2 subscriber to get joint states from /joint_states topic."""

    def __init__(self, node, topic: str = "/joint_states"):
        """Initialize the joint state subscriber.

        Args:
            node: ROS2 node to attach the subscription to.
            joint_names: List of joint names to track (in desired order).
            topic: Topic name to subscribe to.
        """
        self._node = node
        self._franka_received = False
        self._robotiq_received = False
        self.franak_joint_array = None
        self.gripper_joint_array = None
        self.gripper_width = 0.7894070484581498
        self.franka_joint_names = ["fr3_joint1", "fr3_joint2", "fr3_joint3", "fr3_joint4", "fr3_joint5", "fr3_joint6", "fr3_joint7"]
        self.gripper_joint_names = ['gripper_joint']

        self._subscription = node.create_subscription(
            JointState,
            topic,
            self._franka_callback,
            10
        )
        self._robotiq_subscription = node.create_subscription(
            JointState,
            "/gripper/gripper_state",
            self._robotiq_callback,
            10
        )

    def _franka_callback(self, msg: JointState):
        """Update joint positions from the message."""
        joint_names = msg.name
        joint_positions = msg.position
        # print(joint_names)
        joint_array = []
        for i, j in enumerate(joint_names):
            if j in self.franka_joint_names:
                joint_array.append(joint_positions[i])

        self.franak_joint_array = np.array(joint_array, dtype=np.float32)
        self._franka_received = True

    def _robotiq_callback(self, msg: JointState):
        """Update joint positions from the message."""
        joint_names = msg.name
        joint_positions = msg.position
        gripper_state = []
        for i, j in enumerate(joint_names):
            if j in self.gripper_joint_names:
                gripper_state.append(joint_positions[i])

        self.gripper_joint_array = np.array(gripper_state, dtype=np.float32)
        self._robotiq_received = True

    @property
    def joint_values(self) -> np.ndarray:
        """Get joint values, skipping world joint (first element) for Franka."""
        return self.franak_joint_array
    
    @property
    def gripper_state(self) -> np.ndarray:
        """Get joint values, skipping world joint (first element) for Franka."""
        return np.round(self.gripper_joint_array)

    @property
    def is_ready(self) -> bool:
        """Check if at least one message has been received."""
        return self._franka_received and self._robotiq_received
