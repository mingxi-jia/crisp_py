"""Simple example to control the gripper."""

# %%
import time
from pathlib import Path

import yaml

from crisp_py.gripper.gripper import Gripper, GripperConfig

project_root_path = Path(".")

gripper_config = GripperConfig.from_yaml("/home/mrping/mingxi_ws/crisp/crisp_py/config/gripper_right.yaml")
# %%

gripper = Gripper(gripper_config=gripper_config, namespace="/right/gripper")
gripper.wait_until_ready()

# %%
freq = 1.0
rate = gripper.node.create_rate(freq)
t = 0.0
while t < 2.0:
    print(f"gripper.value: {gripper.value}")
    print(f"gripper.torque: {gripper.torque}")
    rate.sleep()
    t += 1.0 / freq

# franka gripper only support width \in {0, 1}

width = 0.
print(f"moving to {width}")
gripper.set_target(width)
rate.sleep()
# time.sleep(3.0)

width = 1.0
print(f"moving to {width}")
gripper.set_target(width)
rate.sleep()
# time.sleep(3.0)

gripper.close()
rate.sleep()

gripper.open()
rate.sleep()

gripper.shutdown()

