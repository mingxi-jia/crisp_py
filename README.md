![crisp_py_logo](https://github.com/user-attachments/assets/374ae11a-4d82-4bb7-8b93-152bde13aa5b)

[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
![MIT Badge](https://img.shields.io/badge/MIT-License-blue?style=flat)
<a href="https://utiasDSL.github.io/crisp_controllers/"><img alt="Static Badge" src="https://img.shields.io/badge/docs-passing-blue?style=flat&link=https%3A%2F%2FutiasDSL.github.io%2Fcrisp_controllers%2F"></a>
<a href="https://github.com/utiasDSL/crisp_py/actions/workflows/ruff_ci.yml"><img src="https://github.com/utiasDSL/crisp_py/actions/workflows/ruff_ci.yml/badge.svg"/></a>
<a href="https://github.com/utiasDSL/crisp_py/actions/workflows/pixi_ci.yml"><img src="https://github.com/utiasDSL/crisp_py/actions/workflows/pixi_ci.yml/badge.svg"/></a>
<a href="https://utiasDSL.github.io/crisp_controllers#citing"><img alt="Static Badge" src="https://img.shields.io/badge/arxiv-cite-b31b1b?style=flat"></a>

*CRISP_PY /krɪspi/*, a python package to interface with robots using [CRISP controllers](https://github.com/utiasDSL/crisp_controllers). Check the [project website](https://utiasdsl.github.io/crisp_controllers/) for further information!

![crisp_py](https://github.com/user-attachments/assets/e4cbf5fd-6ba7-4d7c-917a-bbb78d79ab10)

# install pytorch3d
1. pytorch3d problem
CC=/usr/bin/gcc CXX=/usr/bin/g++ pip install -e . --no-build-isolation
2. import problem
check PATH and PYTHON: should not have system ros
```
export PYTHONPATH=/home/mingxi/miniconda3/envs/ros_env/lib/python311/site-packages:/home/mingxi/miniconda3/envs/ros_env/lib/python3.11/site-packages:/opt/openrobots/lib/python3.10/site-packages:/home/mingxi/ros2_ws/install/realsense2_camera_msgs/local/lib/python3.10/dist-packages:/home/mingxi/ros2_ws/install/pymoveit2/local/lib/python3.10/dist-packages:/home/mingxi/ros2_ws/install/franka_gripper/local/lib/python3.10/dist-packages:/home/mingxi/ros2_ws/install/franka_msgs/local/lib/python3.10/dist-packages:/home/mingxi/ros2_ws/build/easy_handeye2:/home/mingxi/ros2_ws/install/easy_handeye2/lib/python3.10/site-packages:/home/mingxi/ros2_ws/install/easy_handeye2_msgs/local/lib/python3.10/dist-packages:/home/mingxi/ros2_ws/install/aruco_msgs/local/lib/python3.10/dist-packages
```

```
export PATH=/home/mingxi/miniconda3/envs/ros_env/bin:/home/mingxi/miniconda3/condabin:/home/mingxi/.pixi/bin:/opt/openrobots/bin:/opt/openrobots/bin:/home/mingxi/ros2_ws/install/libfranka/bin:/home/mingxi/.local/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/games:/usr/local/games:/snap/bin:/snap/bin:/home/mingxi/.vscode/extensions/ms-python.debugpy-2025.16.0-linux-x64/bundled/scripts/noConfigScripts:/home/mingxi/.config/Code/User/globalStorage/github.copilot-chat/debugCommand
```
3. import error (no control msg)
conda install ros-humble-ros2-control ros-humble-ros2-controllers