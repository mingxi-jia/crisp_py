# Guide
```
./scripts/start_all.sh --ctrl-space joint --home
python -m deploy.deploy_spacemouse

```

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