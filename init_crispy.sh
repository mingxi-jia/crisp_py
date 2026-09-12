myconda
conda activate ros_env
export PYTHONPATH="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)${PYTHONPATH:+:$PYTHONPATH}"

export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-100}"

export CRISP_CONFIG_PATH="${CRISP_CONFIG_PATH:-/home/mingxi/mingxi_ws/crisp/crisp_py/config}"

export ROS_NETWORK_INTERFACE="${ROS_NETWORK_INTERFACE:-enp4s0}"
export RMW_IMPLEMENTATION="${RMW_IMPLEMENTATION:-rmw_cyclonedds_cpp}"
export CYCLONEDDS_URI="${CYCLONEDDS_URI:-/home/mingxi/mingxi_ws/crisp/crisp_py/config/cyclone_config.xml}"
