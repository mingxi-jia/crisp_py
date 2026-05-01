SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"

export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-100}"

export PYTHONPATH="${SCRIPT_DIR}${PYTHONPATH:+:$PYTHONPATH}"

export CRISP_CONFIG_PATH="${CRISP_CONFIG_PATH:-${SCRIPT_DIR}/config}"

export ROS_NETWORK_INTERFACE="${ROS_NETWORK_INTERFACE:-enp4s0}"
export RMW_IMPLEMENTATION="${RMW_IMPLEMENTATION:-rmw_cyclonedds_cpp}"
export CYCLONEDDS_URI="${CYCLONEDDS_URI:-${SCRIPT_DIR}/config/cyclone_config.xml}"
