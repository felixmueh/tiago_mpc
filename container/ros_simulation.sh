#!/usr/bin/env bash
# Source this file to select the same-host PAL simulator in the current shell.
if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    echo 'Use: source container/ros_simulation.sh' >&2
    exit 1
fi

# PAL's interface overlay chains /opt/ros/humble.
source /opt/pal-interfaces/setup.bash || return 1

_ros_config_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)" || return 1
export ROS_DOMAIN_ID=1
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
export CYCLONEDDS_URI="${_ros_config_dir}/cyclone_pal_loopback.xml"
export ROS_LOCALHOST_ONLY=0
unset _ros_config_dir
echo 'ROS profile: simulation (domain 1, loopback only)'
