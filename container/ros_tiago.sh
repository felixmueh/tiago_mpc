#!/usr/bin/env bash
# Source this file to select the university TIAGo Pro 12 over Jetson Ethernet in the current shell.
if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    echo 'Use: source container/ros_tiago.sh' >&2
    exit 1
fi

# PAL's interface overlay chains /opt/ros/humble.
source /opt/pal-interfaces/setup.bash || return 1

_ros_config_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)" || return 1
export ROS_DOMAIN_ID=2
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
export CYCLONEDDS_URI="${_ros_config_dir}/cyclone_tiago_ethernet.xml"
# PAL's guidance: ROS_LOCALHOST_ONLY must not be set.
unset ROS_LOCALHOST_ONLY
unset _ros_config_dir
echo 'ROS profile: tiago (domain 2, enP4p1s0 -> 10.68.0.1)'
