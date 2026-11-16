# The ROS 2 container for tiago_mpc: system Python 3.10, a pinned uv, ROS 2 Humble and
# PAL's interface overlay. No Python environment is baked in; `uv sync` creates the
# checkout's .venv-container (docs/robot.md).
# BASE_IMAGE is the CUDA devel image of the target (docker.sh: --target desktop|jetson).
ARG BASE_IMAGE=nvidia/cuda:12.9.1-devel-ubuntu22.04
FROM ${BASE_IMAGE}

ARG LOCAL_UID=1000
ARG LOCAL_GID=1000
ARG LOCAL_USER=developer
ARG UV_VERSION=0.11.6
ARG ROS_DISTRO=humble

ENV DEBIAN_FRONTEND=noninteractive
ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8

# nvcc comes with the base image. GATO compiles its modules at run time with the
# host compiler and the Python headers; record.py calls git.
RUN apt-get update \
        && apt-get install -y ca-certificates curl gnupg build-essential cmake git \
            python3.10 python3.10-dev python3.10-venv \
        && curl -fsSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key \
            | gpg --dearmor --yes -o /usr/share/keyrings/ros-archive-keyring.gpg \
        && echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu jammy main" \
            > /etc/apt/sources.list.d/ros2.list \
        && apt-get update \
        && apt-get install -y \
            "ros-${ROS_DISTRO}-ros-base" \
            "ros-${ROS_DISTRO}-rclpy" \
            "ros-${ROS_DISTRO}-control-msgs" \
            "ros-${ROS_DISTRO}-trajectory-msgs" \
            "ros-${ROS_DISTRO}-sensor-msgs" \
            "ros-${ROS_DISTRO}-geometry-msgs" \
            "ros-${ROS_DISTRO}-rmw-cyclonedds-cpp" \
            "ros-${ROS_DISTRO}-ros2controlcli" \
            "ros-${ROS_DISTRO}-rosidl-default-generators" \
            "ros-${ROS_DISTRO}-lifecycle-msgs" \
            python3-colcon-common-extensions \
        && rm -rf /var/lib/apt/lists/*

RUN curl -LsSf "https://astral.sh/uv/${UV_VERSION}/install.sh" \
        | env UV_INSTALL_DIR=/usr/local/bin UV_NO_MODIFY_PATH=1 sh

# The robot and PAL simulator use different wire layouts from standard Humble.
# Build only their message package, natively on either amd64 or arm64.
# Humble's generators use PYTHON_EXECUTABLE as well as Python3_EXECUTABLE;
# keep both on system Python, which has the apt-installed generator modules.
# ros_tiago.sh and ros_simulation.sh source /opt/pal-interfaces/setup.bash.
COPY pal_interfaces /tmp/pal-interfaces
RUN /bin/bash -c 'set -e; cd /tmp/pal-interfaces; \
        sha256sum -c SHA256SUMS; \
        source /opt/ros/${ROS_DISTRO}/setup.bash; \
        colcon build --merge-install --install-base /opt/pal-interfaces \
            --allow-overriding controller_manager_msgs \
            --cmake-args -DBUILD_TESTING=OFF \
                -DPYTHON_EXECUTABLE=/usr/bin/python3 -DPython3_EXECUTABLE=/usr/bin/python3' \
        && rm -rf /tmp/pal-interfaces

# The user has the host's IDs, so files written to the mounted workspace stay yours.
RUN groupadd --gid "${LOCAL_GID}" "${LOCAL_USER}" \
        && useradd --uid "${LOCAL_UID}" --gid "${LOCAL_GID}" --create-home --shell /bin/bash "${LOCAL_USER}"

# Each checkout's environment is .venv-container (relative: the checkout's root), built
# from the system Python that ROS 2 Humble's rclpy is built for.
ENV UV_PROJECT_ENVIRONMENT=.venv-container
ENV UV_PYTHON_PREFERENCE=only-system
ENV HOME=/home/${LOCAL_USER}
ENV USER=${LOCAL_USER}
USER ${LOCAL_USER}

CMD ["/bin/bash"]
