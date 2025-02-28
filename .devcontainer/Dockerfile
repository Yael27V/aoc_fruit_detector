# syntax = devthefuture/dockerfile-x

ARG BASE_IMAGE=lcas.lincoln.ac.uk/lcas/ros:jammy-humble-cuda11.8-opengl
############################################################################################################
FROM ${BASE_IMAGE} as base

USER root

ENV DEBIAN_FRONTEND=noninteractive
ENV ROS_DISTRO=humble
ARG user_id=1000
ENV user_id=${user_id}

ARG COLCON_WS=/home/ros/fruit_detector_ws
ENV COLCON_WS=${COLCON_WS}

# Install all required APT packages -----------------------------------------------------------------------
# INCLUDE .devcontainer/docker_include/apt_install.dockerfile

RUN apt-get update \
    && apt-get install -qq -y --no-install-recommends \
    git \
    ufw \
    wget \
    libcairo2-dev \
    pkg-config \
    python3-pip \
    python3-rosdep \
    python3-debian \
    python3-dev \
    python3-opencv \
    libsystemd-dev \
    libgl1-mesa-glx \
    unattended-upgrades \
    usb-creator-gtk \
    libcups2-dev \
    screen-resolution-extra \
    cuda-toolkit-11-8 \
    ninja-build && \
    rm -rf /var/lib/apt/lists/*

# Set Python3 as default (if there is no link before)
#RUN ln -s /usr/bin/python3 /usr/bin/python
RUN if [ ! -e /usr/bin/python ]; then ln -s /usr/bin/python3 /usr/bin/python; fi

# Include env variables -----------------------------------------------------------------------------------
# INCLUDE .devcontainer/docker_include/env_setup.dockerfile

# This Dockerfile sets up a ROS 2 development environment. It starts by setting the environment variable USER to 'ros'.
# The 'chpasswd' command sets the password for the 'ros' user to 'ros'. The 'adduser' command adds the 'ros' user to the 'sudo' group.
# A directory for the ROS 2 workspace is created at the path specified by the COLCON_WS environment variable.
# The working directory is set to /home/ros, and ownership and permissions for this directory are adjusted to ensure the 'ros' user has the necessary access.
ENV USER=ros
RUN echo 'ros:ros' | chpasswd
RUN adduser ${USER} sudo
RUN mkdir -p ${COLCON_WS}/src/
WORKDIR /home/ros
RUN chown -R ros:ros /home/ros
RUN chmod 755 /home/ros

# ENVs
ENV HOME=/home/ros
ENV PATH="/home/ros/.local/bin:${PATH}"
# Force Python stdout and stderr streams to be unbuffered.
ENV PYTHONUNBUFFERED=1
# Set the LD_PRELOAD environment variable
ENV LD_PRELOAD=/usr/lib/libdlfaker.so:/usr/lib/libvglfaker.so
# enable all capabilities for the container
ENV NVIDIA_DRIVER_CAPABILITIES=all

# Add the current user to the dialout, video, and tty groups.
# This grants the user permissions to access serial ports, video devices, and terminal devices respectively.
RUN usermod -a -G dialout $USER && \
    usermod -a -G video $USER && \
    usermod -a -G tty $USER

############################################################################################################
FROM base AS user_space

# -----------------------------------------------------------------------------------------------------------
# INCLUDE .devcontainer/docker_include/user_install.dockerfile

# Install external repos 
COPY .devcontainer/repos ${COLCON_WS}/src/repos

# Upgrade pip
RUN pip install --no-cache-dir --upgrade pip

# Install PyTorch with specified versions and CUDA
RUN pip install --no-cache-dir \
    torch==2.0.0 torchvision==0.15.1 \
    --index-url https://download.pytorch.org/whl/cu118

# Install additional Python packages
RUN pip install --no-cache-dir \
    numpy==1.24.2 \
    opencv-python==4.10.0.84 \
    cv-bridge \
    scikit-image==0.24.0 \
    matplotlib==3.7.0 \
    requests==2.28.2 \
    kmeans-pytorch==0.3

# RUN apt-get update && apt-get install -qq -y --no-install-recommends \
#     ros-humble-rmw-cyclonedds-cpp \
#     ros-humble-cyclonedds \
#     ros-humble-cv-bridge \
#     ros-humble-rviz2 \
#     && rm -rf /var/lib/apt/lists/*

# RUN pip install meson meson-python

# # Get the requirements file
# COPY requirements.txt ${COLCON_WS}/src/requirements.txt
# RUN pip install -r ${COLCON_WS}/src/requirements.txt && rm ${COLCON_WS}/src/requirements.txt

USER $USER

# Copy the script to checkout public git repos and make it executable
COPY .devcontainer/scripts/install_external_ros_packages.sh ${COLCON_WS}/src/install_external_ros_packages.sh
# Make the script executable and run it, then remove it
RUN /bin/bash -c '${COLCON_WS}/src/install_external_ros_packages.sh ${COLCON_WS}' && \
    sudo rm -f ${COLCON_WS}/src/install_external_ros_packages.sh && \
    sudo rm -f -r ${COLCON_WS}/src/repos

USER root

############################################################################################################
# get the source tree and analyse it for its package.xml only
FROM user_space as sourcefilter
COPY . /tmp/src
# remove everything that isn't package.xml
RUN find /tmp/src -type f \! -name "package.xml" -print | xargs rm -rf

############################################################################################################
# install all dependencies listed in the package.xml
FROM user_space as depbuilder
# copy the reduced source tree (only package.xml) from previous stage
COPY --from=sourcefilter /tmp/src /tmp/src
RUN rosdep update && apt-get update
RUN cd /tmp/src && rosdep install --from-paths . --ignore-src -r -y && cd && rm -rf /tmp/src

# Remove unnecessary files or temporary files created during the setup
RUN rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

############################################################################################################
FROM depbuilder as final

USER $USER

# include the file to configure the user's .bashrc ---------------------------------------------------------
# INCLUDE .devcontainer/docker_include/bashrc_config.dockerfile

RUN echo "source /opt/ros/${ROS_DISTRO}/setup.bash" >> ~/.bashrc && \
    echo "source /usr/share/colcon_cd/function/colcon_cd.sh" >> ~/.bashrc && \
    echo "export _colcon_cd_root=${COLCON_WS}" >> ~/.bashrc && \
    echo "source ${COLCON_WS}/install/setup.bash" >> ~/.bashrc

CMD ["/bin/bash"]
