FROM ros:melodic

RUN apt-get update && apt-get install -y \
	apt-utils \
	bpython3 \
	mlocate \
	nano \
	python3-pip \
	ros-melodic-geodesy \
	ros-melodic-global-planner \
	ros-melodic-jackal-* \
	ros-melodic-mav* \
	ros-melodic-message-to-tf \
	ros-melodic-pointgrey-camera-driver \
	ros-melodic-roslint \
	ros-melodic-rqt-graph \
	ros-melodic-tf \
	tree \
	udev \
	wget \
	&& rm -rf /var/lib/apt/lists/*

# ROS Workspace
ENV ROS_WS /opt/ros_ws

WORKDIR $ROS_WS

# Use bash because of setup.bash files
SHELL ["/bin/bash", "-c"]

# Multiple Jackal
WORKDIR $ROS_WS/src
RUN source /opt/ros/melodic/setup.bash; catkin_init_workspace
RUN git clone https://github.com/NicksSimulationsROS/multi_jackal

# Gazebo models
RUN mkdir -p /root/.gazebo/models/ground_plane
COPY gazebo_models/ground_plane/ /root/.gazebo/models/ground_plane
RUN mkdir -p /root/.gazebo/models/sun
COPY gazebo_models/sun/ /root/.gazebo/models/sun

# Anything after this point will be rerun
ARG CACHEBUST=1

# PNT-DDF Setup
ENV PNTDDF_WS /opt/pnt_ddf_ws
RUN mkdir -p $PNTDDF_WS
WORKDIR $PNTDDF_WS

COPY pntddf/ $PNTDDF_WS/pntddf
COPY pntddf_ros/ $PNTDDF_WS/pntddf_ros
COPY config/ $PNTDDF_WS/config
COPY requirements.txt $PNTDDF_WS
COPY setup.py $PNTDDF_WS

RUN pip3 install -e . -r requirements.txt

RUN ln -s $PNTDDF_WS/pntddf_ros/ $ROS_WS/src

# Compile everything
WORKDIR $ROS_WS

ENV ROS_PARALLEL_JOBS -j2
RUN source /opt/ros/melodic/setup.bash; catkin_make

RUN updatedb

# source ROS
RUN echo "source /opt/ros/melodic/setup.bash" >> ~/.bashrc
RUN echo "source /ros_entrypoint.sh" >> ~/.bashrc
RUN echo "source /opt/ros_ws/devel/setup.bash" >> ~/.bashrc

WORKDIR $ROS_WS

CMD ["/bin/bash"]
