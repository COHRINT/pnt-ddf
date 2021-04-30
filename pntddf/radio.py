from copy import copy, deepcopy

import numpy as np
import simpy
from bpdb import set_trace
from scipy.constants import c
from scipy.spatial.distance import euclidean as distance

from pntddf.measurements import Pseudorange


class Radio:
    def __init__(self, env, agent):
        self.env = env
        self.agent = agent

        self.receive_log = {agent: [] for agent in self.env.AGENT_NAMES}

        self.cycle_number_previous = -1

        self.message_index = 1

        # Amount of time to wait before transmitting
        if "wait" in self.agent.config.keys():
            self.wait = self.agent.config.getfloat("wait")
        else:
            self.wait = 0.0

        if not self.env.ros:
            self.beacon_process = self.env.process(self.beacon())
        else:
            self.ros_init()

    def ros_init(self):
        global rospy
        import rospy

        global Transmission_ROS
        from pntddf_ros.msg import Transmission as Transmission_ROS

        # set up publisher
        topic_name = "transmission_{}".format(self.agent.name)
        self.pub = rospy.Publisher(topic_name, Transmission_ROS, queue_size=10)

        # register subscriber
        for agent in self.env.agents:
            if agent.name == self.agent.name:
                continue
            topic_name = "/agent_{}/transmission_{}".format(agent.name, agent.name)
            rospy.Subscriber(topic_name, Transmission_ROS, self.receive_ros)

        self.beacon_ros()

    def beacon_ros(self):
        while not rospy.is_shutdown():
            wait_time, cycle_number = self.agent.clock.get_transmit_wait()
            rospy.sleep(wait_time)

            self.prepare_message()

            transmission = Transmission_ROS()
            transmission.transmitter = self.agent.name
            transmission.timestamp_transmit = self.agent.clock.time()
            transmission.measurements = [
                meas.to_ros_message() for meas in self.message.measurements
            ]
            self.pub.publish(transmission)

    def beacon(self):
        # Initial wait before transmitting
        finished_wait = False
        while not finished_wait:
            try:
                yield self.env.timeout(self.wait - self.env.now)
                finished_wait = True
            except simpy.Interrupt:
                pass

        while True:
            try:
                wait_time, cycle_number = self.agent.clock.get_transmit_wait()
                yield self.env.timeout(wait_time)
            except simpy.Interrupt:
                continue

            self.prepare_message()

            self.transmit_message()

            self.cycle_number_previous = cycle_number

    def prepare_message(self):
        self.message = Message()

        # Transmitter
        self.message.transmitter = self.agent

        # Transmit time
        timestamp_transmit = self.agent.clock.time()
        self.message.timestamp_transmit = timestamp_transmit

        # Measurements
        if hasattr(self.agent, "estimator"):
            self.message.measurements = (
                self.agent.estimator.get_event_triggering_measurements()
            )
        else:
            self.message.measurements = []

        self.message_index += 1

    def transmit_message(self):
        agent_recipients = [
            agent
            for agent in self.env.agents
            if not agent.name == self.message.transmitter.name
        ]

        for recipient in agent_recipients:
            message = copy(self.message)
            message.receiver = recipient

            Transmission(self.env, message)

    def receive(self, message):
        self.beacon_process.interrupt()

        timestamp_receive = self.agent.clock.time()
        message.timestamp_receive = timestamp_receive

        self.receive_log[message.transmitter.name].append(
            (self.agent.clock.magic_time(), message.timestamp_receive)
        )

        pseudorange = Pseudorange(
            self.env,
            message.transmitter,
            message.receiver,
            message.timestamp_transmit,
            message.timestamp_receive,
        )

        # Pass the measurements to estimator
        self.agent.estimator.new_measurement(pseudorange)

        for measurement in message.measurements:
            measurement = copy(measurement)
            measurement.time_receive = timestamp_receive
            self.agent.estimator.new_measurement(measurement)

    def receive_ros(self, transmission):
        transmission = copy(transmission)
        transmission.receiver = self.agent.name

        dist = self.env.dynamics.distance_between_agents_true(
            transmission.transmitter, transmission.receiver
        )
        propagation_time = dist / self.env.c

        rospy.sleep(propagation_time)
        timestamp_receive = self.agent.clock.time()
        transmission.timestamp_receive = timestamp_receive

        # create pseudorange measurement
        pseudorange = Pseudorange(
            self.env,
            self.env.agent_dict[transmission.transmitter],
            self.env.agent_dict[transmission.receiver],
            transmission.timestamp_transmit,
            transmission.timestamp_receive,
        )

        # Pass the measurements to estimator
        # self.agent.estimator.new_measurement(pseudorange)

        # for measurement in transmission.measurements:
        # measurement = copy(measurement)
        # measurement.time_receive = timestamp_receive
        # self.agent.estimator.new_measurement(measurement)

        # rospy.loginfo(
        # "{} received from {}: {:.2f} m".format(
        # self.agent.name, transmission.transmitter, pseudorange.z
        # )
        # )


class Transmission:
    def __init__(self, env, message):
        self.env = env
        self.transmitter = message.transmitter
        self.receiver = message.receiver
        self.message = message

        self.propagated = self.env.event()

        self.env.process(self.send())
        self.env.process(self.propagate())

    def send(self):
        yield self.propagated

        self.receiver.radio.receive(self.message)

    def propagate(self):
        dist = self.env.dynamics.distance_between_agents_true(
            self.transmitter.name, self.receiver.name
        )

        propagation_time = dist / self.env.c

        yield self.env.timeout(propagation_time)
        self.propagated.succeed()


class Message:
    def __init__(self):
        self.transmitter = None
        self.receiver = None

        self.measurements = None

    def __copy__(self):
        new = type(self)()

        new.__dict__.update(self.__dict__)
        new.measurements = [copy(meas) for meas in self.measurements]

        return new
