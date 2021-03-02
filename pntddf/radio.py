from copy import copy, deepcopy

import numpy as np
import simpy
from bpdb import set_trace
from scipy.constants import c
from scipy.spatial.distance import euclidean as distance

from measurements import Pseudorange


class Radio:
    def __init__(self, env, agent):
        self.env = env
        self.agent = agent

        self.receive_log = {agent: [] for agent in self.env.AGENT_NAMES}

        self.cycle_number_previous = -1

        # Amount of time to wait before transmitting
        if "wait" in self.agent.config.keys():
            self.wait = self.agent.config.getfloat("wait")
        else:
            self.wait = 0.0

        if not self.env.ros:
            self.beacon_process = self.env.process(self.beacon())
        else:
            self.beacon_ros()

    def beacon_ros(self):
        # TODO: Use get_transmit_wait and then call rospy.sleep()

        while not rospy.is_shutdown():
            try:
                wait_time, cycle_number = self.agent.clock.get_transmit_wait()
                yield self.env.timeout(wait_time)
            except simpy.Interrupt:
                continue

            self.prepare_message()

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
        pseudorange = Pseudorange(self.env)
        pseudorange.local = True
        self.message.pseudorange = pseudorange

        # Transmitter
        self.message.transmitter = self.agent
        pseudorange.transmitter = self.agent

        # Transmit time
        timestamp_transmit = self.agent.clock.time()
        self.message.timestamp_transmit = timestamp_transmit
        pseudorange.timestamp_transmit = timestamp_transmit

        self.agent.estimator.run_filter()

        # Measurements
        self.message.measurements = (
            self.agent.estimator.get_event_triggering_measurements()
        )

        # Local Info
        # self.message.local_info = self.agent.estimator.get_local_info()

    def transmit_message(self):
        agent_recipients = [
            agent
            for agent in self.env.agents
            if not agent.name == self.message.transmitter.name
        ]

        for recipient in agent_recipients:
            message = copy(self.message)
            message.receiver = recipient
            message.pseudorange.receiver = recipient

            Transmission(self.env, message)

    def receive(self, message):
        self.beacon_process.interrupt()

        timestamp_receive = self.agent.clock.time()
        message.timestamp_receive = timestamp_receive
        message.pseudorange.timestamp_receive = timestamp_receive

        self.receive_log[message.transmitter.name].append(
            (self.agent.clock.magic_time(), message.timestamp_receive)
        )

        # Pass the measurements to estimator
        self.agent.estimator.new_measurement(message.pseudorange)
        for measurement in message.measurements:
            measurement = copy(measurement)
            measurement.time_receive = timestamp_receive
            self.agent.estimator.new_measurement(measurement)


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

        self.local_info = None
        self.messages = None

        self.pseudorange = None

    def __copy__(self):
        new = type(self)()

        new.__dict__.update(self.__dict__)
        new.messages = deepcopy(self.messages)
        new.pseudorange = copy(self.pseudorange)

        new.local_info = copy(self.local_info)

        return new
