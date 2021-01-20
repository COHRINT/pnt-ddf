from copy import copy

import numpy as np
import simpy
from bpdb import set_trace
from scipy.spatial.distance import euclidean as distance


class Radio:
    def __init__(self, env, agent):
        self.env = env
        self.agent = agent

        self.message_index = 0

        self.receive_log = {agent: [] for agent in self.env.AGENT_NAMES}

        self.cycle_number_previous = -1

        # Amount of time to wait before transmitting
        if "wait" in self.agent.config.keys():
            set_trace()
            self.wait = self.agent.config.getfloat("wait")
        else:
            self.wait = 0.0

        self.beacon_process = self.env.process(self.beacon())

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
        self.message.transmitter = self.agent
        self.message.time_transmit = self.agent.clock.time()

        self.message.index = self.message_index

        self.message.local_info = self.agent.estimator.get_local_info()

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

        message.time_receive = self.agent.clock.time()

        self.receive_log[message.transmitter.name].append(
            (self.agent.clock.magic_time(), message.time_receive)
        )

        # Pass the message to clock_estimator
        self.agent.estimator.new_message(message)


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
        self.time_transmit = None

        self.receiver = None
        self.time_receive = None

        self.local_info = None

        self.propagated = False

        self.index = 0

    def __repr__(self):
        return str(self.__dict__)

    def __copy__(self):
        new = type(self)()

        new.__dict__.update(self.__dict__)
        new.local_info = copy(self.local_info)

        return new
