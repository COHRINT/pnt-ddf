from copy import copy

from bpdb import set_trace

from pntddf.filters import Unscented_Information_Filter
from pntddf.information import invert


class Estimator:
    def __init__(self, env, agent):
        self.env = env
        self.agent = agent

        self.filt = Unscented_Information_Filter(self.env, agent)

        self.new_information_for_radio = False

        self.message_queue = []
        self.message_queue_centralized = []

    def new_message(self, message):
        self.message_queue.append(message)
        self.message_queue = sorted(
            self.message_queue, key=lambda msg: msg.transmitter.name
        )
        self.message_queue_centralized.append(copy(message))
        self.message_queue_centralized = sorted(
            self.message_queue_centralized, key=lambda msg: msg.transmitter.name
        )

    def run_filter(self):
        self.set_time()
        self.prediction()

        if len(self.message_queue) >= self.env.NUM_AGENTS - 1:
            self.local_update()

            # self.iterate()

            self.fusion_update()

            self.step()

    def set_time(self):
        self.filt.set_time()

    def prediction(self):
        self.filt.predict_self()

        for message in self.message_queue:
            self.filt.predict_message(message)

    def local_update(self):
        self.filt.local_update()

    def iterate(self):
        self.filt.iterate()

    def fusion_update(self):
        self.filt.fusion_update()

    def step(self):
        self.message_queue = []
        self.filt.step()

    def get_state_estimate(self):
        x, P = invert(self.agent.estimator.filt.y_k_k, self.agent.estimator.filt.Y_k_k)

        return x

    def get_local_info(self):
        self.run_filter()
        if self.env.centralized:
            self.env.agent_centralized.estimator.run_filter(self.agent.name)

        return self.filt.get_local_info()

    def get_clock_estimate(self):
        return self.filt.get_clock_estimate()

    def get_state_log_df(self):
        return self.filt.state_log.get_state_log_df()

    def get_residuals_log_df(self):
        return self.filt.sensors.get_residuals_log_df()
