from copy import copy

from bpdb import set_trace

from pntddf.filters import LSQ_Filter, Unscented_Kalman_Filter
from pntddf.information import invert


class Estimator:
    def __init__(self, env, agent):
        self.env = env
        self.agent = agent

        self.measurement_queue = []
        self.measurement_queue_centralized = []

        # Main Filter
        self.filt = Unscented_Kalman_Filter(self.env, agent)

        # Initial filter
        self.lsq_init_completed = True
        if self.env.lsq_init:
            self.lsq_filter = LSQ_Filter(self.env, agent)

    def new_measurement(self, measurement):
        self.measurement_queue.append(measurement)
        self.measurement_queue_centralized.append(copy(measurement))

    def run_filter(self):
        self.set_time()

        if self.env.lsq_init and not self.lsq_init_completed:
            self.lsq_filter.estimate(self.message_queue)

        self.prediction()

        self.local_update()

        # self.fusion_update()

        self.step()

    def set_time(self):
        self.filt.set_time()

    def prediction(self):
        self.filt.predict_self()

        # for message in self.message_queue:
        # self.filt.predict_message(message)

    def local_update(self):
        self.filt.local_update()

    def fusion_update(self):
        self.filt.fusion_update()

    def step(self):
        self.message_queue = []
        self.filt.step()

    def get_state_estimate(self):
        x = self.agent.estimator.filt.x.copy()

        return x

    # def get_local_info(self):
    #     self.run_filter()

    #     if self.env.centralized:
    #         self.env.agent_centralized.estimator.run_filter(self.agent.name)

    #     return self.filt.get_local_info()

    def get_event_triggering_measurements(self):
        return self.filt.get_event_triggering_measurements()

    def get_clock_estimate(self):
        return self.filt.get_clock_estimate()

    def get_state_log_df(self):
        return self.filt.state_log.get_state_log_df()

    def get_residuals_log_df(self):
        return self.filt.sensors.get_residuals_log_df()
