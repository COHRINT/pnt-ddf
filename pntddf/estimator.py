from copy import copy

import pandas as pd
from bpdb import set_trace

from pntddf.filters import LSQ_Filter, Unscented_Kalman_Filter
from pntddf.information import invert


class Estimator:
    def __init__(self, env, agent):
        self.env = env
        self.agent = agent

        self.measurement_log = []

        # Main Filter
        self.filt = Unscented_Kalman_Filter(self.env, agent)

        self.agent_centralized = self.env.agent_centralized

        # Initial filter
        self.lsq_init_completed = False
        if self.env.lsq_init:
            self.lsq_filter = LSQ_Filter(self.env, agent)

    def new_measurement(self, measurement):
        measurement.processor = self.agent
        if measurement.local:
            measurement.time_process_local = self.agent.clock.time()
            measurement.x_true = self.filt.state_log.get_true()
            self.run_centralized_filter(copy(measurement))
        else:
            measurement.time_process_external = self.agent.clock.time()

        self.run_filter(measurement)

    def run_filter(self, measurement):
        self.set_time(measurement)

        if self.env.lsq_init and not self.lsq_init_completed:
            completed = self.lsq_filter.estimate(measurement)
            self.lsq_init_completed = completed
            self.step(measurement)

            if completed:
                self.filt.define_initial_state(x=self.lsq_filter.x, P=self.lsq_filter.P)

            return

        self.prediction(measurement)

        self.local_update(measurement)

        # self.fusion_update()

        self.step(measurement)
        self.log_measurement(measurement)

    def run_centralized_filter(self, measurement):
        self.agent_centralized.estimator.run_filter(measurement)

    def set_time(self, measurement):
        self.filt.set_time(measurement)

    def prediction(self, measurement):
        self.filt.predict_self(measurement)

        # for message in self.message_queue:
        # self.filt.predict_message(message)

    def local_update(self, measurement):
        self.filt.local_update(measurement)

    def fusion_update(self):
        self.filt.fusion_update()

    def log_measurement(self, measurement):
        self.measurement_log.append(measurement)

    def step(self, measurement):
        self.filt.step(measurement)

    def get_state_estimate(self):
        x = self.agent.estimator.filt.x.copy()

        return x

    # def get_local_info(self):
    #     self.run_filter()

    #     if self.env.centralized:
    #         self.env.agent_centralized.estimator.run_filter(self.agent.name)

    #     return self.filt.get_local_info()

    def get_event_triggering_measurements(self):
        if not self.lsq_init_completed:
            measurements = self.lsq_filter.get_local_measurements_for_retransmission()
        else:
            measurements = self.filt.get_event_triggering_measurements()
        return measurements

    def get_clock_estimate(self):
        return self.filt.get_clock_estimate()

    def get_state_log_df(self):
        return self.filt.state_log.get_state_log_df()

    def get_residuals_log_df(self):
        residuals_df = pd.DataFrame([meas.data for meas in self.measurement_log])
        return residuals_df
