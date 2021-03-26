from copy import copy

import pandas as pd
from bpdb import set_trace

from pntddf.filters import LSQ_Filter, Unscented_Kalman_Filter
from pntddf.information import invert
from pntddf.measurements import Measurement


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

    def new_info(self, info):
        self.set_time_from_info(info)
        self.prediction()
        self.filt.predict_info(info)
        self.fusion_update(info)
        self.step()

    def run_filter(self, measurement):
        self.set_time_from_measurement(measurement)

        if self.env.lsq_init and not self.lsq_init_completed:
            completed = self.lsq_filter.estimate(measurement)
            self.lsq_init_completed = completed
            self.step(measurement)

            if completed:
                self.filt.define_initial_state(x=self.lsq_filter.x, P=self.lsq_filter.P)

            return

        self.prediction()

        self.local_update(measurement)

        self.step(measurement)
        self.log_measurement(measurement)

    def run_centralized_filter(self, measurement):
        self.agent_centralized.estimator.run_filter(measurement)

    def set_time_from_measurement(self, measurement):
        self.filt.set_time_from_measurement(measurement)

    def set_time_from_info(self, info):
        self.filt.set_time_from_info(info)

    def prediction(self):
        self.filt.predict_self()

    def local_update(self, measurement):
        self.filt.local_update(measurement)

    def fusion_update(self, info):
        self.filt.fusion_update(info)

    def log_measurement(self, measurement):
        self.measurement_log.append(measurement)

    def step(self, measurement=None):
        if measurement is not None:
            self.filt.log(measurement)
        self.filt.step()

    def get_state_estimate(self):
        x = self.agent.estimator.filt.x.copy()

        return x

    def get_local_info(self):
        if self.lsq_init_completed:
            fake_meas = Measurement()
            fake_meas.local = True
            fake_meas.time_process_local = self.agent.clock.time()
            self.set_time_from_measurement(fake_meas)
            self.prediction()
            return self.filt.get_local_info()
        else:
            return None

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
