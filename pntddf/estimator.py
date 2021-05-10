from copy import copy

import pandas as pd
from bpdb import set_trace

from pntddf.filters import LSQ_Filter, Unscented_Kalman_Filter
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
        if measurement.local and self.env.centralized:
            measurement.time_process_local = self.agent.clock.time()
            self.run_centralized_filter(copy(measurement))

        self.run_filter(measurement)

    def run_filter(self, measurement):
        if self.env.ros:
            import rospy

            rospy.loginfo(
                "{} has measurement {}".format(self.agent.name, measurement.name)
            )
        self.set_time_from_measurement(measurement)

        if self.env.lsq_init and not self.lsq_init_completed:
            completed = self.lsq_filter.estimate(measurement)
            self.lsq_init_completed = completed
            self.log(measurement)
            self.step()

            if completed:
                self.filt.define_initial_state(x=self.lsq_filter.x, P=self.lsq_filter.P)

            return

        self.prediction()

        self.local_update(measurement)

        self.log(measurement)
        self.step()
        self.log_measurement(measurement)

    def run_centralized_filter(self, measurement):
        self.agent_centralized.estimator.run_filter(measurement)

    def set_time_from_measurement(self, measurement):
        self.filt.set_time_from_measurement(measurement)

    def prediction(self):
        self.filt.predict_self()

    def local_update(self, measurement):
        self.filt.local_update(measurement)

    def log_measurement(self, measurement):
        self.measurement_log.append(measurement)

    def log(self, measurement):
        self.filt.log(measurement)

    def step(self):
        self.filt.step()

    def get_state_estimate(self):
        x = self.filt.x.copy()
        P = self.filt.P.copy()

        return x, P

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
        get_col = lambda name: [getattr(meas, name) for meas in self.measurement_log]
        columns = [
            "t",
            "z",
            "r",
            "sigma",
            "P_yy_sigma",
            "name",
            "latex_name",
            "local",
            "implicit",
            "explicit",
        ]
        data_dict = {name: get_col(name) for name in columns}
        residuals_df = pd.DataFrame.from_dict(data_dict)
        return residuals_df
