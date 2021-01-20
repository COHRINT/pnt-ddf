from copy import copy
from functools import reduce

import pandas as pd
from bpdb import set_trace

from pntddf.clock import Clock
from pntddf.estimator import Estimator
from pntddf.filters import Unscented_Information_Filter
from pntddf.information import invert
from pntddf.sensors import Sensors


class Agent_Centralized:
    def __init__(self, env, name):
        self.env = env
        self.name = name

        self.config = self.env.config_centralized

        self.agent_reference = "A"

    def init(self):
        self.clock = self.env.agent_dict[self.agent_reference].clock

        self.sensors_centralized = Sensors_Centralized(self.env, self)

        self.sensors = self.sensors_centralized

        # Estimator
        self.estimator = Estimator_Centralized(self.env, self)


class Estimator_Centralized:
    def __init__(self, env, agent):
        self.env = env
        self.agent = agent

        self.filt = Unscented_Information_Filter(self.env, agent)

        self.message_queue = []

    def run_filter(self, agent_name):
        self.set_time()
        self.prediction()

        self.message_queue = self.env.agent_dict[
            agent_name
        ].estimator.message_queue_centralized
        self.filt.sensors = self.agent.sensors_centralized.agent_sensors[agent_name]

        if len(self.message_queue) > 4:
            self.local_update()

            self.step()

            self.env.agent_dict[agent_name].estimator.message_queue_centralized = []

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

    def step(self):
        self.message_queue = []
        self.filt.step()

    def get_state_estimate(self):
        x, P = invert(self.agent.estimator.filt.y_k_k, self.agent.estimator.filt.Y_k_k)

        return x

    def get_local_info(self):
        self.run_filter()

        return self.filt.get_local_info()

    def get_clock_estimate(self):
        return self.filt.get_clock_estimate()

    def get_state_log_df(self):
        return self.filt.state_log.get_state_log_df()

    def get_residuals_log_df(self):
        return self.agent.sensors_centralized.get_residuals_log_df()


class Sensors_Centralized:
    def __init__(self, env, agent):
        self.env = env
        self.agent = agent

        self.define_sensors()

    def define_sensors(self):
        self.agent_sensors = {}
        self.measurement_names = []
        self.measurement_names_latex = []

        for agent in self.env.agents:
            sensors = Sensors(self.env, agent, centralized=True)
            self.agent_sensors[agent.name] = sensors
            self.measurement_names.extend(sensors.measurement_names)
            self.measurement_names_latex.extend(sensors.measurement_names_latex)

    def get_residuals_log_df(self):
        dfs = []

        for agent in self.env.agents:
            dfs.append(self.agent_sensors[agent.name].get_residuals_log_df())

        df = reduce(lambda left, right: pd.merge(left, right, on="t", how="outer"), dfs)

        df.sort_values(by="t", inplace=True)

        return df
