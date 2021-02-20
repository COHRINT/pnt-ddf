from itertools import chain
from statistics import mode

import numpy as np
import pandas as pd
from bpdb import set_trace
from numpy.linalg import inv


class State_Log:
    def __init__(self, env, agent):
        self.env = env
        self.agent = agent

        self.define_logs()

    def define_logs(self):
        self.log_t = []
        self.log_t_estimate = []

        self.log_x = []
        self.log_P = []

        self._log_u = []

        self.log_x_true = []

        self.log_epsilon_x = []

    def log_state(self):
        t = self.agent.clock.magic_time()

        x = self.agent.estimator.filt.x.copy()
        P = self.agent.estimator.filt.P.copy()

        t_estimate = self.agent.estimator.filt.get_time_estimate()

        self.log_t.append(t)
        self.log_t_estimate.append(t_estimate)
        self.log_x.append(x)
        self.log_P.append(P)

    def get_true(self):
        # Clock states
        b_values = [
            self.env.agent_dict[agent].clock.b
            for agent in self.env.agent_clocks_to_be_estimated
        ]
        b_dot_values = [
            self.env.agent_dict[agent].clock.b_dot
            for agent in self.env.agent_clocks_to_be_estimated
        ]

        clock_states = np.array(list(chain(*zip(b_values, b_dot_values))))

        if self.env.ROVER_NAMES:
            rover_states = np.concatenate(
                [
                    self.env.dynamics.get_true_state(rover_name)
                    for rover_name in self.env.ROVER_NAMES
                ]
            )
        else:
            rover_states = np.empty(0)

        x_true = np.concatenate([clock_states, rover_states])

        return x_true

    def log_u(self):
        t_estimate = self.agent.estimator.filt.get_time_estimate()

        x = self.agent.estimator.filt.x.copy()

        u = self.env.dynamics.u(t_estimate, x)

        self._log_u.append(u)

    def log_true(self):
        x_true = self.get_true()

        self.log_x_true.append(x_true)

    def log_NEES_errors(self):
        # State
        x = self.agent.estimator.filt.x.copy()
        P = self.agent.estimator.filt.P.copy()

        x_true = self.get_true()
        e_x = x - x_true

        epsilon_x = e_x @ inv(P) @ e_x

        # Measurements
        # e_z = self._log_residuals[-1]

        # epsilon_z = e_z @ inv(self._log_P_yy[-1]) @ e_z

        self.log_epsilon_x.append(epsilon_x)
        # self.log_epsilon_z.append(epsilon_z)

    def get_state_log_df(self):
        data = np.hstack(
            [np.array(self.log_t)[np.newaxis].T]
            + [np.array(self.log_t_estimate)[np.newaxis].T]
            + [np.stack(self.log_x)]
            + [np.stack(self.log_x) - np.stack(self.log_x_true)]
            + [np.stack(self.log_x_true)]
            + [
                np.sqrt(np.dstack(self.log_P)[i, i, :])[np.newaxis].T
                for i in range(self.env.NUM_STATES)
            ]
            + [np.stack(self._log_u)]
            + [np.array(self.log_epsilon_x)[np.newaxis].T]
        )

        state_names = self.env.STATE_NAMES
        control_names = list(
            chain.from_iterable(
                [
                    ["u_{}_{}".format(v, rover_name) for v in self.env.dim_names]
                    for rover_name in self.env.ROVER_NAMES
                ]
            )
        )

        columns = (
            ["t", "t_estimate"]
            + state_names
            + ["{}_error".format(var) for var in state_names]
            + ["{}_true".format(var) for var in state_names]
            + ["{}_sigma".format(var) for var in state_names]
            + control_names
            + ["epsilon_x"]
        )

        df = pd.DataFrame(data=data, columns=columns)

        df["P"] = self.log_P

        return df

    def get_P(self):
        data = np.dstack(self.log_P)

        df = pd.DataFrame(data=data, columns=["P"])

        return df
