from functools import reduce

import numpy as np
import pandas as pd
from bpdb import set_trace
from numpy.linalg import inv, norm
from sympy import Matrix, symbols
from sympy.utilities.lambdify import lambdify


def unit(x):
    return x / norm(x)


class Ranging_Module:
    def __init__(self, env, agent):
        self.env = env
        self.agent = agent

        self.define_measurement()
        self.define_measurement_names()
        self.define_measurement_log()

    def define_measurement(self):
        self.evaluate_h = {}
        # measurement noise matrix
        self.R = {}

        # Delays
        tau_distance, tau_process = symbols(
            "tau_distance tau_process", real=True, positive=True
        )
        tau = tau_distance + tau_process

        # rho_R/T
        for T in self.env.AGENT_NAMES:
            if T == self.agent.name:
                continue

            # Position of receiver at current time
            x_R_rT = self.env.dynamics.get_sym_position(self.agent.name)
            # Velocity of receiver at current time
            x_dot_R_rT = self.env.dynamics.get_sym_velocity(self.agent.name)
            # Position of receiver when message was received
            x_R_tR = x_R_rT - x_dot_R_rT * tau_process

            # Position of transmitter at current time
            x_T_rT = self.env.dynamics.get_sym_position(T)
            # Velocity of transmitter at current time
            x_dot_T_rT = self.env.dynamics.get_sym_velocity(T)
            # Position of transmitter when message was transmitted
            x_T_tT = x_T_rT - x_dot_T_rT * tau

            d = Matrix(x_R_tR - x_T_tT).norm()

            # Delay of receiver at current time
            b_R_rT = self.env.dynamics.get_sym("b", self.agent.name)
            # Delay rate of receiver at current time
            b_dot_R_rT = self.env.dynamics.get_sym("b_dot", self.agent.name)
            # Delay of receiver when message was received
            b_R_tR = b_R_rT - b_dot_R_rT * tau_process

            # Delay of transmitter at current time
            b_T_rT = self.env.dynamics.get_sym("b", T)
            # Delay rate of transmitter at current time
            b_dot_T_rT = self.env.dynamics.get_sym("b_dot", T)
            # Delay of transmitter when message was transmitted
            b_T_tT = b_T_rT - b_dot_T_rT * tau

            rho = d + b_R_tR - b_T_tT

            h = Matrix([rho])
            x_vec_augmented = [tau_process, tau_distance] + self.env.dynamics.x_vec
            self.evaluate_h[self.agent.name + T] = lambdify(
                x_vec_augmented, np.squeeze(h), "numpy"
            )

            # Measurement noise matrix
            R_sigma_clock = self.env.agent_configs[self.agent.name].getfloat(
                "sigma_clock_reading"
            )
            T_sigma_clock = self.env.agent_configs[T].getfloat("sigma_clock_reading")

            # TODO: MOVE ALL THE R MATH UP TO HERE AT SOME POINT
            self.R[self.agent.name + T] = self.env.c ** 2 * (
                R_sigma_clock ** 2 + T_sigma_clock ** 2
            )

    def define_measurement_names(self):
        self.measurement_names = [
            "rho_" + self.agent.name + name
            for name in self.env.AGENT_NAMES
            if name != self.agent.name
        ]
        self.measurement_names_latex = [
            "$\\rho_{" + self.agent.name + name + "}$"
            for name in self.env.AGENT_NAMES
            if name != self.agent.name
        ]

    def define_measurement_log(self):
        self._log_measurement = []
        self._log_residuals_pre_fit = []
        self._log_residuals_post_fit = []
        self._log_residuals_fused = []
        self._log_R_sigma = []
        self._log_P_yy_sigma = []

    def true_measurement(self, message_queue):
        rho = self.env.c * np.array(
            [msg.time_receive - msg.time_transmit for msg in message_queue]
        )

        return rho

    def predict_measurement(self, x_hat, message_queue):
        rho = []
        for msg in message_queue:
            distance_estimate = self.env.dynamics.distance_between_agents(
                msg.receiver.name, msg.transmitter.name, x_hat
            )
            tau_distance = distance_estimate / self.env.c
            tau_process = self.agent.estimator.filt.t_kp1 - msg.time_receive

            rho.append(
                self.evaluate_h[msg.receiver.name + msg.transmitter.name](
                    tau_process, tau_distance, *x_hat
                )
            )

        rho = np.array(rho)

        return rho

    def generate_R(self, x_hat, message_queue):
        R_values = []

        if self.env.now > 10:
            set_trace()

        for msg in message_queue:
            R_read = self.R[msg.receiver.name + msg.transmitter.name]

            distance_estimate = self.env.dynamics.distance_between_agents(
                msg.receiver.name, msg.transmitter.name, x_hat
            )
            tau_distance = distance_estimate / self.env.c
            tau_process = self.agent.estimator.filt.t_kp1 - msg.time_receive

            # Transmitter clock changes during travel and while waiting
            R_process_transmitter = (
                1
                / 3
                * (tau_distance + tau_process) ** 3
                * msg.transmitter.clock.sigma_clock_process ** 2
                * self.env.c ** 2
            )

            # Receiver clock only changes while waiting
            R_process_receiver = (
                1
                / 3
                * tau_process ** 3
                * msg.receiver.clock.sigma_clock_process ** 2
                * self.env.c ** 2
            )

            R = R_read  # + R_process_transmitter + R_process_receiver
            R_values.append(R)

        R = np.diag(R_values)

        return R

    def log_measurement(self, y, message_queue):
        measurements = self.sort_measurements(y, message_queue)

        self._log_measurement.append(measurements)

    def log_residuals(self, r, message_queue, pre, post, fused):
        residuals = self.sort_measurements(r, message_queue)

        if pre:
            self._log_residuals_pre_fit.append(residuals)
        elif post:
            self._log_residuals_post_fit.append(residuals)
        elif fused:
            self._log_residuals_fused.append(residuals)

    def log_R(self, R, message_queue):
        sigma = self.sort_measurements(np.sqrt(np.diag(R)), message_queue)

        self._log_R_sigma.append(sigma)

    def log_P_yy(self, P_yy, message_queue):
        sigma = self.sort_measurements(np.sqrt(np.diag(P_yy)), message_queue)

        self._log_P_yy_sigma.append(sigma)

    def get_df(self):
        dfs = []

        meas_df = pd.DataFrame(
            data=np.vstack(self._log_measurement),
            columns=["t"] + self.measurement_names,
        )
        dfs.append(meas_df)

        pre_df = pd.DataFrame(
            data=np.vstack(self._log_residuals_pre_fit),
            columns=["t"]
            + [meas + "_residual_pre_fit" for meas in self.measurement_names],
        )
        dfs.append(pre_df)

        post_df = pd.DataFrame(
            data=np.vstack(self._log_residuals_post_fit),
            columns=["t"]
            + [meas + "_residual_post_fit" for meas in self.measurement_names],
        )
        dfs.append(post_df)

        if self._log_residuals_fused:
            fused_df = pd.DataFrame(
                data=np.vstack(self._log_residuals_fused),
                columns=["t"]
                + [meas + "_residual_fused" for meas in self.measurement_names],
            )
            dfs.append(fused_df)

        R_sigma_df = pd.DataFrame(
            data=np.vstack(self._log_R_sigma),
            columns=["t"] + [meas + "_R_sigma" for meas in self.measurement_names],
        )
        dfs.append(R_sigma_df)

        P_yy_sigma_df = pd.DataFrame(
            data=np.vstack(self._log_P_yy_sigma),
            columns=["t"] + [meas + "_P_yy_sigma" for meas in self.measurement_names],
        )
        dfs.append(P_yy_sigma_df)

        dfs = [df.dropna() for df in dfs]

        df = reduce(lambda left, right: pd.merge(left, right, on="t"), dfs)

        return df

    def sort_measurements(self, y, message_queue):
        n_rows = max(
            [
                sum(
                    [
                        "rho_" + msg.receiver.name + msg.transmitter.name == meas_name
                        for msg in message_queue
                    ]
                )
                for meas_name in self.measurement_names
            ]
        )
        measurements = np.ones([n_rows, 1 + len(self.measurement_names)]) * np.nan

        for index, meas_name in enumerate(self.measurement_names):
            meas_indices = np.argwhere(
                [
                    "rho_" + msg.receiver.name + msg.transmitter.name == meas_name
                    for msg in message_queue
                ]
            ).ravel()
            meas = y[meas_indices]
            measurements[: meas.size, index + 1] = meas

        t = np.ones(n_rows) * self.agent.clock.magic_time()
        measurements[:, 0] = t

        return measurements

    def generate_DOP(self):
        x_hat = self.agent.estimator.get_state_estimate()
        los_vectors = [
            unit(
                self.env.dynamics.los_between_agents(
                    msg.receiver.name, msg.transmitter.name, x_hat
                )
            )
            for msg in self.agent.estimator.message_queue
        ]
        G = np.hstack([np.vstack(los_vectors), np.ones([len(los_vectors), 1])])

        # CHANGE ME TO ACCOUNT FOR DIFFERENT SIGMA
        sigma_squared = list(self.R.items())[0][1]

        H = sigma_squared * inv(G.T @ G)

        return H
