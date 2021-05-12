import re
from copy import copy

import numpy as np
from bpdb import set_trace
from numpy import sqrt
from numpy.linalg import cholesky, inv, norm
from scipy.linalg import block_diag
from scipy.optimize import least_squares
from scipy.spatial.distance import mahalanobis
from scipy.special import erf
from scipy.stats import multivariate_normal

from pntddf.covariance_intersection import covariance_intersection
from pntddf.information import Information, invert
from pntddf.state_log import State_Log

np.set_printoptions(suppress=True)


def unit(x):
    return x / np.linalg.norm(x)


class LSQ_Filter:
    def __init__(self, env, agent):
        self.env = env
        self.agent = agent

        self.measurements = []
        self.local_measurements_for_retransmission = []

        self.completed = False

        self.broadcast_time = self.env.NUM_AGENTS * self.env.TRANSMISSION_WINDOW * 2

    def estimate(self, measurement):
        self.add_measurement(measurement)
        self.add_local_measurement(measurement)

        measurements = self.measurements

        if self.completed and self.agent.clock.time() > self.broadcast_time:
            return True
        elif self.completed:
            return False

        duplex_pairs = set(self.env.PAIRS_DUPLEX)
        measurement_pairs = set(
            [
                meas.receiver.name + meas.transmitter.name
                for meas in measurements
                if "rho" in meas.name
            ]
        )

        all_measurement_pairs = len(duplex_pairs - measurement_pairs) == 0

        if all_measurement_pairs and self.agent.clock.time() > self.broadcast_time:
            x0 = self.env.x0.copy()
            lower_bounds = np.array(
                [
                    -1e-6 if re.match(r"[xyz]_dot", str(state)) else -np.inf
                    for state in self.env.dynamics.x_vec
                ]
            )
            upper_bounds = np.array(
                [
                    1e-6 if re.match(r"[xyz]_dot", str(state)) else np.inf
                    for state in self.env.dynamics.x_vec
                ]
            )
            bounds = (lower_bounds, upper_bounds)
            lsq_result = least_squares(self.lsq_func, x0, bounds=bounds)

            if lsq_result.success:
                self.completed = True
                x = lsq_result.x
                H = lsq_result.jac
                R = np.diag([meas.sigma_adjusted for meas in measurements])
                P = inv(H.T @ inv(R) @ H)
                time_delta = self.env.TRANSMISSION_WINDOW * self.env.NUM_AGENTS
                Q = self.agent.estimator.filt.generate_Q(time_delta, np.array([0]))
                self.x = x
                self.P = P + Q

    def lsq_func(self, x):
        measurements = self.measurements

        residuals = []

        for measurement in measurements:
            tau = self.get_tau(measurement, x)

            x_back_prediction = self.env.dynamics.step_x(x, tau, 0)

            r = measurement.z - measurement.predict(x_back_prediction)

            if "rho" in measurement.name:
                sigma_process_T = measurement.transmitter.clock.sigma_clock_process
                sigma_process_R = measurement.receiver.clock.sigma_clock_process

                sigma_s_T = abs(
                    1 / 3 * sigma_process_T ** 2 * tau ** 3 * self.env.c ** 2
                )
                sigma_s_R = abs(
                    1 / 3 * sigma_process_R ** 2 * tau ** 3 * self.env.c ** 2
                )

                sigma = np.sqrt(measurement.sigma ** 2 + sigma_s_T + sigma_s_R)
            else:
                sigma = measurement.sigma

            measurement.sigma_adjusted = sigma
            residuals.append(r / sigma)

        return np.array(residuals)

    def get_tau(self, measurement, x):
        t = self.agent.clock.time()

        if measurement.local:
            tau = measurement.timestamp_receive - t
        else:
            R = measurement.receiver.name
            Delta_R, _ = self.agent.estimator.filt.get_clock_estimate(R, x)
            P = self.agent.name
            Delta_P, _ = self.agent.estimator.filt.get_clock_estimate(P, x)
            tau = measurement.timestamp_receive - Delta_R + Delta_P - t

        return tau

    def add_measurement(self, measurement):
        self.measurements.append(copy(measurement))

    def add_local_measurement(self, measurement):
        if measurement.local:
            measurement.local = False
            measurement.explicit = True
            self.local_measurements_for_retransmission.append(measurement)

    def get_local_measurements_for_retransmission(self):
        local_measurements = copy(self.local_measurements_for_retransmission)
        self.local_measurements_for_retransmission = []
        return local_measurements


class Unscented_Kalman_Filter:
    def __init__(self, env, agent):
        self.env = env
        self.agent = agent

        self.t_k = 0
        self.t_kp1 = 0

        # tracking how far we've added process noise
        self.t_Q = 0

        self.clock_estimate_dict = self.env.dynamics.get_clock_estimate_function()

        self.state_log = State_Log(env, agent)

        self.define_initial_state()
        self.define_event_triggering_measurements()
        self.define_constants()

    def define_initial_state(self, x=np.array([]), P=np.array([])):
        if x.size == 0:
            self.x = self.env.x0.copy()
        else:
            self.x = x

        if P.size == 0:
            self.P = self.env.P0.copy()
        else:
            self.P = P

    def define_event_triggering_measurements(self):
        self.event_triggering_messages = []

    def define_constants(self):
        self.Q_sigma_clock = self.env.filter_config.getfloat("Q_sigma_clock")
        self.Q_sigma_target = self.env.filter_config.getfloat("Q_sigma_target")

        alpha = self.env.filter_config.getfloat("alpha")
        beta = self.env.filter_config.getfloat("beta")

        N = self.env.NUM_STATES
        kappa = 6 - N
        lamb = alpha ** 2 * (N + kappa) - N

        self.w_m = [lamb / (N + lamb)] + [1 / (2 * (N + lamb))] * 2 * N
        self.w_c = [lamb / (N + lamb) + (1 - alpha ** 2 + beta)] + [
            1 / (2 * (N + lamb))
        ] * 2 * N

        self.N = N
        self.lamb = lamb

    def predict_self(self):
        x_hat = self.x.copy()
        P = self.P.copy()

        tau = self.get_tau()

        t_estimate = self.get_time_estimate()

        x_prediction = self.env.dynamics.step_x(x_hat, tau, t_estimate)
        P_prediction = self.env.dynamics.step_P(P, tau)

        tau_Q = self.get_tau_Q()
        u = self.env.dynamics.u(t_estimate)
        Q = self.generate_Q(time_delta=tau_Q, u=u)

        P_prediction += Q

        self.x = x_prediction
        self.P = P_prediction

    def predict_info(self, info):
        pass

    def generate_Q(self, time_delta, u):
        Q_clock = block_diag(
            *[
                np.array(
                    [
                        [1 / 3 * time_delta ** 3, 1 / 2 * time_delta ** 2],
                        [1 / 2 * time_delta ** 2, 1 / 1 * time_delta ** 1],
                    ]
                )
                * self.Q_sigma_clock ** 2
                * self.env.c ** 2
                for agent_name in self.env.agent_clocks_to_be_estimated
            ]
        )

        Q_target_scale = self.Q_sigma_target ** 2
        Q_target = block_diag(
            *[
                np.block(
                    [
                        [
                            np.eye(self.env.n_dim) * 1 / 3 * time_delta ** 3,
                            np.eye(self.env.n_dim) * 1 / 2 * time_delta ** 2,
                        ],
                        [
                            np.eye(self.env.n_dim) * 1 / 2 * time_delta ** 2,
                            np.eye(self.env.n_dim) * 1 / 1 * time_delta ** 1,
                        ],
                    ]
                )
                * Q_target_scale
                for rover_name in self.env.ROVER_NAMES
            ]
        )

        if Q_clock.size > 0 and Q_target.size > 0:
            Q = block_diag(Q_clock, Q_target)
        elif Q_target.size > 0:
            Q = Q_target
        else:
            Q = Q_clock

        return Q

    def generate_sigma_points(self, x, P):
        sigma_points = [
            x,
            *np.split(
                np.ravel(x + cholesky((self.N + self.lamb) * P).T),
                self.N,
            ),
            *np.split(
                np.ravel(x - cholesky((self.N + self.lamb) * P).T),
                self.N,
            ),
        ]

        return sigma_points

    def local_update(self, measurement):
        x_prediction = self.x.copy()
        P_prediction = self.P.copy()

        chi = self.generate_sigma_points(x_prediction, P_prediction)

        upsilon = [measurement.predict(chi_i) for chi_i in chi]

        y_prediction = sum([w * upsilon_i for w, upsilon_i in zip(self.w_m, upsilon)])

        P_xy = sum(
            [
                w
                * (chi_i - x_prediction)[np.newaxis].T
                @ (upsilon_i - y_prediction)[np.newaxis]
                for w, chi_i, upsilon_i in zip(self.w_c, chi, upsilon)
            ]
        )

        R = measurement.R

        P_yy = (
            sum(
                [
                    w
                    * (upsilon_i - y_prediction)[np.newaxis].T
                    @ (upsilon_i - y_prediction)[np.newaxis]
                    for w, upsilon_i in zip(self.w_c, upsilon)
                ]
            )
            + R
        )

        if measurement.implicit:
            phi = lambda z: (1 / np.sqrt(2 * np.pi)) * np.exp(-0.5 * (z ** 2))
            Qfxn = lambda x: 1 - 0.5 * (1 + erf(x / np.sqrt(2)))

            Q_e = P_yy

            nu_minus = -self.env.delta / np.sqrt(Q_e)
            nu_plus = self.env.delta / np.sqrt(Q_e)

            z_bar = (
                (phi(nu_minus) - phi(nu_plus))
                / (Qfxn(nu_minus) - Qfxn(nu_plus))
                * np.sqrt(Q_e)
            )
            var_theta = (
                (phi(nu_minus) - phi(nu_plus)) / (Qfxn(nu_minus) - Qfxn(nu_plus))
            ) ** 2 - (
                ((nu_minus) * phi(nu_minus) - nu_plus * phi(nu_plus))
                / (Qfxn(nu_minus) - Qfxn(nu_plus))
            )
            K = np.array(P_xy / P_yy, ndmin=2)

            measurement.r = z_bar
            # just for plotting residuals
            measurement.t = self.agent.clock.magic_time()
            measurement.P_yy_sigma = np.sqrt(P_yy)
            x_hat = (x_prediction + K * z_bar).ravel()
            P_hat = P_prediction - var_theta * P_yy * (K.T @ K)
            assert 0 <= var_theta <= 1

        else:
            y = measurement.z
            K = np.array(P_xy / P_yy, ndmin=2)

            r = y - y_prediction
            measurement.r = r
            # just for plotting residuals
            measurement.t = self.agent.clock.magic_time()
            measurement.P_yy_sigma = np.sqrt(P_yy)
            x_hat = (x_prediction + K * r).ravel()
            P_hat = P_prediction - P_yy * (K.T @ K)

        if measurement.local and self.env.et:
            meas = copy(measurement)
            meas.local = False

            if np.abs(r) > self.env.delta:
                meas.explicit = True
                meas.implicit = False
            else:
                meas.explicit = False
                meas.implicit = True

            self.event_triggering_messages.append(meas)

        self.x = x_hat
        self.P = P_hat

    def log(self, measurement):
        if measurement.local:
            self.state_log.log_state()
            self.state_log.log_u()

    def step(self):
        self.update_t()

    def set_time_from_measurement(self, measurement):
        if self.agent.name == "Z":
            self.t_kp1 = self.agent.clock.magic_time()
        elif measurement.local:
            self.t_kp1 = measurement.timestamp_receive
        else:
            Delta_R, _ = self.get_clock_estimate(measurement.receiver.name)
            Delta_P, _ = self.get_clock_estimate(self.agent.name)
            self.t_kp1 = measurement.timestamp_receive - Delta_R + Delta_P

        self.t_Q_prev = self.t_Q
        self.t_Q = max(self.t_Q, self.t_kp1)

    def update_t(self):
        self.t_k = self.t_kp1

    def get_tau(self):
        tau = self.t_kp1 - self.t_k

        return tau

    def get_tau_Q(self):
        tau_Q = self.t_Q - self.t_Q_prev

        return tau_Q

    def get_local_info(self):
        Y = inv(self.P)
        y = Y @ self.x
        local_info = Information(y, Y)

        return local_info

    def get_event_triggering_measurements(self):
        et_messages = copy(self.event_triggering_messages)
        self.event_triggering_messages = []
        return et_messages

    def get_rover_state_estimate(self):
        x = self.x.copy()

        x_rover = self.env.dynamics.get_rover_state_estimate(self.agent.name, x)

        return x_rover

    def get_clock_estimate(self, agent_name="", x=np.array([])):
        if x.size == 0:
            x = self.x.copy()

        if not agent_name:
            agent_name = self.agent.name

        Delta, Delta_dot = self.env.dynamics.Delta_functions[agent_name](*x)

        return Delta, Delta_dot

    def get_time_estimate(self, x=np.array([])):
        if x.size == 0:
            x = self.x.copy()

        Delta, _ = self.env.dynamics.Delta_functions[self.agent.name](*x)

        t_estimate = self.t_kp1 - Delta

        return t_estimate
