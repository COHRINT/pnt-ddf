import re
from copy import copy

import numpy as np
from bpdb import set_trace
from numpy import sqrt
from numpy.linalg import cholesky, inv, norm
from scipy.linalg import block_diag
from scipy.optimize import least_squares
from scipy.special import erf

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

        self.local_measurements_for_retransmission = []

        self.completed = False
        self.broadcast_time = 2.0

    def estimate(self):
        measurements = self.agent.estimator.measurement_queue

        if self.completed and self.agent.clock.time() > self.broadcast_time:
            return True
        elif self.completed:
            self.add_local_measurements(measurements)
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

        if all_measurement_pairs:
            x0 = self.env.x0.copy()
            lower_bounds = np.array(
                [
                    -1e-3 if re.match(r"[xyz]_dot", str(state)) else -np.inf
                    for state in self.env.dynamics.x_vec
                ]
            )
            upper_bounds = np.array(
                [
                    1e-3 if re.match(r"[xyz]_dot", str(state)) else np.inf
                    for state in self.env.dynamics.x_vec
                ]
            )
            bounds = (lower_bounds, upper_bounds)
            lsq_result = least_squares(self.lsq_func, x0, bounds=bounds)

            if lsq_result.success:
                self.completed = True
                H = lsq_result.jac
                R = np.diag([measurement.R for measurement in measurements])
                P = inv(H.T @ inv(R) @ H)
                time_delta = self.env.TRANSMISSION_WINDOW * self.env.NUM_AGENTS
                Q = self.agent.estimator.filt.generate_Q(time_delta)
                self.x = lsq_result.x
                self.P = P + Q

        self.add_local_measurements(measurements)

    def lsq_func(self, x):
        measurements = self.agent.estimator.measurement_queue

        residuals = np.array(
            [measurement.true - measurement.predict(x) for measurement in measurements]
        )
        return residuals

    def add_local_measurements(self, measurements):
        for measurement in measurements:
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

        # self.sensors = agent.sensors

        self.t_k = 0
        self.t_kp1 = 0

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

        Q = self.generate_Q(time_delta=tau)

        P_prediction += Q

        self.x = x_prediction
        self.P = P_prediction

    # def predict_message(self, message):
    #     assert not np.isnan(message.local_info.time)
    #     assert message.local_info.time <= self.t_kp1
    #     if message.local_info.time == self.t_kp1:
    #         return

    #     x_hat, P = invert(message.local_info.y, message.local_info.Y)

    #     distance_estimate = self.env.dynamics.distance_between_agents(
    #         message.receiver.name, message.transmitter.name, x_hat
    #     )

    #     # Delay during transit
    #     tau_distance = distance_estimate / self.env.c

    #     # Delay since received
    #     tau_process = self.t_kp1 - message.time_receive

    #     tau = tau_distance + tau_process

    #     t_estimate = self.get_time_estimate(x_hat)

    #     x_prediction = self.env.dynamics.step_x(x_hat, tau, t_estimate)
    #     P_prediction = self.env.dynamics.step_P(P, tau)

    #     Q = self.generate_Q(time_delta=tau)

    #     P_prediction += Q

    #     message.local_info.Y = inv(P_prediction)
    #     message.local_info.y = message.local_info.Y @ x_prediction

    #     message.propagated = True

    def generate_Q(self, time_delta):
        Q_clock = block_diag(
            *[
                np.array(
                    [
                        [1 / 3 * time_delta ** 3, 1 / 2 * time_delta ** 2],
                        [1 / 2 * time_delta ** 2, 1 / 1 * time_delta ** 1],
                    ]
                )
                * self.env.agent_dict[agent_name].clock.sigma_clock_process ** 2
                * self.env.c ** 2
                for agent_name in self.env.agent_clocks_to_be_estimated
            ]
        )

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
                * self.Q_sigma_target ** 2
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

    def local_update(self):
        x_prediction = self.x.copy()
        P_prediction = self.P.copy()

        # Store reference estimate
        x_ref = x_prediction.copy()
        P_ref = P_prediction.copy()

        for measurement in self.agent.estimator.measurement_queue:
            chi = self.generate_sigma_points(x_prediction, P_prediction)

            upsilon = [measurement.predict(chi_i) for chi_i in chi]

            y_prediction = sum(
                [w * upsilon_i for w, upsilon_i in zip(self.w_m, upsilon)]
            )

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

                mu = measurement.predict(x_prediction) - measurement.predict(x_ref)
                Q_e = P_yy
                alpha = measurement.predict(x_ref) - measurement.predict(x_ref)

                nu_minus = (-self.env.delta + alpha - mu) / np.sqrt(Q_e)
                nu_plus = (self.env.delta + alpha - mu) / np.sqrt(Q_e)

                try:
                    assert Qfxn(nu_minus) - Qfxn(nu_plus) != 0.0
                except:
                    set_trace()

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
                measurement.t = self.agent.clock.magic_time()
                measurement.P_yy_sigma = np.sqrt(P_yy)
                x_prediction = (x_prediction + K * z_bar).ravel()
                P_prediction = P_prediction - var_theta * P_yy * (K.T @ K)

            else:
                y = measurement.true
                K = np.array(P_xy / P_yy, ndmin=2)

                r = y - y_prediction
                measurement.r = r
                measurement.t = self.agent.clock.magic_time()
                measurement.P_yy_sigma = np.sqrt(P_yy)
                x_prediction = (x_prediction + K * r).ravel()
                P_prediction = P_prediction - P_yy * (K.T @ K)

            if measurement.local:
                # Check if measurement is expected to be surprising
                y_ref = measurement.predict(x_ref)
                y = measurement.true
                r_ref = y - y_ref

                meas = copy(measurement)
                meas.local = False
                # if self.agent.name == "T":
                # P_b_T = P_prediction[2, 2]
                # P_x_T = P_prediction[4, 4]
                # P_xb_T = P_prediction[2, 4]
                if np.abs(r_ref) > self.env.delta:
                    meas.explicit = True
                    meas.implicit = False
                else:
                    meas.explicit = False
                    meas.implicit = True

                self.event_triggering_messages.append(meas)

        self.x = x_prediction
        self.P = P_prediction

    # def fusion_update(self):
    #     # Local
    #     local_info = Information(copy(self.y_k_k), copy(self.Y_k_k))
    #     x_local_info = invert(local_info.y, local_info.Y)[0]

    #     # x_true = self.state_log.get_true()

    #     # Message
    #     messages_info = []

    #     for message in self.agent.estimator.message_queue:
    #         message_info = message.local_info
    #         messages_info.append(message_info)

    #         x_message_info = invert(message_info.y, message_info.Y)[0]

    #     # Fused
    #     information_set = [local_info] + messages_info
    #     fused_info = covariance_intersection(information_set, fast=True)
    #     # fused_info = covariance_intersection(information_set, sensors=self.sensors)
    #     x_fused_info = invert(fused_info.y, fused_info.Y)[0]

    #     y = self.sensors.true_measurement()

    #     y_prediction = self.sensors.predict_measurement(x_fused_info)

    #     r_post_fusion = y - y_prediction

    #     self.sensors.log_residuals(r_post_fusion, fused=True)

    #     self.y_k_k = fused_info.y
    #     self.Y_k_k = fused_info.Y

    def step(self):
        self.state_log.log_state()
        self.state_log.log_u()
        self.state_log.log_true()
        self.state_log.log_NEES_errors()

        self.update_t()

    def set_time(self):
        self.t_kp1 = self.agent.clock.time()

    def update_t(self):
        self.t_k = self.t_kp1

    def get_tau(self):
        tau = self.t_kp1 - self.t_k

        return tau

    # def get_local_info(self):
    # local_info_copy = copy(self.local_info)
    # local_info_copy.time = copy(self.t_kp1)
    # return local_info_copy

    def get_event_triggering_measurements(self):
        et_messages = copy(self.event_triggering_messages)
        self.event_triggering_messages = []
        return et_messages

    def get_rover_state_estimate(self):
        x = self.x.copy()

        x_rover = self.env.dynamics.get_rover_state_estimate(self.agent.name, x)

        return x_rover

    def get_clock_estimate(self, agent_name=""):
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
