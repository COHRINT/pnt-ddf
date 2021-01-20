from functools import reduce

import numpy as np
import pandas as pd
from bpdb import set_trace
from numpy.random import multivariate_normal
from sympy import Matrix
from sympy.utilities.lambdify import lambdify


class GPS:
    def __init__(self, env, agent):
        self.env = env
        self.agent = agent

        self.sigma_gps = self.agent.config.getfloat("sigma_gps")

        self.R_gps = np.eye(2) * self.sigma_gps ** 2

        self.define_measurement()
        self.define_measurement_names()
        self.define_measurement_log()

    def define_measurement(self):
        x, y = self.env.dynamics.get_sym_position(self.agent.name)

        h = Matrix([x, y])
        self.evaluate_h = lambdify(self.env.dynamics.x_vec, np.squeeze(h), "numpy")

    def define_measurement_names(self):
        self.measurement_names = [
            "gps_x_{}".format(self.agent.name),
            "gps_y_{}".format(self.agent.name),
        ]
        self.measurement_names_latex = [
            "GPS $x$ {}".format(self.agent.name),
            "GPS $y$ {}".format(self.agent.name),
        ]

    def define_measurement_log(self):
        self._log_measurement = []
        self._log_residuals_pre_fit = []
        self._log_residuals_post_fit = []
        self._log_residuals_fused = []
        self._log_R_sigma = []
        self._log_P_yy_sigma = []

    def true_measurement(self):
        gps_meas = self.env.dynamics.get_true_position(self.agent.name)

        gps_meas += multivariate_normal(np.zeros(2), self.R_gps)

        return gps_meas

    def predict_measurement(self, x_hat):
        gps_meas = self.evaluate_h(*x_hat)

        return np.array(gps_meas)

    def generate_R(self):
        return self.R_gps

    def log_measurement(self, y):
        measurements = self.sort_measurements(y)

        self._log_measurement.append(measurements)

    def log_residuals(self, r, pre, post, fused):
        residuals = self.sort_measurements(r)

        if pre:
            self._log_residuals_pre_fit.append(residuals)
        elif post:
            self._log_residuals_post_fit.append(residuals)
        elif fused:
            self._log_residuals_fused.append(residuals)

    def log_R(self, R):
        sigma = self.sort_measurements(np.sqrt(np.diag(R)))

        self._log_R_sigma.append(sigma)

    def log_P_yy(self, P_yy):
        sigma = self.sort_measurements(np.sqrt(np.diag(P_yy)))

        self._log_P_yy_sigma.append(sigma)

    def sort_measurements(self, y):
        measurements = np.zeros(y.size + 1)

        t = self.agent.clock.magic_time()

        measurements[0] = t
        measurements[1:] = y

        return measurements

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
