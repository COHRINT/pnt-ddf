import itertools
from functools import reduce

import numpy as np
import pandas as pd
from bpdb import set_trace
from scipy.linalg import block_diag

from pntddf.camera import Camera
from pntddf.gps import GPS
from pntddf.magnetometer import Magnetometer
from pntddf.odometer import Odometer
from pntddf.ranging_module import Ranging_Module


class Sensors:
    def __init__(self, env, agent, centralized=False):
        self.env = env
        self.agent = agent
        self.centralized = centralized

        self.define_sensors()

    def define_sensors(self):
        self.sensors = []

        # Ranging Module
        self.ranging_module = Ranging_Module(self.env, self.agent)
        self.sensors.append(self.ranging_module)

        # GPS
        if self.agent.config.getboolean("gps"):
            self.gps = GPS(self.env, self.agent)
            self.sensors.append(self.gps)

        # Odometer
        if self.agent.config.getboolean("odometer"):
            self.odometer = Odometer(self.env, self.agent)
            self.sensors.append(self.odometer)
            self.measurement_names.append("odom")
            self.measurement_names_latex.append("Odometry")

        # Magnetometer
        if self.agent.config.getboolean("magnetometer"):
            self.magnetometer = Magnetometer(self.env, self.agent)
            self.sensors.append(self.magnetometer)
            self.measurement_names.append("magnetometer")
            self.measurement_names_latex.append("Magnetometer")

        # Camera
        if self.agent.config.getboolean("camera"):
            self.camera = Camera(self.env, self.agent)
            self.sensors.append(self.camera)
            self.measurement_names.append("camera_r")
            self.measurement_names.append("camera_theta")
            self.measurement_names_latex.append("Camera $r$")
            self.measurement_names_latex.append("Camera $\\theta$")

        # Measurement names
        self.measurement_names = []
        self.measurement_names_latex = []

        for sensor in self.sensors:
            self.measurement_names.extend(sensor.measurement_names)
            self.measurement_names_latex.extend(sensor.measurement_names_latex)

    def get_message_queue(self):
        if not self.centralized:
            message_queue = self.agent.estimator.message_queue
        else:
            message_queue = self.env.agent_centralized.estimator.message_queue

        return message_queue

    def true_measurement(self):
        message_queue = self.get_message_queue()

        y = [
            sensor.true_measurement()
            if not isinstance(sensor, Ranging_Module)
            else sensor.true_measurement(message_queue)
            for sensor in self.sensors
        ]

        self.y_lengths = list(map(len, y))

        return np.concatenate(y).astype(np.float)

    def predict_measurement(self, x_hat):
        message_queue = self.get_message_queue()

        y_prediction = [
            sensor.predict_measurement(x_hat)
            if not isinstance(sensor, Ranging_Module)
            else sensor.predict_measurement(x_hat, message_queue)
            for sensor in self.sensors
        ]

        return np.concatenate(y_prediction).astype(np.float)

    def generate_R(self, x_hat):
        message_queue = self.get_message_queue()

        R = [
            sensor.generate_R()
            if not isinstance(sensor, Ranging_Module)
            else sensor.generate_R(x_hat, message_queue)
            for sensor in self.sensors
        ]

        R = block_diag(*R)

        return R

    def log_measurements(self, y):
        message_queue = self.get_message_queue()

        y_split = self.split(y)

        for sensor, y_slice in zip(self.sensors, y_split):
            if not isinstance(sensor, Ranging_Module):
                sensor.log_measurement(y_slice)
            else:
                sensor.log_measurement(y_slice, message_queue)

    def log_residuals(self, r, pre=False, post=False, fused=False):
        assert pre or post or fused

        message_queue = self.get_message_queue()

        r_split = self.split(r)

        for sensor, r_slice in zip(self.sensors, r_split):
            if not isinstance(sensor, Ranging_Module):
                sensor.log_residuals(r_slice, pre, post, fused)
            else:
                sensor.log_residuals(r_slice, message_queue, pre, post, fused)

    def log_R(self, R):
        R_split = self.split(R)

        message_queue = self.get_message_queue()

        for sensor, R_slice in zip(self.sensors, R_split):
            if not isinstance(sensor, Ranging_Module):
                sensor.log_R(R_slice)
            else:
                sensor.log_R(R_slice, message_queue)

    def log_P_yy(self, P_yy):
        P_yy_split = self.split(P_yy)

        message_queue = self.get_message_queue()

        for sensor, P_yy_slice in zip(self.sensors, P_yy_split):
            if not isinstance(sensor, Ranging_Module):
                sensor.log_P_yy(P_yy_slice)
            else:
                sensor.log_P_yy(P_yy_slice, message_queue)

    def split(self, a):
        if a.ndim == 1:
            split = np.split(a, np.cumsum(self.y_lengths))[: len(self.y_lengths)]
        elif a.ndim == 2:
            split = []
            for y_length, y_cum in zip(self.y_lengths, np.cumsum(self.y_lengths)):
                begin = y_cum - y_length
                end = y_cum
                split.append(a[begin:end, begin:end])

        return split

    def get_residuals_log_df(self):
        dfs = []

        for sensor in self.sensors:
            dfs.append(sensor.get_df())

        df = reduce(lambda left, right: pd.merge(left, right, on="t"), dfs)

        return df
