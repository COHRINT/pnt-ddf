from copy import copy

import numpy as np
from bpdb import set_trace
from numpy import sqrt
from scipy.constants import c


class Measurement:
    def __init__(self, env):
        self.env = env

        self.t = None
        self.z = None
        self.r = None
        self.sigma = None
        self.R = None
        self.P_yy_sigma = None

        self.name = None
        self.latex_name = None

        self._local = True
        self._explicit = False
        self._implicit = False

    @property
    def local(self):
        # only one of these can be true
        assert self._local ^ self._explicit ^ self._implicit
        return self._local

    @local.setter
    def local(self, local):
        self._local = local

    @property
    def explicit(self):
        assert self._local ^ self._explicit ^ self._implicit
        return self._explicit

    @explicit.setter
    def explicit(self, explicit):
        self._explicit = explicit

    @property
    def implicit(self):
        assert self._local ^ self._explicit ^ self._implicit
        return self._implicit

    @implicit.setter
    def implicit(self, implicit):
        self._implicit = implicit

    def __repr__(self):
        return "{} = {:.2f}".format(self.name, self.z)

    def to_ros_message(self):
        from pntddf_ros.msg import Measurement as Measurement_ROS

        measurement = Measurement_ROS()
        measurement.z = self.z
        measurement.sigma = self.sigma
        measurement.implicit = self.implicit

        return measurement


class Pseudorange(Measurement):
    def __init__(
        self, env, transmitter, receiver, timestamp_transmit, timestamp_receive
    ):
        super().__init__(env)

        self.transmitter = transmitter
        self.receiver = receiver

        self.timestamp_transmit = timestamp_transmit
        self.timestamp_receive = timestamp_receive

        self.define_measurement()

    def define_measurement(self):
        self.z = c * (self.timestamp_receive - self.timestamp_transmit)

        TR = self.transmitter.name + self.receiver.name
        R = self.env.sensors.evaluate_pseudorange_R[TR]
        self.R = R
        self.sigma = sqrt(R)

        self.name = "rho_{}{}".format(self.receiver.name, self.transmitter.name)
        self.latex_name = "$\\rho_{{{}{}}}$".format(
            self.receiver.name, self.transmitter.name
        )

    def predict(self, x_hat):
        TR = self.transmitter.name + self.receiver.name

        prediction_func = self.env.sensors.evaluate_pseudorange[TR]

        rho = prediction_func(*x_hat)

        return np.array(rho)


class GPS_Measurement(Measurement):
    def __init__(self, env, z, axis, agent, timestamp_receive):
        super().__init__(env)

        self.z = z
        self.axis = axis
        self.dim_name = self.env.dim_names[self.axis]

        self.receiver = agent
        self.agent = agent

        self.timestamp_receive = timestamp_receive

        self.define_measurement()

    def define_measurement(self):
        R = self.env.sensors.evaluate_gps_R[self.agent.name]
        self.R = R
        self.sigma = np.sqrt(R)

        self.name = "gps_{}_{}".format(self.env.dim_names[self.axis], self.agent.name)
        self.latex_name = "GPS {} {}".format(self.dim_name, self.agent.name)

    def predict(self, x_hat):
        prediction_func = self.env.sensors.evaluate_gps[self.agent.name]

        pos = prediction_func(*x_hat)

        if type(pos) == float or type(pos) == np.float64:
            pass
        else:
            pos = pos[self.axis]

        return pos


class Asset_Detection(Measurement):
    def __init__(self, env, z, var, axis, agent, timestamp_receive):
        super().__init__(env)

        self.z = z
        self.sigma = np.sqrt(var)
        self.R = var

        self.axis = axis
        self.dim_name = self.env.dim_names[self.axis]

        self.receiver = agent
        self.agent = agent

        self.timestamp_receive = timestamp_receive

        self.define_measurement()

    def define_measurement(self):
        self.name = "detection_{}_{}".format(
            self.env.dim_names[self.axis], self.agent.name
        )
        self.latex_name = "Detection {} {}".format(self.dim_name, self.agent.name)

    def predict(self, x_hat):
        prediction_func = self.env.sensors.evaluate_gps[self.agent.name]

        pos = prediction_func(*x_hat)

        if type(pos) == float or type(pos) == np.float64:
            pass
        else:
            pos = pos[self.axis]

        return pos
