from copy import copy
from dataclasses import dataclass

from bpdb import set_trace
from numpy import sqrt
from scipy.constants import c


@dataclass
class Measurement_Data:
    t: float
    z: float
    r: float
    sigma: float
    P_yy_sigma: float
    name: str
    latex_name: str
    local: bool
    implicit: bool
    explicit: bool


class Measurement:
    def __init__(self):
        self._t = None
        self._z = None
        self._r = None
        self._sigma = None
        self._P_yy_sigma = None
        self._name = None
        self._latex_name = None

        self.local = False
        self.implicit = False
        self.explicit = False

        self._processor = None

    @property
    def processor(self):
        return self._processor

    @processor.setter
    def processor(self, processor):
        self._processor = processor

    @property
    def data(self):
        return Measurement_Data(
            self._t,
            self._z,
            self._r,
            self._sigma,
            self._P_yy_sigma,
            self._name,
            self._latex_name,
            self.local,
            self.implicit,
            self.explicit,
        )

    @property
    def z(self):
        return self._z

    @z.setter
    def z(self, z):
        self._z = z

    @property
    def t(self):
        return self._t

    @t.setter
    def t(self, t):
        self._t = t

    @property
    def r(self):
        return self._r

    @r.setter
    def r(self, r):
        self._r = r

    @property
    def P_yy_sigma(self):
        return self._P_yy_sigma

    @P_yy_sigma.setter
    def P_yy_sigma(self, P_yy_sigma):
        self._P_yy_sigma = P_yy_sigma

    @property
    def sigma(self):
        return self._sigma

    @sigma.setter
    def sigma(self, sigma):
        self._sigma = sigma

    def __repr__(self):
        return "{} = {:.2f}".format(self._name, self.true)


class Pseudorange(Measurement):
    def __init__(self, env):
        super().__init__()

        self.env = env

        self._transmitter = None
        self._timestamp_transmit = None

        self._receiver = None
        self._timestamp_receive = None

    @property
    def transmitter(self):
        return self._transmitter

    @transmitter.setter
    def transmitter(self, transmitter):
        assert self._transmitter is None
        self._transmitter = transmitter

    @property
    def timestamp_transmit(self):
        return self._timestamp_transmit

    @timestamp_transmit.setter
    def timestamp_transmit(self, timestamp_transmit):
        assert self._timestamp_transmit is None
        self._timestamp_transmit = timestamp_transmit

    @property
    def receiver(self):
        return self._receiver

    @receiver.setter
    def receiver(self, receiver):
        assert self._receiver is None
        self._receiver = receiver

    @property
    def timestamp_receive(self):
        return self._timestamp_receive

    @timestamp_receive.setter
    def timestamp_receive(self, timestamp_receive):
        assert self._timestamp_receive is None
        self._timestamp_receive = timestamp_receive

    @property
    def true(self):
        self.define_true_measurement()
        return copy(self._true_measurement)

    def predict(self, x_hat):
        TRP = self.transmitter.name + self.receiver.name + self.processor.name

        prediction_func = self.processor.sensors.evaluate_pseudorange[TRP]

        rho = prediction_func(*x_hat)

        return rho

    def define_true_measurement(self):
        self._true_measurement = c * (self.timestamp_receive - self.timestamp_transmit)
        self._name = "rho_{}{}".format(self.receiver.name, self.transmitter.name)
        self._latex_name = "$\\rho_{{{}{}}}$".format(
            self.receiver.name, self.transmitter.name
        )
        self._z = copy(self._true_measurement)

    @property
    def R(self):
        self.define_R()
        return copy(self._R)

    def define_R(self):
        R_sigma_clock = self.env.agent_configs[self.receiver.name].getfloat(
            "sigma_clock_reading"
        )
        T_sigma_clock = self.env.agent_configs[self.transmitter.name].getfloat(
            "sigma_clock_reading"
        )

        self._R = c ** 2 * (R_sigma_clock ** 2 + T_sigma_clock ** 2)
        self._sigma = copy(sqrt(self._R))
