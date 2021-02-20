from copy import copy
from dataclasses import dataclass

from numpy import sqrt
from scipy.constants import c


@dataclass
class Measurement_Data:
    z: float
    r: float
    sigma: float


class Measurement:
    def __init__(self):
        self.measurement_data = Measurement_Data(0, 0, 0)

    def predict_measurement(self, x_hat):
        assert False

    def true_measurement(self):
        assert False

    def get_R(self):
        assert False


class Pseudorange(Measurement):
    def __init__(self, env):
        super().__init__()

        self.env = env

        self.transmitter = None
        self.time_transmit = None

        self.receiver = None
        self.time_receive = None

    @property
    def true_measurement(self):
        self.define_true_measurement()
        return copy(self._true_measurement)

    def define_true_measurement(self):
        self._true_measurement = c * (self.time_receive - self.time_transmit)
        self.measurement_data.z = copy(self._true_measurement)

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
        self.measurement_data.sigma = copy(sqrt(self._R))
