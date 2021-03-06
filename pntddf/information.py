from copy import copy

import numpy as np
from bpdb import set_trace
from numpy.linalg import inv


def invert(y, Y):
    try:
        P = inv(Y)
        x = P @ y
    except:
        P = np.full_like(Y, np.nan)
        x = np.full_like(y, np.nan)

    return x, P


def empty_info(n):
    return Information(np.zeros([n]), np.zeros([n, n]))


class Information:
    def __init__(self, y, Y):
        assert y.size == Y.shape[0] == Y.shape[1], "Incorrect y/Y dimensions"

        self.y = y
        self.Y = Y

        self.transmitter = None
        self.timestamp_transmit = None
        self.receiver = None
        self.timestamp_receive = None

    @property
    def Y(self):
        return self._Y

    @Y.setter
    def Y(self, Y):
        self._Y = Y

    @property
    def P(self):
        return self.invert()[1]

    @property
    def x(self):
        return self.invert()[0]

    def invert(self):
        return invert(self.y, self.Y)

    def __add__(self, other):
        return Information(self.y + other.y, self.Y + other.Y)

    def __radd__(self, other):
        if other == 0:
            return self
        else:
            return self.__add__(other)

    def __sub__(self, other):
        return Information(self.y - other.y, self.Y - other.Y)

    def __eq__(self, other):
        return np.allclose(self.y, other.y) and np.allclose(self.Y, other.Y)

    def isempty(self):
        return np.allclose(self.y, np.zeros(self.y.shape)) and np.allclose(
            self.Y, np.zeros(self.Y.shape)
        )

    def __copy__(self):
        new = type(self)(self.y.copy(), self.Y.copy())
        new.transmitter = self.transmitter
        new.timestamp_transmit = self.timestamp_transmit
        new.receiver = self.receiver
        new.timestamp_receive = self.timestamp_receive

        return new

    def __repr__(self):
        repr_str = f"{invert(self.y, self.Y)[0]}"

        return repr_str
