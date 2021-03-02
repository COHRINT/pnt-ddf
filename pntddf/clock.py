import numpy as np
from bpdb import set_trace
from numpy.random import multivariate_normal, normal
from sympy import Matrix, exp, symbols
from sympy.utilities.lambdify import lambdify


class Clock:
    def __init__(self, env, agent):
        self.env = env
        self.agent = agent

        self.time_previous = 0

        self.define_parameters()
        self.define_noise()

        self.derivation()

    def define_parameters(self):
        if self.agent.config.getboolean("perfect_clock"):
            self.alpha = 1.0
        else:
            sigma_Delta_dot = self.env.P0_config.getfloat("sigma_b_dot") / self.env.c
            self.alpha = 1.0 + np.random.normal(0, sigma_Delta_dot)

        if self.agent.config.getboolean("perfect_clock"):
            self.beta = 0.0
        else:
            sigma_Delta = self.env.P0_config.getfloat("sigma_b") / self.env.c
            self.beta = 0.0 + np.random.normal(0, sigma_Delta)

        self._Delta = self.beta
        self._Delta_dot = self.alpha - 1

        self.q = np.array([self.beta, self.alpha])

    def define_noise(self):
        if self.agent.config.getboolean("perfect_clock"):
            self.sigma_clock_process = 0
        else:
            self.sigma_clock_process = self.agent.config.getfloat("sigma_clock_process")

        self.sigma_clock_reading = self.agent.config.getfloat("sigma_clock_reading")
        # self.sigma_clock_reading = 0
        # self.sigma_clock_process = 0

    def update_time(self):
        # T is time delta since last clock check
        T = self.env.now - self.time_previous
        self.time_previous = self.env.now

        self.q = self.F(T) @ self.q + np.random.multivariate_normal(
            0 * self.q, self.Q(T)
        )

        self._Delta = self.q[0] - self.env.now
        self._Delta_dot = self.q[1] - 1

        self._alpha = self.q[1]

    def time(self):
        self.update_time()

        return self.q[0] + np.random.normal(0, self.sigma_clock_reading)

    @property
    def Delta(self):
        self.update_time()
        return self._Delta

    @property
    def b(self):
        self.update_time()
        return self._Delta * self.env.c

    @property
    def Delta_dot(self):
        self.update_time()
        return self._Delta_dot

    @property
    def b_dot(self):
        self.update_time()
        return self._Delta_dot * self.env.c

    def magic_time(self):
        return self.env.now

    def get_transmit_wait(self):
        Delta, Delta_dot = self.agent.estimator.get_clock_estimate()

        t = self.time()
        index = self.agent.index
        window = self.env.TRANSMISSION_WINDOW
        num_agents = self.env.NUM_AGENTS
        cycle_time = window * num_agents

        cycle_number = np.floor((t + cycle_time - index * window) / cycle_time)
        cycle_number = int(cycle_number)

        if cycle_number == self.agent.radio.cycle_number_previous:
            cycle_number += 1

        t_transmit = cycle_number * cycle_time + index * window

        # Force Delta_dot to be greater than -0.5 (should be greater than -1 but don't want /0 error)
        Delta_dot = np.max([-0.5, Delta_dot])

        # Estimated true wait time to plug into simpy timeout
        wait_time = (t_transmit - (t - Delta)) / (1 + Delta_dot)

        if wait_time < 0:
            wait_time = window

        return wait_time, cycle_number

    def derivation(self):
        # q = [ h[t] h_dot[t] ]
        # q_dot = A @ q + w
        # A = [[ 0 1 ]
        #      [ 0 0 ]]
        # w ~ N(0, Q)
        # Q = [[ 0 0 ]
        #      [ 0 sigma_q ** 2 ]]

        A = np.array(
            [
                [0, 1],
                [0, 0],
            ]
        )

        Q = np.array(
            [
                [0, 0],
                [0, self.sigma_clock_process ** 2],
            ]
        )

        T = symbols("T", real=True)

        Z = Matrix(np.block([[-A, Q], [np.zeros([2, 2]), A.T]])) * T

        eZ = exp(Z)

        F = eZ[2:, 2:].T
        Q_d = F @ eZ[:2, 2:]

        self.F = lambdify(T, F)
        self.Q = lambdify(T, Q_d)
