import numpy as np
from bpdb import set_trace
from numpy.linalg import norm
from sympy import Matrix, sqrt, symbols
from sympy.utilities.lambdify import lambdify


class Odometer:
    def __init__(self, env, agent):
        self.env = env
        self.agent = agent

        self.time_previous = self.agent.clock.magic_time()

        self.sigma_odometer = self.agent.config.getfloat("sigma_odometer")

        self.R_odometer = np.eye(1) * self.sigma_odometer ** 2

        self.define_measurement()

    def define_measurement(self):
        tau = symbols("tau")

        tau_expr = tau / (
            1 + self.agent.dynamics.get_sym("b_dot", self.agent.name) / self.env.c
        )

        x_dot, y_dot = self.agent.dynamics.get_sym_velocity(self.agent.name)

        displacement_delta = tau_expr * sqrt(x_dot ** 2 + y_dot ** 2)

        h = Matrix([displacement_delta])
        self.evaluate_h = lambdify(
            [tau] + self.agent.dynamics.x_vec, np.squeeze(h), "numpy"
        )

    def true_measurement(self):
        time_current = self.agent.clock.magic_time()
        time_delta = time_current - self.time_previous
        self.time_previous = time_current

        displacement_delta = norm(self.agent.dynamics.get_true_velocity()) * time_delta

        displacement_delta += np.random.normal(0, self.sigma_odometer)

        return np.array([displacement_delta])

    def predict_measurement(self, x_hat):
        tau = self.agent.radio.get_tau()

        displacement_delta = self.evaluate_h(tau, *x_hat)

        return np.array([displacement_delta])

    def generate_R(self):
        return self.R_odometer
