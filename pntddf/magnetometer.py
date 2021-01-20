import numpy as np
from bpdb import set_trace
from sympy import Matrix, atan2
from sympy.utilities.lambdify import lambdify


class Magnetometer:
    def __init__(self, env, agent):
        self.env = env
        self.agent = agent

        self.sigma_magnetometer = self.agent.config.getfloat("sigma_magnetometer")

        self.R_magnetometer = np.eye(1) * self.sigma_magnetometer ** 2

        self.define_measurement()

    def define_measurement(self):
        x_dot, y_dot = self.agent.dynamics.get_sym_velocity(self.agent.name)

        phi = atan2(y_dot, x_dot)

        h = Matrix([phi])
        self.evaluate_h = lambdify(
            self.agent.dynamics.x_vec, np.squeeze(h), modules=["numpy", "sympy"]
        )

    def true_measurement(self):
        x_dot, y_dot = self.agent.dynamics.get_true_velocity()

        phi = np.arctan2(y_dot, x_dot)

        # phi += np.random.normal(0, self.sigma_magnetometer)

        return np.array([phi])

    def predict_measurement(self, x_hat):
        phi = self.evaluate_h(*x_hat)

        return np.array([phi])

    def generate_R(self):
        return self.R_magnetometer
