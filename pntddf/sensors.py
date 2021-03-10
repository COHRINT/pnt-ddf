import numpy as np
from bpdb import set_trace
from scipy.constants import c
from sympy import Matrix, symbols
from sympy.utilities.lambdify import lambdify


class Sensors:
    def __init__(self, env, agent):
        self.env = env
        self.agent = agent

        self.define_measurement_models()

    def define_measurement_models(self):
        self.define_pseudorange_model()
        self.define_gps_model()

    def define_pseudorange_model(self):
        self.evaluate_pseudorange = {}
        self.evaluate_pseudorange_jac = {}
        self.evaluate_pseudorange_R = {}

        x_vec = self.env.dynamics.x_vec
        sigma_read_func = lambda agent_name: self.env.agent_configs[
            agent_name
        ].getfloat("sigma_clock_reading")
        R_read_func = lambda agent_name_0, agent_name_1: self.env.c ** 2 * (
            sigma_read_func(agent_name_0) ** 2 + sigma_read_func(agent_name_1) ** 2
        )

        for T in self.env.AGENT_NAMES:
            for R in self.env.AGENT_NAMES:
                if T == R:
                    continue
                # receiver states
                x_R = self.env.dynamics.get_sym_position(R)
                b_R = self.env.dynamics.get_sym("b", R)

                # transmitter states
                x_T = self.env.dynamics.get_sym_position(T)
                x_dot_T = self.env.dynamics.get_sym_velocity(T)
                b_T = self.env.dynamics.get_sym("b", T)
                b_dot_T = self.env.dynamics.get_sym("b_dot", T)

                # distance
                d = Matrix(x_R - x_T).norm()

                # transmit time
                tau_dist = d / c

                # transmitter states at transmit time
                x_T = x_T - x_dot_T * tau_dist
                b_T = b_T - b_dot_T * tau_dist

                # pseudorange measurement
                rho = d + b_R - b_T
                h = Matrix([rho])

                dh_dx = h.jacobian(x_vec)

                # pseudorange noise
                R_matrix = R_read_func(T, R)

                # lambdify
                TR = T + R
                self.evaluate_pseudorange[TR] = lambdify(x_vec, np.squeeze(h), "numpy")
                self.evaluate_pseudorange_jac[TR] = lambdify(
                    x_vec, np.squeeze(dh_dx), "numpy"
                )
                self.evaluate_pseudorange_R[TR] = R_matrix

    def define_gps_model(self):
        self.evaluate_gps = {}

        x_vec = self.env.dynamics.x_vec

        for agent_name in self.env.AGENT_NAMES:
            x = self.env.dynamics.get_sym_position(agent_name)
            h = Matrix([x])

            self.evaluate_gps[agent_name] = lambdify(x_vec, np.squeeze(h), "numpy")
