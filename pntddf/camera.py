import numpy as np
from bpdb import set_trace
from scipy.linalg import block_diag
from sympy import Matrix, atan2
from sympy.utilities.lambdify import lambdify


class Camera:
    def __init__(self, env, agent):
        self.env = env
        self.agent = agent

        self.sigma_camera_r = self.agent.config.getfloat("sigma_camera_r")
        self.sigma_camera_theta = self.agent.config.getfloat("sigma_camera_theta")

        self.camera_range = self.agent.config.getfloat("camera_range")

        self.R_camera = np.array(
            [
                [self.sigma_camera_r ** 2, 0],
                [0, self.sigma_camera_theta ** 2],
            ]
        )

        self.define_measurement()

    def define_measurement(self):
        self.evaluate_h = dict()

        for agent_name in self.env.AGENT_NAMES:
            if agent_name == self.agent.name:
                continue
            agent_position = self.agent.dynamics.get_sym_position(agent_name)
            self_position = self.agent.dynamics.get_sym_position(self.agent.name)

            # Range
            r = Matrix(agent_position - self_position).norm()

            # Get estimated heading
            rho = agent_position - self_position

            theta = atan2(rho[1], rho[0])

            h = Matrix([r, theta])

            self.evaluate_h[agent_name] = lambdify(
                self.agent.dynamics.x_vec, np.squeeze(h), modules=["numpy", "sympy"]
            )

    def true_measurement(self):
        self.agents_in_view = []
        camera_measurement_pairs = []

        for agent_name in self.env.AGENT_NAMES:
            if agent_name == self.agent.name:
                continue

            agent = self.env.agent_dict[agent_name]
            agent_position = agent.dynamics.get_true_position()

            self_position = self.agent.dynamics.get_true_position()

            # Relative vector
            rho = agent_position - self_position

            # Range
            r = np.linalg.norm(rho)

            if r > self.camera_range:
                continue
            else:
                self.agents_in_view.append(agent_name)

            # Theta
            theta = np.arctan2(rho[1], rho[0])

            # Add noise
            r += np.random.normal(0, self.sigma_camera_r)
            theta += np.random.normal(0, self.sigma_camera_theta)

            camera_measurement_pairs.append([r, theta])

        return camera_measurement_pairs

    def predict_measurement(self, x_hat):
        camera_measurement_pairs = []

        for agent_name in self.agents_in_view:
            r, theta = self.evaluate_h[agent_name](*x_hat)
            camera_measurement_pairs.append([r, theta])

        return camera_measurement_pairs

    def generate_R(self):
        R_block = block_diag(*[self.R_camera] * len(self.agents_in_view))
        return R_block
