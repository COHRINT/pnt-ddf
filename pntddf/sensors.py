import numpy as np
from bpdb import set_trace
from sympy import Matrix, symbols
from sympy.utilities.lambdify import lambdify


class Sensors:
    def __init__(self, env, agent):
        self.env = env
        self.agent = agent

        self.define_measurement_models()

    def define_measurement_models(self):
        self.define_pseudorange_model()

    def define_pseudorange_model(self):
        self.evaluate_pseudorange = {}

        tau_process_func = (
            lambda Y, Z: (
                (self.env.AGENT_NAMES.index(Y) - self.env.AGENT_NAMES.index(Z))
                % self.env.NUM_AGENTS
            )
            * self.env.TRANSMISSION_WINDOW
        )

        for T in self.env.AGENT_NAMES:
            for R in self.env.AGENT_NAMES:
                for P in self.env.AGENT_NAMES:
                    if T == R:
                        continue

                    # processing times
                    tau_process_RT = tau_process_func(R, T)
                    tau_process_PR = tau_process_func(P, R)

                    # symbolic position of processor
                    x_P = self.env.dynamics.get_sym_position(P)

                    # symbolic position of receiver
                    x_R_rT = self.env.dynamics.get_sym_position(R)
                    # symbolic velocity of receiver
                    x_dot_R_rT = self.env.dynamics.get_sym_velocity(R)
                    # symbolic delay of receiver
                    b_R_rT = self.env.dynamics.get_sym("b", R)
                    # symbolic delay rate of receiver
                    b_dot_R_rT = self.env.dynamics.get_sym("b_dot", R)

                    # symbolic position of transmitter
                    x_T_rT = self.env.dynamics.get_sym_position(T)
                    # symbolic velocity of transmitter
                    x_dot_T_rT = self.env.dynamics.get_sym_velocity(T)
                    # symbolic delay of transmitter
                    b_T_rT = self.env.dynamics.get_sym("b", T)
                    # symbolic delay rate of transmitter
                    b_dot_T_rT = self.env.dynamics.get_sym("b_dot", T)

                    # time-of-flight
                    # tau_dist_PR = Matrix(x_P - x_R_rT).norm() / self.env.c
                    # tau_dist_RT = Matrix(x_R_rT - x_T_rT).norm() / self.env.c
                    # These are approximately zero and the symbolic norm() takes forever
                    tau_dist_PR = 0
                    tau_dist_RT = 0

                    # symbolic position/bias of receiver when message was received
                    tau_receiver = tau_process_PR + tau_dist_PR + tau_process_RT
                    x_R_tR = x_R_rT - x_dot_R_rT * tau_receiver
                    b_R_tR = b_R_rT - b_dot_R_rT * tau_receiver

                    # symbolic position/bias of transmitter when message was transmitted
                    tau_transmitter = tau_receiver + tau_dist_RT
                    x_T_tT = x_T_rT - x_dot_T_rT * tau_transmitter
                    b_T_tT = b_T_rT - b_dot_T_rT * tau_transmitter

                    # symbolic distance
                    d = Matrix(x_R_tR - x_T_tT).norm()

                    # pseudorange measurement
                    rho = d + b_R_tR - b_T_tT

                    h = Matrix([rho])
                    x_vec = self.env.dynamics.x_vec
                    self.evaluate_pseudorange[T + R + P] = lambdify(
                        x_vec, np.squeeze(h), "numpy"
                    )
