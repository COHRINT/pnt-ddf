import numpy as np
from bpdb import set_trace
from numpy import cos, pi, sign, sin, sqrt, square, sum
from numpy.linalg import inv
from scipy.signal import place_poles
from sympy import BlockMatrix, Matrix, diag, exp, eye, integrate, symbols
from sympy.utilities.lambdify import lambdify


class Rover:
    def __init__(self, env, agent):
        self.env = env
        self.agent = agent
        self.agent_name = agent.name
        self.agent_config = self.env.agent_configs[agent.name]

        self.setup_x()
        self.setup_controls()

        self.setup_symbols()
        self.setup_dynamics()

        self.t_previous = 0

    def setup_x(self):
        if self.agent_config.getboolean("random_initial_state"):
            self.x = []
            for state in self.env.ROVER_STATES:
                mean = self.env.x0_config.getfloat("{}_hat".format(state))
                sigma = self.env.P0_config.getfloat("sigma_{}".format(state))
                value = np.random.normal(mean, sigma)

                self.x.append(value)

            self.x = np.array(self.x)
        else:
            self.x = np.array(
                [self.agent_config.getfloat(v + "_0") for v in self.env.ROVER_STATES]
            )

    def setup_controls(self):
        self.u_x_mult = self.agent.config.getfloat("u_x_mult")
        self.u_y_mult = self.agent.config.getfloat("u_y_mult")

    def setup_symbols(self):
        self.x_vec = []

        for state_name in self.env.ROVER_STATES:
            state_name += "_{}".format(self.agent_name)
            state_name_sym = symbols(state_name, real=True)
            setattr(self, state_name + "_sym", state_name_sym)

            self.x_vec.append(state_name_sym)

    def setup_dynamics(self):
        self.A = Matrix(
            np.block(
                [
                    [
                        np.zeros([self.env.n_dim, self.env.n_dim]),
                        np.eye(self.env.n_dim),
                    ],
                    [
                        np.zeros([self.env.n_dim, self.env.n_dim]),
                        np.zeros([self.env.n_dim, self.env.n_dim]),
                    ],
                ]
            )
        )

        A = np.array(self.A, dtype=np.float)

        self.B = Matrix(
            np.block(
                [
                    [np.zeros([self.env.n_dim, self.env.n_dim])],
                    [np.eye(self.env.n_dim)],
                ]
            )
        )
        B = np.array(self.B, dtype=np.float)

        C = np.block(
            [[np.eye(self.env.n_dim), np.zeros([self.env.n_dim, self.env.n_dim])]],
        )

        lambda_desired = np.linspace(-1, -2, self.env.n_dim * 2) * 1e-2
        self.K = place_poles(A, B, lambda_desired).gain_matrix
        self.L = inv(C @ inv(-A + B @ self.K) @ B)

        alpha = symbols("alpha", real=True, positive=True)
        Delta_t = symbols("Delta_t", real=True, positive=True)

        self.F = (self.A * Delta_t).exp()
        self.G = (
            self.F @ integrate((-self.A * alpha).exp(), (alpha, 0, Delta_t)) @ self.B
        )

        self.F_fxn = lambdify(Delta_t, self.F)
        self.G_fxn = lambdify(Delta_t, self.G)

    def u(self, t):
        initial_wait = 5.0

        if t < initial_wait:
            return np.zeros(self.env.n_dim)

        R = 0.01
        omega = 2 * np.pi / 1000

        u = np.array(
            [
                self.u_x_mult * R * np.sin(omega * t),
                self.u_y_mult * R * np.cos(omega * t),
            ]
        )[: self.env.n_dim]

        return u

    def update_state(self):
        t = self.agent.estimator.filt.get_time_estimate()
        Delta_t = max(0, t - self.t_previous)

        noise_scale = 0.001

        F = self.F_fxn(Delta_t)
        x = self.x  # True state
        G = self.G_fxn(Delta_t)
        # Controls applied using estimated state
        u = self.u(t) + np.random.normal(0, noise_scale * Delta_t, self.env.n_dim)

        self.x = F @ x + G @ u

        self.t_previous = t

    def get_sym(self, variable):
        return getattr(self, variable + "_" + self.agent.name + "_sym")

    def get_sym_position(self):
        return np.array([self.get_sym(v) for v in self.env.dim_names])

    def get_true_position(self):
        self.update_state()

        return self.x[: self.env.n_dim].copy()

    def get_sym_velocity(self):
        return np.array([self.get_sym(v + "_dot") for v in self.env.dim_names])

    def get_true_velocity(self):
        self.update_state()

        return self.x[self.env.n_dim :].copy()


class Beacon:
    def __init__(self, env, agent):
        self.env = env
        self.agent = agent

        self.setup()

    def setup(self):
        for dim in self.env.dim_names:
            setattr(
                self,
                "{}_true".format(dim),
                self.env.agent_configs[self.agent.name].getfloat(dim),
            )

        self.position_true = np.array(
            [getattr(self, "{}_true".format(dim)) for dim in self.env.dim_names]
        )
        self.velocity_true = np.zeros(self.env.n_dim)

    def get_true_position(self):
        return self.position_true.copy()

    def get_true_velocity(self):
        return self.velocity_true.copy()


class Clock:
    def __init__(self, env, agent):
        self.env = env
        self.agent = agent

        self.setup_symbols()
        self.setup_dynamics()

    def setup_symbols(self):
        self.x_vec = []

        for state_name in self.env.B_STATES:
            state_name += "_{}".format(self.agent.name)
            if not state_name in self.env.FIXED_STATE_NAMES:
                state_name_sym = symbols(state_name, real=True)
                setattr(self, state_name + "_sym", state_name_sym)
                self.x_vec.append(state_name_sym)
            else:
                state_name_sym = 0.0
                setattr(self, state_name + "_sym", state_name_sym)

    def setup_dynamics(self):
        self.A = Matrix(
            [
                [0, 1],
                [0, 0],
            ]
        )
        self.B = Matrix(
            [
                [0, 0],
                [0, 0],
            ]
        )

        Delta_t = symbols("Delta_t", real=True, positive=True)

        self.F = (self.A * Delta_t).exp()
        self.G = self.B * 0

    def u(self, t):
        return np.array([])

    def get_sym(self, variable):
        return getattr(self, variable + "_" + self.agent.name + "_sym")


class Dynamics:
    def __init__(self, env):
        self.env = env

        self.setup_rovers()
        self.setup_beacons()
        self.setup_clocks()

        self.define_symbols()

        self.setup_distance_function()
        self.setup_rover_state_function()
        self.setup_time_function()
        self.define_propagation()

    def setup_rovers(self):
        self.rover_dict = {
            rover_name: Rover(self.env, self.env.agent_dict[rover_name])
            for rover_name in self.env.ROVER_NAMES
        }

    def setup_beacons(self):
        self.beacon_dict = {
            beacon_name: Beacon(self.env, self.env.agent_dict[beacon_name])
            for beacon_name in self.env.BEACON_NAMES
        }

    def setup_clocks(self):
        self.clock_dict = {
            agent_name: Clock(self.env, self.env.agent_dict[agent_name])
            for agent_name in self.env.AGENT_NAMES
        }

    def define_symbols(self):
        # State variable symbols
        self.x_vec = []

        for clock in self.clock_dict.keys():
            self.x_vec.extend(self.clock_dict[clock].x_vec)

        for rover in self.rover_dict.keys():
            self.x_vec.extend(self.rover_dict[rover].x_vec)

    def setup_distance_function(self):
        self.distance_target_functions = dict()
        self.los_target_functions = dict()

        for pair in self.env.PAIRS_DUPLEX:
            alpha_name = pair[0]
            beta_name = pair[1]

            # Distance
            distance_func = lambdify(
                self.x_vec,
                Matrix(
                    self.get_sym_position(alpha_name) - self.get_sym_position(beta_name)
                ).norm(),
            )
            self.distance_target_functions[pair] = distance_func

            # Line of Sight
            los_func = lambdify(
                self.x_vec,
                self.get_sym_position(alpha_name) - self.get_sym_position(beta_name),
            )
            self.los_target_functions[pair] = los_func

    def setup_rover_state_function(self):
        self.rover_state = {
            rover_name: lambdify(self.x_vec, self.rover_dict[rover_name].x_vec)
            for rover_name in self.rover_dict.keys()
        }

    def setup_time_function(self):
        self.Delta_functions = dict()

        for clock_name in self.clock_dict.keys():
            clock_params = [
                self.get_sym("b", clock_name) / self.env.c,
                self.get_sym("b_dot", clock_name) / self.env.c,
            ]
            Delta_function = lambdify(self.x_vec, clock_params)

            self.Delta_functions[clock_name] = Delta_function

        if self.env.centralized:
            self.Delta_functions["Z"] = self.Delta_functions[
                self.env.agent_centralized.agent_reference
            ]

    def get_sym_position(self, agent_name):
        if agent_name in self.beacon_dict.keys():
            return self.beacon_dict[agent_name].get_true_position()
        elif agent_name in self.rover_dict.keys():
            return self.rover_dict[agent_name].get_sym_position()

    def get_true_position(self, agent_name):
        if agent_name in self.beacon_dict.keys():
            return self.beacon_dict[agent_name].get_true_position()
        elif agent_name in self.rover_dict.keys():
            return self.rover_dict[agent_name].get_true_position()

    def get_true_velocity(self, agent_name):
        if agent_name in self.beacon_dict.keys():
            return self.beacon_dict[agent_name].get_true_velocity()
        elif agent_name in self.rover_dict.keys():
            return self.rover_dict[agent_name].get_true_velocity()

    def get_sym_velocity(self, agent_name):
        if agent_name in self.beacon_dict.keys():
            return self.beacon_dict[agent_name].get_true_velocity()
        elif agent_name in self.rover_dict.keys():
            return self.rover_dict[agent_name].get_sym_velocity()

    def get_true_state(self, agent_name):
        state = np.concatenate(
            [self.get_true_position(agent_name), self.get_true_velocity(agent_name)]
        )

        return state

    def get_rover_state_estimate(self, rover_name, x_hat):
        return self.rover_state[rover_name](*x_hat)

    def define_propagation(self):
        F = diag(
            *[
                self.clock_dict[clock_name].F
                for clock_name in self.env.agent_clocks_to_be_estimated
            ],
            *[self.rover_dict[rover_name].F for rover_name in self.env.ROVER_NAMES]
        )
        G = Matrix(
            np.zeros([len(self.x_vec), self.env.n_dim * len(self.env.ROVER_NAMES)])
        )
        if self.env.ROVER_NAMES:
            G_rover = diag(
                *[self.rover_dict[rover_name].G for rover_name in self.env.ROVER_NAMES]
            )
            G[-self.env.NUM_ROVER_STATES :, :] = G_rover

        Delta_t = symbols("Delta_t", real=True, positive=True)
        self.F = lambdify(Delta_t, F)
        self.G = lambdify(Delta_t, G)

    def u(self, t):
        if self.env.ROVER_NAMES:
            u = np.concatenate(
                [
                    self.rover_dict[rover_name].u(t)
                    for rover_name in self.env.ROVER_NAMES
                ]
            )
        else:
            u = np.empty(0)

        return u

    def distance_between_agents(self, alpha, beta, x_hat):
        distance = self.distance_target_functions[alpha + beta](*x_hat)

        return distance

    def distance_between_agents_true(self, alpha, beta):
        alpha_position = self.get_true_position(alpha)
        beta_position = self.get_true_position(beta)

        distance = sqrt(sum(square(alpha_position - beta_position)))

        return distance

    def los_between_agents(self, alpha, beta, x_hat):
        if alpha == "T":
            los = np.array(self.los_target_functions[beta](*x_hat))
        elif beta == "T":
            los = np.array(self.los_target_functions[alpha](*x_hat))
        else:
            los = self.get_sym_position(alpha) - self.get_sym_position(beta)

        return los

    def get_sym(self, variable, agent_name):
        if variable in self.env.B_STATES:
            return self.clock_dict[agent_name].get_sym(variable)

    def step(self, x, taus, t, h_time=True):
        if h_time:
            return np.array(self.evaluate_f_h_time(*x, *taus, t))
        else:
            return np.array(self.evaluate_f_true_time(*x, *taus, t))

    def step_x(self, x, tau, t_estimate):
        x_prediction = self.F(tau) @ x + self.G(tau) @ self.u(t_estimate)
        return x_prediction

    def step_P(self, P, tau):
        P_prediction = self.F(tau) @ P @ self.F(tau).T
        return P_prediction

    def get_clock_estimate_function(self):
        clock_estimate_dict = {}

        for agent_name in self.env.AGENT_NAMES:
            clock_params = [
                self.get_sym("b", agent_name) / self.env.c,
                self.get_sym("b_dot", agent_name) / self.env.c,
            ]
            clock_estimate_function = lambdify(self.x_vec, clock_params)

            clock_estimate_dict[agent_name] = clock_estimate_function

        return clock_estimate_dict

    def print_x(self, x):
        for value, var_name in zip(x, self.x_vec):
            print(str(var_name).ljust(15), value)

    def print_xP(self, x, P):
        for x_value, var_name, P_value in zip(x, self.x_vec, np.diag(P)):
            print(
                "{:<12} {:< 10.2e} +/- {:< 10.2e}".format(
                    str(var_name), x_value, P_value
                )
            )
