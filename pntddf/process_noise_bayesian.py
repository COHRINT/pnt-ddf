import pickle
from configparser import ConfigParser

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import progressbar
from bayes_opt import BayesianOptimization
from bpdb import set_trace
from numpy import abs, log, sqrt
from scipy.constants import c

from env import setup_env
from results import check_NEES_NIS

plt.style.use("seaborn-talk")

config_file = "process_noise_tuning.config"


def write_config(q_sigma_clock, q_sigma_target, transmission_window):
    config = ConfigParser()
    config.read(config_file)

    config["FILTER"]["Q_sigma_clock"] = q_sigma_clock
    config["FILTER"]["Q_sigma_target"] = q_sigma_target
    config["ENV"]["transmission_window"] = transmission_window
    config["ENV"]["MAX_TIME"] = str(float(transmission_window) * 5 * 100)

    with open(config_file, "w") as cf_handle:
        config.write(cf_handle)


# def run_sim(q_clock, q_target, transmission_window):
# write_config(q_clock, q_target, transmission_window)
def run_sim():

    N = 5

    epsilon_x = 0
    epsilon_z = 0

    for i in range(N):
        succeeded = False
        while not succeeded:
            env = setup_env(config_file)
            try:
                env.run(until=env.MAX_TIME)

                for agent in env.agents:
                    agent.cleanup()

                eps_x, eps_z = check_NEES_NIS(env)

                epsilon_x += eps_x
                epsilon_z += eps_z

                set_trace()

                succeeded = True
            except np.linalg.LinAlgError:
                for agent in env.agents:
                    agent.cleanup()
                continue

    epsilon_x_bar = epsilon_x / N
    epsilon_z_bar = epsilon_z / N

    J_NEES = abs(log(epsilon_x_bar))
    J_NIS = abs(log(epsilon_z_bar))

    print(f"{J_NEES=}")
    print(f"{J_NIS=}")

    R = -(J_NEES + J_NIS)

    return R


# pbounds = {"q": (-2, 2), "t_window": (-3, 0)}

# optimizer = BayesianOptimization(
# f=lambda q, t_window: run_sim(str(10 ** q), str(10 ** q), str(10 ** t_window)),
# pbounds=pbounds,
# random_state=1,
# )

# optimizer.maximize(
# init_points=5,
# n_iter=45,
# )

# print(optimizer.max)

R = run_sim()

set_trace()
