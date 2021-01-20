from ast import literal_eval
from configparser import ConfigParser
from itertools import chain, combinations

import numpy as np
import simpy
from bpdb import set_trace
from scipy.constants import c

from agent import Agent
from centralized import Agent_Centralized
from dynamics import Dynamics


def setup_env(config_file):
    env = simpy.Environment()
    env.c = c

    # Config
    config = ConfigParser()
    config.read(config_file)

    # Max time
    env.MAX_TIME = config.getfloat("ENV", "MAX_TIME")

    # Filter config
    env.filter_config = config["FILTER"]

    # N DIM
    env.n_dim = config.getint("ENV", "n_dim")

    # x0, P0
    env.x0_config = config["x0"]
    env.P0_config = config["P0"]

    # Agents
    env.AGENT_NAMES = literal_eval(config.get("ENV", "AGENT_NAMES"))
    env.NUM_AGENTS = len(env.AGENT_NAMES)

    # Agent Config
    env.agent_configs = {
        agent_name: config[agent_name] for agent_name in env.AGENT_NAMES
    }
    env.BEACON_NAMES = [
        name
        for name in env.AGENT_NAMES
        if env.agent_configs[name].get("form") == "beacon"
    ]
    env.ROVER_NAMES = [
        name
        for name in env.AGENT_NAMES
        if env.agent_configs[name].get("form") == "rover"
    ]

    # Transmission
    env.TRANSMISSION_WINDOW = config.getfloat("ENV", "TRANSMISSION_WINDOW")

    # Pairs
    env.PAIRS = ["".join(pair) for pair in combinations(env.AGENT_NAMES, 2)]
    env.PAIRS_DUPLEX = env.PAIRS + [pair[::-1] for pair in env.PAIRS]
    env.NUM_PAIRS = len(env.PAIRS)

    # Which agents will have their clock parameters estimated
    agent_clocks_to_be_estimated = [
        agent
        for agent in env.AGENT_NAMES
        if env.agent_configs[agent].getboolean("estimate_clock")
    ]

    env.agent_clocks_to_be_estimated = agent_clocks_to_be_estimated

    # b states
    env.B_STATES = ["b", "b_dot"]
    b_states = ["b_{}".format(agent) for agent in agent_clocks_to_be_estimated]
    b_dot_states = ["b_dot_{}".format(agent) for agent in agent_clocks_to_be_estimated]

    # Rover states
    env.dim_names = ["x", "y", "z"][: env.n_dim]
    env.ROVER_STATES = env.dim_names + [v + "_dot" for v in env.dim_names]

    rover_states = []
    rover_states_latex = []
    for rover_name in env.ROVER_NAMES:
        rover_states.extend([v + "_" + rover_name for v in env.ROVER_STATES])
        rover_states_latex.extend(
            ["${}_{}$".format(v, rover_name) for v in env.dim_names]
        )
        rover_states_latex.extend(
            ["$\dot{{{}}}_{}$".format(v, rover_name) for v in env.dim_names]
        )

    state_names = list(chain(*zip(b_states, b_dot_states))) + rover_states

    env.STATE_NAMES = state_names
    env.NUM_B_STATES = len(b_states)
    env.NUM_B_DOT_STATES = len(b_dot_states)
    env.NUM_ROVER_STATES = len(rover_states)
    env.NUM_STATES = len(env.STATE_NAMES)

    # Latex names
    b_states_latex = ["$b_{}$".format(agent) for agent in agent_clocks_to_be_estimated]
    b_dot_states_latex = [
        "$\dot{{b}}_{}$".format(agent) for agent in agent_clocks_to_be_estimated
    ]

    state_names_latex = (
        list(chain(*zip(b_states_latex, b_dot_states_latex))) + rover_states_latex
    )

    env.STATE_NAMES_LATEX = state_names_latex

    # Fixed states
    agent_clocks_fixed = [
        agent
        for agent in env.AGENT_NAMES
        if not env.agent_configs[agent].getboolean("estimate_clock")
    ]

    b_fixed = [0] * len(agent_clocks_fixed)
    b_dot_fixed = [0] * len(agent_clocks_fixed)

    b_fixed_names = ["b_{}".format(agent) for agent in agent_clocks_fixed]
    b_dot_fixed_names = ["b_dot_{}".format(agent) for agent in agent_clocks_fixed]

    env.FIXED_STATES = b_fixed + b_dot_fixed
    env.FIXED_STATE_NAMES = b_fixed_names + b_dot_fixed_names

    # x0, P0
    env.x0, env.P0 = get_x0_P0(env, config)
    assert len(env.x0) == env.NUM_STATES

    # Create agents
    env.agents = [Agent(env, name) for name in env.AGENT_NAMES]

    env.agent_dict = {name: agent for name, agent in zip(env.AGENT_NAMES, env.agents)}

    # Centralized
    env.centralized = config.getboolean("ENV", "centralized")
    env.config_centralized = config["Z"]
    env.agent_centralized = Agent_Centralized(env, "Z")

    # Dynamics
    env.dynamics = Dynamics(env)

    for agent in env.agents:
        agent.init()

    # Centralized
    if env.centralized:
        env.agent_centralized.init()
        env.agent_dict["Z"] = env.agent_centralized

    return env


def get_x0_P0(env, config):
    # x0
    b_0 = config.getfloat("x0", "b")
    b_dot_0 = config.getfloat("x0", "b_dot")

    rover_0 = [config.getfloat("x0", "{}_hat".format(v)) for v in env.dim_names] + [
        config.getfloat("x0", "{}_dot_hat".format(v)) for v in env.dim_names
    ]

    x_0 = config.getfloat("x0", "x_hat")
    y_0 = config.getfloat("x0", "y_hat")
    x_dot_0 = config.getfloat("x0", "x_dot_hat")
    y_dot_0 = config.getfloat("x0", "y_dot_hat")

    x0 = list(
        chain(
            *zip(
                [b_0] * env.NUM_B_STATES,
                [b_dot_0] * env.NUM_B_DOT_STATES,
            )
        )
    )
    for rover in env.ROVER_NAMES:
        x0 += rover_0

    # P0
    sigma_b = config.getfloat("P0", "sigma_b")
    sigma_b_dot = config.getfloat("P0", "sigma_b_dot")

    sigma_rover = [
        config.getfloat("P0", "sigma_{}".format(v)) for v in env.dim_names
    ] + [config.getfloat("P0", "sigma_{}_dot".format(v)) for v in env.dim_names]

    sigmas = list(
        chain(
            *zip(
                [sigma_b ** 2] * env.NUM_B_STATES,
                [sigma_b_dot ** 2] * env.NUM_B_DOT_STATES,
            )
        )
    )

    for rover in env.ROVER_NAMES:
        sigmas += list(map(lambda x: x ** 2, sigma_rover))

    P0 = np.diag(sigmas)

    return x0, P0
