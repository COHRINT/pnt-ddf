from copy import copy

import numpy as np
import progressbar
from bpdb import set_trace

from pntddf.env import setup_env
from pntddf.results import (comm_savings, plot_b, plot_b_dot, plot_residuals,
                            plot_rover_state_errors, plot_test, plot_time,
                            plot_trajectory)

config_file = "../config/sim.config"
# config_file = "../config/1d.config"
# config_file = "../config/demo.config"

env = setup_env(config_file)

print("Running Simulation")
increments = 1000

with progressbar.ProgressBar(max_value=env.MAX_TIME) as bar:
    for t in np.linspace(env.MAX_TIME / increments, env.MAX_TIME, increments):
        bar.update(np.round(t))
        env.run(until=t)

print("Generating Plots")
plots = [
    plot_b,
    plot_b_dot,
    plot_rover_state_errors,
    plot_trajectory,
    plot_residuals,
    plot_time,
]

agents_to_plot = copy(env.agents)
if env.centralized:
    agents_to_plot += [env.agent_centralized]

counter = 0
with progressbar.ProgressBar(max_value=len(agents_to_plot) * len(plots)) as bar:
    for agent in agents_to_plot:
        for plot in plots:
            df_state = agent.estimator.get_state_log_df()
            df_meas = agent.estimator.get_residuals_log_df()
            plot(env, agent, df_state, df_meas)
            counter += 1
            bar.update(counter)

# comm_savings(env, env.agents)
plot_test(env, env.agents[0])
