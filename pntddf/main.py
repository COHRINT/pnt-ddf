import numpy as np
import progressbar
from bpdb import set_trace

from pntddf.env import setup_env
from pntddf.results import (plot_b, plot_b_dot, plot_residuals,
                            plot_residuals_post_fusion,
                            plot_residuals_post_iteration,
                            plot_rover_state_errors, plot_test, plot_time,
                            plot_trajectory)

config_file = "../config/sim.config"
# config_file = "../config/1d.config"

env = setup_env(config_file)

print("Running Simulation")
increments = 100

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
    plot_residuals_post_fusion,
    plot_residuals_post_iteration,
    plot_time,
]

agents_to_plot = env.AGENT_NAMES
# agents_to_plot = []
if env.centralized:
    agents_to_plot += ["Z"]

counter = 0
with progressbar.ProgressBar(max_value=len(agents_to_plot) * len(plots)) as bar:
    for agent_name in [
        agent for agent in env.agent_dict.keys() if agent in agents_to_plot
    ]:
        for plot in plots:
            plot(env, agent_name)
            counter += 1
            bar.update(counter)

plot_test(env, env.AGENT_NAMES[0])
