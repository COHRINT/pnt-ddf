import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from bpdb import set_trace
from mpl_toolkits.axes_grid1 import make_axes_locatable
from numpy import abs, mean, sqrt, square

plt.style.use("seaborn-talk")

matplotlib.rcParams["figure.figsize"] = [10.0, 10.0]
matplotlib.rcParams["figure.constrained_layout.use"] = True
matplotlib.rcParams["agg.path.chunksize"] = 10000


def rms(data):
    return sqrt(mean(square(data)))


def get_state_df(env, agent_name):
    agent = env.agent_dict[agent_name]

    df_state = agent.estimator.get_state_log_df()

    return df_state


def plot_test(env, agent_name):
    agent = env.agent_dict[agent_name]

    df_state = agent.estimator.get_state_log_df()

    df_meas = agent.estimator.get_residuals_log_df()

    set_trace()


def check_NEES_NIS(env):
    epsilon_x = 0
    epsilon_z = 0

    for agent_index in range(env.NUM_AGENTS):
        agent = env.agents[agent_index]
        df_state = agent.estimator.get_state_log_df()
        df_meas = agent.estimator.get_residuals_log_df()

        epsilon_x += df_state.epsilon_x.median() / env.NUM_STATES
        epsilon_z += df_meas.epsilon_z.median() / len(agent.sensors.measurement_names)

    epsilon_x /= env.NUM_AGENTS
    epsilon_z /= env.NUM_AGENTS

    return epsilon_x, epsilon_z


def plot_b(env, agent_name):
    if env.NUM_B_STATES == 0:
        return

    agent = env.agent_dict[agent_name]

    df = agent.estimator.get_state_log_df()

    fig, axes = plt.subplots(
        env.NUM_B_STATES,
        figsize=(10, 3 * env.NUM_B_STATES),
        constrained_layout=True,
        squeeze=False,
    )
    axes = axes.ravel()

    fig.suptitle("Agent {} $b$ Estimates".format(agent.name), fontsize="xx-large")

    for index, agent_name in enumerate(env.agent_clocks_to_be_estimated):
        axes[index].plot(
            df.t,
            df["b_{}_error".format(agent_name)],
            marker=".",
            linestyle="None",
            color="k",
        )
        axes[index].plot(
            df.t,
            +2 * df["b_{}_sigma".format(agent_name)],
            linestyle="--",
            color="C0",
        )
        axes[index].plot(
            df.t,
            -2 * df["b_{}_sigma".format(agent_name)],
            linestyle="--",
            color="C0",
        )

        axes[index].set(
            xlabel="$t$ [ sec ]",
            ylabel="$b_{} - \hat{{b_{}}}$ [ m ]".format(agent_name, agent_name),
            ylim=[
                -5 * df["b_{}_sigma".format(agent_name)].iloc[-1],
                5 * df["b_{}_sigma".format(agent_name)].iloc[-1],
            ],
        )

    plt.savefig("images/b_error_{}.png".format(agent.name))
    plt.close()


def plot_b_dot(env, agent_name):
    if env.NUM_B_DOT_STATES == 0:
        return

    agent = env.agent_dict[agent_name]

    df = agent.estimator.get_state_log_df()

    fig, axes = plt.subplots(
        env.NUM_B_DOT_STATES,
        figsize=(10, 3 * env.NUM_B_DOT_STATES),
        constrained_layout=True,
        squeeze=False,
    )
    axes = axes.ravel()

    fig.suptitle(
        "Agent {} $\dot{{b}}$ Estimates".format(agent.name), fontsize="xx-large"
    )

    for index, agent_name in enumerate(env.agent_clocks_to_be_estimated):
        axes[index].plot(
            df.t,
            df["b_dot_{}_error".format(agent_name)],
            marker=".",
            linestyle="None",
            color="k",
        )
        axes[index].plot(
            df.t,
            +2 * df["b_dot_{}_sigma".format(agent_name)],
            linestyle="--",
            color="C0",
        )
        axes[index].plot(
            df.t,
            -2 * df["b_dot_{}_sigma".format(agent_name)],
            linestyle="--",
            color="C0",
        )

        axes[index].set(
            xlabel="$t$ [ sec ]",
            ylabel="$\dot{{b_{}}} - \hat{{\dot{{b_{}}}}}$ [ m / s ]".format(
                agent_name, agent_name
            ),
            ylim=[
                -5 * df["b_dot_{}_sigma".format(agent_name)].iloc[-1],
                5 * df["b_dot_{}_sigma".format(agent_name)].iloc[-1],
            ],
        )

    plt.savefig("images/b_dot_error_{}.png".format(agent.name))
    plt.close()


def plot_rover_state_errors(env, agent_name):
    agent = env.agent_dict[agent_name]

    df = agent.estimator.get_state_log_df()

    if df.filter(regex="[xy]_").empty:
        return

    fig, axes = plt.subplots(
        env.n_dim * 2,
        len(env.ROVER_NAMES),
        figsize=(10 * len(env.ROVER_NAMES), 3 * 2 * env.n_dim),
        constrained_layout=True,
        squeeze=False,
    )

    fig.suptitle(
        "Agent {} Rover State Estimates".format(agent.name), fontsize="xx-large"
    )

    units = ["[ m ]"] * env.n_dim + ["[ m / s ]"] * env.n_dim

    for i, rover in enumerate(env.ROVER_NAMES):
        state_names_latex = ["{}_{}".format(v, rover) for v in env.dim_names]
        state_names_latex += ["\dot{{{}}}_{}".format(v, rover) for v in env.dim_names]
        for s, state in enumerate(env.ROVER_STATES):
            axes[s, i].plot(
                df.t,
                df[state + "_{}_error".format(rover)],
                linestyle="None",
                color="k",
                marker=".",
            )
            axes[s, i].plot(
                df.t,
                +2 * df[state + "_{}_sigma".format(rover)],
                linestyle="--",
                color="C0",
            )
            axes[s, i].plot(
                df.t,
                -2 * df[state + "_{}_sigma".format(rover)],
                linestyle="--",
                color="C0",
            )
            state_latex = state_names_latex[s]
            axes[s, i].set(
                xlabel="$t$ [ sec ]",
                ylabel="${} - \hat{{{}}}$ {}".format(
                    state_latex, state_latex, units[s]
                ),
                ylim=[
                    -5 * df[state + "_{}_sigma".format(rover)].iloc[-1],
                    5 * df[state + "_{}_sigma".format(rover)].iloc[-1],
                ],
            )

    plt.savefig("images/rover_state_error_{}.png".format(agent.name))
    plt.close()


def plot_trajectory(env, agent_name, show_beacons=False):
    agent = env.agent_dict[agent_name]

    df = agent.estimator.get_state_log_df()

    if df.filter(regex="[xy]_").empty:
        return

    fig, ax = plt.subplots(figsize=(10, 10), constrained_layout=True)

    if show_beacons:
        plt.plot(
            np.array(
                [b[1].get_true_position()[0] for b in env.dynamics.beacon_dict.items()]
            ),
            np.array(
                [b[1].get_true_position()[1] for b in env.dynamics.beacon_dict.items()]
            ),
            marker=".",
            color="C2",
            markersize=10,
            linestyle="None",
            label="Beacon",
        )

    for rover in env.ROVER_NAMES:
        ax.plot(
            df["x_{}_true".format(rover)],
            df["y_{}_true".format(rover)],
            linestyle="--",
            color="k",
            label="{} True".format(rover),
        )
        ax.plot(
            df["x_{}".format(rover)],
            df["y_{}".format(rover)],
            linestyle="None",
            marker=".",
            label="{} Estimate".format(rover),
        )

    ax.set(
        xlabel="$x$ [ m ]",
        ylabel="$y$ [ m ]",
        xlim=[-120, 120],
        ylim=[-120, 120],
        title="Agent {} Trajectory Estimate".format(agent.name),
    )
    ax.legend()

    plt.savefig("images/trajectory_{}.png".format(agent.name))
    plt.close()


def plot_residuals(env, agent_name):
    agent = env.agent_dict[agent_name]

    meas_df = agent.estimator.get_residuals_log_df()

    fig, axes = plt.subplots(
        len(agent.sensors.measurement_names),
        figsize=(10, int(3 * len(agent.sensors.measurement_names))),
        constrained_layout=True,
        squeeze=False,
    )
    axes = axes.ravel()

    fig.suptitle(
        "Agent {} Measurement Residuals".format(agent.name), fontsize="xx-large"
    )

    for index, measurement_name in enumerate(agent.sensors.measurement_names):
        df = (
            meas_df[meas_df["{}_P_yy_sigma".format(measurement_name)].notna()]
            .dropna(axis=1)
            .copy()
        )
        sigma = df["{}_P_yy_sigma".format(measurement_name)]
        axes[index].plot(
            df.t,
            df["{}_residual_pre_fit".format(measurement_name)],
            marker=".",
            linestyle="None",
            color="C0",
        )
        axes[index].plot(
            df.t,
            df["{}_residual_post_fit".format(measurement_name)],
            marker=".",
            linestyle="None",
            color="C1",
        )
        axes[index].plot(
            df.t,
            +2 * sigma,
            linestyle="--",
            color="k",
        )
        axes[index].plot(
            df.t,
            -2 * sigma,
            linestyle="--",
            color="k",
        )

        axes[index].set(
            xlabel="$t$ [ s ]",
            ylabel="{} Residuals [ m ]".format(
                agent.sensors.measurement_names_latex[index]
            ),
            ylim=[
                -5 * sigma.iloc[-1],
                5 * sigma.iloc[-1],
            ],
        )

    plt.savefig("images/residuals_{}.png".format(agent.name))
    plt.close()


def plot_residuals_post_fusion(env, agent_name):
    agent = env.agent_dict[agent_name]

    df = agent.estimator.get_residuals_log_df()

    if df.filter(regex="fused").empty:
        return

    fig, axes = plt.subplots(
        len(agent.sensors.measurement_names),
        figsize=(10, int(3 * len(agent.sensors.measurement_names))),
        constrained_layout=True,
        squeeze=False,
    )
    axes = axes.ravel()

    fig.suptitle(
        "Agent {} Measurement Residuals Post Fusion".format(agent.name),
        fontsize="xx-large",
    )

    for index, measurement_name in enumerate(agent.sensors.measurement_names):
        sigma = df["{}_P_yy_sigma".format(measurement_name)]

        axes[index].plot(
            df.t,
            df["{}_residual_fused".format(measurement_name)],
            marker=".",
            linestyle="None",
            color="k",
        )
        axes[index].plot(
            df.t,
            +2 * sigma,
            linestyle="--",
            color="C0",
        )
        axes[index].plot(
            df.t,
            -2 * sigma,
            linestyle="--",
            color="C0",
        )

        axes[index].set(
            xlabel="$t$ [ s ]",
            ylabel="{} Residuals [ m ]".format(
                agent.sensors.measurement_names_latex[index]
            ),
            ylim=[
                -5 * sigma.iloc[-1],
                5 * sigma.iloc[-1],
            ],
        )

    plt.savefig("images/residuals_fused_{}.png".format(agent.name))
    plt.close()


def plot_residuals_post_iteration(env, agent_name):
    agent = env.agent_dict[agent_name]

    df = agent.estimator.get_residuals_log_df()

    if df.filter(regex="iteration").empty:
        return

    fig, axes = plt.subplots(
        len(agent.sensors.measurement_names),
        figsize=(10, int(3 * len(agent.sensors.measurement_names))),
        constrained_layout=True,
        squeeze=False,
    )
    axes = axes.ravel()

    fig.suptitle(
        "Agent {} Measurement Residuals Post Iteration".format(agent.name),
        fontsize="xx-large",
    )

    for index, measurement_name in enumerate(agent.sensors.measurement_names):
        sigma = df["{}_sigma".format(measurement_name)]

        axes[index].plot(
            df.t,
            df["{}_post_iteration_residual".format(measurement_name)],
            marker=".",
            linestyle="None",
            color="k",
        )
        axes[index].plot(
            df.t,
            +2 * sigma,
            linestyle="--",
            color="C0",
        )
        axes[index].plot(
            df.t,
            -2 * sigma,
            linestyle="--",
            color="C0",
        )

        axes[index].set(
            xlabel="$t$ [ s ]",
            ylabel="{} Residuals [ m ]".format(
                agent.sensors.measurement_names_latex[index]
            ),
            ylim=[
                -5 * sigma.iloc[-1],
                5 * sigma.iloc[-1],
            ],
        )

    plt.savefig("images/residuals_post_iteration_{}.png".format(agent.name))
    plt.close()


def plot_P(env, P):
    fig, ax = plt.subplots(figsize=(8, 8), constrained_layout=True)
    ax.invert_yaxis()

    im = ax.pcolor(P, edgecolor="black", linestyle=":", lw=1.5)
    fig.colorbar(im)

    ax.set(
        xticks=np.arange(env.NUM_STATES) + 0.5,
        xticklabels=env.STATE_NAMES_LATEX,
        yticks=np.arange(env.NUM_STATES) + 0.5,
        yticklabels=env.STATE_NAMES_LATEX,
    )

    plt.show()


def plot_time(env, agent_name):
    agent = env.agent_dict[agent_name]

    if agent.name == "Z":
        return

    df_state = agent.estimator.get_state_log_df()

    if df_state.filter(regex="b_{}_sigma".format(agent.name)).empty:
        return

    fig, ax = plt.subplots(figsize=(6, 6), constrained_layout=True)

    ax.plot(
        df_state.t,
        df_state.t - df_state.t_estimate,
        marker=".",
        linestyle="None",
        color="k",
    )
    ax.plot(
        df_state.t,
        +2 * df_state["b_{}_sigma".format(agent.name)] / env.c,
        linestyle="--",
        color="C0",
    )
    ax.plot(
        df_state.t,
        -2 * df_state["b_{}_sigma".format(agent.name)] / env.c,
        linestyle="--",
        color="C0",
    )
    ax.set(
        xlabel="$t$ [ s ]",
        ylabel="$t - \hat{t}$ [ s ]",
        title="Agent {} Time Estimate".format(agent.name),
        ylim=[
            -5 * df_state["b_{}_sigma".format(agent.name)].iloc[-1] / env.c,
            5 * df_state["b_{}_sigma".format(agent.name)].iloc[-1] / env.c,
        ],
    )

    plt.savefig("images/time_{}.png".format(agent.name))
    plt.close()
