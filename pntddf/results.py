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


def get_state_df(env, agent):
    df_state = agent.estimator.get_state_log_df()

    return df_state


def plot_test(env, agent):
    df_state = agent.estimator.get_state_log_df()
    df_meas = agent.estimator.get_residuals_log_df()

    if env.centralized:
        df_state_Z = env.agent_centralized.estimator.get_state_log_df()
        df_meas_Z = env.agent_centralized.estimator.get_residuals_log_df()

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


def plot_b(env, agent):
    if env.NUM_B_STATES == 0:
        return

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
            markersize=7,
            marker=".",
            linestyle="None",
            color="k",
        )

        in_bounds = (
            df["b_{}_error".format(agent_name)].abs()
            < 2 * df["b_{}_sigma".format(agent_name)]
        ).sum() / df["b_{}_error".format(agent_name)].shape[0]

        axes[index].fill_between(
            df.t,
            +2 * df["b_{}_sigma".format(agent_name)],
            -2 * df["b_{}_sigma".format(agent_name)],
            color="C0",
            alpha=0.2,
            label="{:.2f}$\in \pm 2\sigma$".format(in_bounds),
        )

        axes[index].set(
            xlabel="$t$ [ sec ]",
            ylabel="$b_{} - \hat{{b_{}}}$ [ m ]".format(agent_name, agent_name),
            xlim=[0, env.MAX_TIME],
            ylim=[
                -5 * df["b_{}_sigma".format(agent_name)].median(),
                5 * df["b_{}_sigma".format(agent_name)].median(),
            ],
        )
        axes[index].legend()

    plt.savefig("images/b_error_{}.png".format(agent.name))
    plt.close()


def plot_b_dot(env, agent):
    if env.NUM_B_DOT_STATES == 0:
        return

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

        in_bounds = (
            df["b_dot_{}_error".format(agent_name)].abs()
            < 2 * df["b_dot_{}_sigma".format(agent_name)]
        ).sum() / df["b_dot_{}_error".format(agent_name)].shape[0]

        axes[index].fill_between(
            df.t,
            +2 * df["b_dot_{}_sigma".format(agent_name)],
            -2 * df["b_dot_{}_sigma".format(agent_name)],
            color="C0",
            alpha=0.2,
            label="{:.2f}$\in \pm 2\sigma$".format(in_bounds),
        )

        axes[index].legend()
        axes[index].set(
            xlabel="$t$ [ sec ]",
            ylabel="$\dot{{b_{}}} - \hat{{\dot{{b_{}}}}}$ [ m / s ]".format(
                agent_name, agent_name
            ),
            xlim=[0, env.MAX_TIME],
            ylim=[
                -5 * df["b_dot_{}_sigma".format(agent_name)].median(),
                5 * df["b_dot_{}_sigma".format(agent_name)].median(),
            ],
        )

    plt.savefig("images/b_dot_error_{}.png".format(agent.name))
    plt.close()


def plot_rover_state_errors(env, agent):
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
            in_bounds = (
                df[state + "_{}_error".format(rover)].abs()
                < 2 * df[state + "_{}_sigma".format(rover)]
            ).sum() / df[state + "_{}_error".format(rover)].shape[0]

            axes[s, i].fill_between(
                df.t,
                +2 * df[state + "_{}_sigma".format(rover)],
                -2 * df[state + "_{}_sigma".format(rover)],
                color="C0",
                alpha=0.2,
                label="{:.2f}$\in \pm 2\sigma$".format(in_bounds),
            )
            axes[s, i].legend()
            state_latex = state_names_latex[s]
            axes[s, i].set(
                xlabel="$t$ [ sec ]",
                ylabel="${} - \hat{{{}}}$ {}".format(
                    state_latex, state_latex, units[s]
                ),
                xlim=[0, env.MAX_TIME],
                ylim=[
                    -5 * df[state + "_{}_sigma".format(rover)].median(),
                    5 * df[state + "_{}_sigma".format(rover)].median(),
                ],
            )

    plt.savefig("images/rover_state_error_{}.png".format(agent.name))
    plt.close()


def plot_trajectory(env, agent, show_beacons=False):
    if env.n_dim != 2:
        return

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


def plot_residuals(env, agent):
    df_meas = agent.estimator.get_residuals_log_df()

    measurement_names = df_meas.name.unique()
    measurement_names.sort()

    fig, axes = plt.subplots(
        len(measurement_names),
        figsize=(10, int(3 * len(measurement_names))),
        constrained_layout=True,
        squeeze=False,
    )
    axes = axes.ravel()

    fig.suptitle(
        "Agent {} Measurement Residuals".format(agent.name), fontsize="xx-large"
    )

    for index, measurement_name in enumerate(measurement_names):
        df = df_meas[df_meas.name == measurement_name]
        if df.empty:
            continue
        for meas_type, color in zip(["local", "explicit"], ["C0", "C1"]):
            df_ = df[df[meas_type]]
            axes[index].plot(
                df_.t, df_.r, marker=".", ls="None", color=color, label=meas_type
            )

        axes[index].fill_between(
            df.t,
            +2 * df.P_yy_sigma,
            -2 * df.P_yy_sigma,
            color="C0",
            alpha=0.2,
        )

        axes[index].legend()
        axes[index].set(
            xlabel="$t$ [ s ]",
            ylabel="{} Residuals [ m ]".format(df.latex_name.iloc[0]),
            xlim=[0, env.MAX_TIME],
            ylim=[
                -5 * df.P_yy_sigma.median(),
                5 * df.P_yy_sigma.median(),
            ],
        )

    plt.savefig("images/residuals_{}.png".format(agent.name))
    plt.close()


def plot_time(env, agent):
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
    ax.fill_between(
        df_state.t,
        +2 * df_state["b_{}_sigma".format(agent.name)] / env.c,
        -2 * df_state["b_{}_sigma".format(agent.name)] / env.c,
        color="C0",
        alpha=0.2,
    )
    ax.set(
        xlabel="$t$ [ s ]",
        ylabel="$t - \hat{t}$ [ s ]",
        title="Agent {} Time Estimate".format(agent.name),
        xlim=[0, env.MAX_TIME],
        ylim=[
            -5 * df_state["b_{}_sigma".format(agent.name)].median() / env.c,
            5 * df_state["b_{}_sigma".format(agent.name)].median() / env.c,
        ],
    )

    plt.savefig("images/time_{}.png".format(agent.name))
    plt.close()
