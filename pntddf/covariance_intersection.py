import warnings

import numpy as np
from bpdb import set_trace
from numpy import trace
from numpy.linalg import cholesky, cond, det, inv
from scipy.optimize import LinearConstraint, minimize, minimize_scalar

from pntddf.information import Information, invert

warnings.filterwarnings("ignore", category=UserWarning)


def covariance_intersection(information_set, fast=False):
    N = len(information_set)

    if N == 1:
        fused_info = information_set[0]
    elif fast:
        fused_info = covariance_intersection_fast(information_set)
    elif N == 2:
        fused_info = covariance_intersection_2_info(information_set)
    else:
        fused_info = covariance_intersection_N_info(information_set)

    return fused_info


def covariance_intersection_fast(information_set):
    # Using inverse of trace so that larger info weighted more highly
    Y_traces = np.array([1 / trace(info.Y) for info in information_set])

    omegas = Y_traces / Y_traces.sum()

    Y = sum([omega * info.Y for omega, info in zip(omegas, information_set)])
    y = sum([omega * info.y for omega, info in zip(omegas, information_set)])

    return Information(y, Y)


def covariance_intersection_2_info(information_set):
    [info_alpha, info_beta] = information_set

    res = minimize_scalar(
        lambda omega: det(inv(omega * info_alpha.Y + (1 - omega) * info_beta.Y)),
        bounds=(0, 1),
        method="bounded",
        options={"disp": 0},
    )
    omega = res.x

    Y = omega * info_alpha.Y + (1 - omega) * info_beta.Y
    y = omega * info_alpha.y + (1 - omega) * info_beta.y

    return Information(y, Y)


def covariance_intersection_N_info(information_set):
    N = len(information_set)

    # Define Constraints
    A = np.vstack([np.eye(N), np.ones(N)])
    lower_bounds = np.append(np.zeros(N), 1)
    upper_bounds = np.append(np.ones(N), 1)
    omega_constraint = LinearConstraint(A, lower_bounds, upper_bounds)

    omegas_0 = np.ones(N) / N

    # This seems to make it work a lot better
    B_set = [cholesky(info.Y) for info in information_set]

    res = minimize(
        lambda omegas: -det(sum([omega * B for omega, B in zip(omegas, B_set)])),
        omegas_0,
        method="trust-constr",
        constraints=omega_constraint,
        options={"xtol": 1e-4, "gtol": 1e-4, "verbose": 0},
    )

    omegas = res.x

    Y = sum([omega * info.Y for omega, info in zip(omegas, information_set)])
    y = sum([omega * info.y for omega, info in zip(omegas, information_set)])

    return Information(y, Y)
