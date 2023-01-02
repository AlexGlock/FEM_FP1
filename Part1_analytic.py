from scipy.constants import pi, mu_0, epsilon_0
from dataclasses import dataclass
import numpy as np
"""A.G. Part1 of Task1

This file plots the analytic solution of a ´magnetostatic coax cable problem

depencies:
    NONE
"""


@dataclass
class ProblemParams:
    z_length: float
    I: float
    J_0: float
    r_1: float
    r_2: float
    mu_wire: float
    mu_shell: float
    eps_shell: float


def define_problem():
    # Problem Parameters according to Task1
    # all parameters in SI base units [m, A, ...]
    l_z = 300e-3                    # [m]: Depth of wire (lz)
    i_curr = 16                     # [A]   : applied current
    r_1 = 2e-3                      # [m]: inner radius (wire)
    r_2 = 3.5e-3                    # [m]: outer radius (shell)
    J_0 = i_curr / (np.pi * r_1 ** 2)    # [A/m] : current density in the wire
    mu_w = mu_0
    mu_s = 5 * mu_0                 # [H/m] : permeability of shell
    eps_s = 1 * epsilon_0

    problem_def = ProblemParams(z_length=l_z, I=i_curr, J_0=J_0, r_1=r_1, r_2=r_2, mu_shell=mu_s,
                  mu_wire=mu_w, eps_shell=eps_s)
    return problem_def


def A_z(r_list, problem_params):
    """
    Analytic solution of magnetic vector potential

    Parameters
    ----------
    r : np.ndarray
        radius in [m]
    problem_params : ProblemParams class
        containing all Problem Parameters

    Returns
    -------
    a_z : np.ndarray
        Magnetic vector potential in [Tm]
    """
    # problem definition
    r_1 = problem_params.r_1
    r_2 = problem_params.r_2
    I = problem_params.I
    mu_w = problem_params.mu_wire
    mu_s = problem_params.mu_shell

    def A2(r):
        # analytic expression for isolator
        return - I * mu_s / (2*np.pi) * np.log(r / r_2)

    def A1(r):
        # analytic expression for inner wire
        return -I / (2 * np.pi) * (mu_w / 2 * (r ** 2 - r_1 ** 2) / r_1 ** 2 + mu_s * np.log(r_1 / r_2))

    A_list = np.zeros(len(r_list))
    for i, r in enumerate(r_list):
        if r < problem_params.r_1:
            # get analytic expression - inner wire
            A_list[i] = A1(r)
        else:
            # get analytic expression - isolator
            A_list[i] = A2(r)
    return A_list


def H_phi(r_list, problem_params):
    """
    Analytic solution of Magnetic Field

    Parameters
    ----------
    r : np.ndarray
        radius in [m]
    problem_params : ProblemParams instance
        containg all Problem Parameter
    Returns
    -------
    h_phi : np.ndarray
        Magnetic field strength in [A/m]:
    """

    def H2(r, I):
        # analytic expression for isolator
        return I / (2 * pi * r)

    def H1(r, r_1, I):
        # analytic expression for inner wire
        return I * r / (2 * pi * r_1 ** 2)

    H_list = np.zeros(len(r_list))
    for i, r in enumerate(r_list):
        if r < problem_params.r_1:
            # get analytic expression - inner wire
            H_list[i] = H1(r, problem_params.r_1, problem_params.I)
        else:
            # get analytic expression - isolator
            H_list[i] = H2(r, problem_params.I)
    return H_list


def calc_energy_ind(problem):
    # Analytic Energy and Inductance in a coax b´cable model
    # Analytic Energy and Inductance
    w_magn_analytic = problem.I ** 2 * problem.z_length / (4 * np.pi) * (
                mu_0 / 4 + problem.mu_shell * np.log(problem.r_2 / problem.r_1))  # [J]   : magnetic energy (analytic)
    l_analytic = 2 * w_magn_analytic / problem.I ** 2                             # [H]   : inductance (analytic)

    return w_magn_analytic, l_analytic
