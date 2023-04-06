from scipy.constants import pi
from scipy.linalg import block_diag
import matplotlib.pyplot as plt
import numpy as np


def plot_start_graphic(freq, eps_shell, wire_l):
    print('Calculating the MTML model for 3 wire coax cable ...')
    print('')
    print('               0====== L ===== R ====================0')
    print('                                           |    |')
    print('   Input                                   C    G        Output')
    print('                                           |    |')
    print('               0=====================================0')
    print('')
    print(str(' Input ').center(100, '#'))
    print('wire Parameters:')
    print(' frequency:        ', freq, ' [Hz]')
    print(' epsilon shell:    ', eps_shell, '  [F/m]')
    print(' wire length:      ', wire_l, '  [m]')


def print_characteristic(modes, beta, phase_const, z_char):
    """

    :param modes: Array with all cable modes
    :param beta: Array with attenuations of each mode
    :param phase_const: Array with phase constant of each mode
    :param z_char: Array with characteistic impedance of each mode
    :return: -
    """
    np.set_printoptions(precision=5)
    print(str(' modal decomposition ').center(100, '#'))
    print('Modes along the cable:')
    print(' Zm:        ', modes)
    print(' beta:      ', beta)
    print(' phaseconst:', beta.imag)
    print(' Z_char:    ', z_char)
    print('')


def create_MTLM(wire_l):
    """
    creates the multiconductor transmission line model matrices of a 3 wire coax cable
    According to the matrices given in lecture.

    Parameters
    ----------
    wire_l : float
        length in [m]

    Returns
    -------
    R_mat, L_mat, C_mat : np.ndarray
        matrices of a 3 wire coax MTML of length wire_l
    """

    R = np.array(np.repeat(1e-3, 3))
    R_mat = np.diag(R) * 1
    L_mat = np.array(([5, 1, 1], [1, 5, 1], [1, 1, 5])) * 1e-6
    C_mat = np.array(([5, -1, -1], [-1, 5, -1], [-1, -1, 5])) * 1e-9

    return R_mat, L_mat, C_mat


def plot_complex(comp_array, title):

    X = [x.real for x in comp_array]
    Y = [x.imag for x in comp_array]

    plt.scatter(X, Y, color='red')
    for i in range(len(X)):
        plt.plot([0, X[i]], [0, Y[i]], '-r')
        plt.annotate('Wire ' + str(i+1), (X[i], Y[i]))

    plt.grid(True, linestyle='--')
    scale = max([max(X), max(Y), max([x * -1 for x in X]), max([x * -1 for x in Y])]) * 1.3
    plt.axis([-scale, scale, -scale, scale])
    plt.xlabel('Re{Z}', fontweight='bold')
    plt.ylabel('Im{Z}', fontweight='bold')
    plt.title(title)
    # plt.show()


def create_A_modal(Z_ch, beta, l):
    """

    :param Z_ch: diagonal vec. of characteristic impedance matrix
    :param beta: vec. of mode attenuations
    :param k: index of mode for which the A mat is calculated
    :param l: z-length for which the matrix is calculated
    :return:
    """
    def block_k(k):
        # Ansatz for one mode prop matrix given in lecture:
        bl = beta[k] * l
        block = np.array([[np.cosh(bl), -Z_ch[k] * np.sinh(bl)],
                         [-np.sinh(bl) / Z_ch[k], np.cosh(bl)]])
        return block

    def swap_cols(arr, frm, to):
        arr[:, [frm, to]] = arr[:, [to, frm]]

    def swap_rows(arr, frm, to):
        arr[[frm, to], :] = arr[[to, frm], :]

    zero_2 = np.zeros((2, 2))
    # matrix with alternating u, i per row
    b0 = block_k(0)
    b1 = block_k(1)
    b2 = block_k(2)

    # sort for u vec first and i vec second
    # build A_m from blocks
    #A_m = np.array(block_diag(b0, b1, b2))
    swap1 = [1, 3, 2]
    swap2 = [2, 4, 3]

    #for i in range(2):
    #    swap_cols(A_m, swap1[i], swap2[i])

    #for j in range(2):
    #    swap_rows(A_m, swap1[i], swap2[i])

    # hard coded version
    A_m = np.array([[b0[0, 0], 0, 0, b0[0, 1], 0, 0],
                    [0, b1[0, 0], 0, 0, b1[0, 1], 0],
                    [0, 0, b2[0, 0], 0, 0, b2[0, 1]],
                    [b0[1, 0], 0, 0, b0[1, 1], 0, 0],
                    [0, b1[1, 0], 0, 0, b1[1, 1], 0],
                    [0, 0, b2[1, 0], 0, 0, b2[1, 1]]])
    # print(A_m)
    return A_m


def modal_decomposition(freq, R_mat, L_mat, C_mat):
    """

    :param freq: excitation
    :param R_mat: resistance matrix
    :param L_mat: inductance matrix
    :param C_mat: capacity matrix
    :return: decomposition variables
    """

    # create impedance and admittance Mat.
    Z_mat = 1j * 2 * pi * freq * L_mat + R_mat
    Y_mat = 1j * 2 * pi * freq * C_mat + 0

    # characteristic matrices
    CH1 = Y_mat @ Z_mat
    CH2 = Z_mat @ Y_mat
    EW1, Qi = np.linalg.eig(CH1)  # along cable length
    EW2, Qu = np.linalg.eig(CH2)  # in cross-section

    # Eigenvalue decomposition
    # Zm = Qu^(-1)*Z*Qi     Ym = Qi^(-1)*Y*Qu
    Qi_inv = np.linalg.inv(Qi)
    Qu_inv = np.linalg.inv(Qu)
    Zmodal_d = np.diag(Qu_inv @ Z_mat @ Qi)
    Ymodal_d = np.diag(Qi_inv @ Y_mat @ Qu)

    # char. Impedance and attenuation of modes
    Z_ch = np.sqrt(Zmodal_d / Ymodal_d)
    beta = np.sqrt(Zmodal_d * Ymodal_d)

    print_characteristic(Zmodal_d, beta, beta.imag, Z_ch)

    return Z_mat, Y_mat, Zmodal_d, Ymodal_d, Qi, Qu, Z_ch, beta

