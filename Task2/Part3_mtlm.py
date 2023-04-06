
from dataclasses import dataclass
from functools import cached_property
import matplotlib.pyplot as plt
import numpy as np

from Part2_functions import modal_decomposition, create_MTLM, create_A_modal

@dataclass
class MTLM:
    freq: float
    wire_len: float
    R_out: float
    u_in: np.array

    def __init__(self, freq, wire_len, R_out, u_in):
        # fill params on init
        self.freq = freq
        self.wire_len = wire_len
        self.u_in = u_in
        self.R_out = R_out
        self.R_mat, self.L_mat, self.C_mat = create_MTLM(wire_len)
        _, _, _, _, self.Qi, self.Qu, self.Z_ch, self.beta = \
            modal_decomposition(self.freq, self.R_mat, self.L_mat, self.C_mat)

        # self.i_in = np.array([0.1313 + 0.7298j, 0.0682 + 0.4239j, 0.00508 + 0.1180j])  # (?)

    def A(self):
        A_m = create_A_modal(self.Z_ch, self.beta, self.wire_len)
        # Qui matrix
        zeros_3b3 = np.zeros((3, 3))
        Qui = np.block([[self.Qu, zeros_3b3],
                        [zeros_3b3, self.Qi]])
        Qui_inv = np.linalg.inv(Qui)
        return Qui @ A_m @ Qui_inv

    def solve(self, A):
        """
        This function calculates the voltage/current output and the current input based on the voltage input v_in
        :param A: Modal transmission Matrix A
        :param R: Load resistance R
        :return: vectors u_in, i_in, u_out, i_out
        """
        # calculates input current according to MTML and currents
        zero_3b3 = np.zeros([3, 3])
        z_3 = np.array([0, 0, 0])
        # input vector
        v_i = np.concatenate([z_3, z_3, self.u_in, z_3])
        # sys matrix
        M_11 = A
        M_12 = -np.eye(6)
        M_21 = np.diag(np.concatenate([np.repeat(1, 3), np.repeat(0, 3)]))
        M_22 = np.block([[zero_3b3, zero_3b3], [np.eye(3), -np.diag([self.R_out, self.R_out, self.R_out])]])
        M = np.block([[M_11, M_12], [M_21, M_22]])
        # solve M * v_o = v_i
        v_o = np.linalg.solve(M, v_i)
        # u_in, i_in, u_out, i_out
        return v_o[0:3], v_o[3:6], v_o[6:9], v_o[9:12]

    def power(self, u_i, i_i, u_o, i_o):
        """
        calculates complex value of input /ouput power in Watts
        and real power loss percentage (rplp as %)
        :param u_i:
        :param i_i:
        :param u_o:
        :param i_o:
        :return: p_in, p_out, plp
        """
        p_in = np.dot(u_i, np.conj(i_i))
        p_out = np.dot(u_o, np.conj(i_o))

        rplp = abs(p_in.real-p_out.real)*100/p_in.real
        return p_in, p_out, rplp






