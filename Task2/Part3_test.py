import matplotlib.pyplot as plt
import numpy as np
from scipy.constants import mu_0, pi, epsilon_0
from Part2_functions import modal_decomposition, plot_start_graphic, create_MTLM, plot_complex, create_A_modal
from Part3_mtlm import MTLM

"""A.G. Part3 of Task2
Calculating a transmission line model for a long cable with load

depencies:
    numpy
    scipy
    matplotlib
"""
# set input
freq = 1e3                              # [Hz] excitation
eps_shell = 10*epsilon_0                # [F/m]
wire_l = 2e3                            # [m] wire length
u_in = np.array([100, 80, 60])

enable_plots = 0

# MTL matrices - input -------------------------------------------------------------------------------------------------
plot_start_graphic(freq, eps_shell, wire_l)
R_mat, L_mat, C_mat = create_MTLM(wire_l)

# modal decomposition --------------------------------------------------------------------------------------------------
Z_mat, Y_mat, Zmodal_d, Ymodal_d, Qi, Qu, Z_ch, beta = modal_decomposition(freq, R_mat, L_mat, C_mat)

# a) calc complex voltages ---------------------------------------------------------------------------------------------
print(str(' transmission along 2km cable ').center(100, '#'))
# modal propagation matrix A_m for given MTML
A_m = create_A_modal(Z_ch, beta, wire_l)

# Qui matrix -----------------------------------------------------------------------------------------------------------
zeros_3b3 = np.zeros((3, 3))
Qui = np.block([[Qu, zeros_3b3],
                [zeros_3b3, Qi]])
Qui_inv = np.linalg.inv(Qui)

# propagation matrix A -------------------------------------------------------------------------------------------------
A_mat = Qui@A_m@Qui_inv

# create input vector by concart. input u and i:
i_in = np.array([0.1313+0.7298j, 0.0682+0.4239j, 0.00508+0.1180j])  # (?)
# i_in = u_in / (Z_ch+1)
TML_in = np.block([u_in, i_in])

# calculate output -----------------------------------------------------------------------------------------------------
TML_out = A_mat@TML_in
u_out = TML_out[0:3]
print(" output voltage vector: ", u_out)
i_out = TML_out[3:6]
print(" output current vector: ", i_out)
if enable_plots:
    plot_complex(u_in, "input voltages on 3 wire coax")
    plot_complex(u_out, "output voltages after 2km cable")

# power of input -------------------------------------------------------------------------------------------------------
p_in = u_in @ np.conj(i_in)
print(" Real input power [W]:   ", p_in.real)
# power of output
p_out = u_out @ np.conj(i_out)
print(" Real output power [W]:  ", p_out.real)
print(" Power Loss [%]:         ", (p_in.real-p_out.real)*100/p_in.real)





