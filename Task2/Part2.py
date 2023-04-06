import matplotlib.pyplot as plt
import numpy as np
from scipy.constants import pi, epsilon_0
from Part2_functions import plot_start_graphic, print_characteristic, create_MTLM, plot_complex

"""A.G. Part2 of Task2
This file creates a Transmission line Model of a 3 wire coax cable

depencies:
    numpy
    scipy
    matplotlib
"""


# set input
freq = 1e3                              # [Hz] Anregung
eps_shell = 10*epsilon_0                # [F/m]
wire_l = 1                              # [m] wire length

plot_start_graphic(freq, eps_shell, wire_l)

# MTL matrices - input -------------------------------------------------------------------------------------------------
R_mat, L_mat, C_mat = create_MTLM(wire_l)

# create impedance and admittance Mat.
# Z = jwL + R
Z_mat = 1j * 2 * pi * freq * L_mat + R_mat
# Y = jwC + G
Y_mat = 1j * 2 * pi * freq * C_mat + 0

# characteristic matrices
# CH1 = Y * Z       CH2 = Z * Y
CH1 = Y_mat @ Z_mat
CH2 = Z_mat @ Y_mat

# Eigenvalue decomposition
EW1, Qi = np.linalg.eig(CH1)          # along cable length
EW2, Qu = np.linalg.eig(CH2)          # in cross-section

# Ähnlichkeitstransformation / Basiswechsel
# Zm = Qu^(-1)*Z*Qi     Ym = Qi^(-1)*Y*Qu
Qi_inv = np.linalg.inv(Qi)
Qu_inv = np.linalg.inv(Qu)
Zmodal_d = np.diag(Qu_inv @ Z_mat @ Qi)
Ymodal_d = np.diag(Qi_inv @ Y_mat @ Qu)

# char. Impedance and attenuation of modes
Z_ch = np.sqrt(Zmodal_d / Ymodal_d)
beta = np.sqrt(Zmodal_d * Ymodal_d)

# a) modes and characteristics of propagation along the cable ----------------------------------------------------------
print_characteristic(Zmodal_d, beta, beta.imag, Z_ch)

# b) plot complex currents ---------------------------------------------------------------------------------------------
# columns of Qi mat = Eigenvectors
i_ev1 = Qi[:, 0]
i_ev2 = Qi[:, 1]
i_ev3 = Qi[:, 2]

fig, axs = plt.subplots(1, 3)
fig.suptitle('current propagation eigenvectors (Qi mat.)')
plt.subplot(1, 3, 1)
plot_complex(i_ev1, 'EV 1')
plt.subplot(1, 3, 2)
plot_complex(i_ev2, 'EV 2')
plt.subplot(1, 3, 3)
plot_complex(i_ev3, 'EV 3')

# columns of Qu mat = Eigenvectors
u_ev1 = Qu[:, 0]
u_ev2 = Qu[:, 1]
u_ev3 = Qu[:, 2]

fig, axs = plt.subplots(1, 3)
fig.suptitle('voltage propagation eigenvectors (Qu mat.)')
plt.subplot(1, 3, 1)
plot_complex(u_ev1, 'EV 1')
plt.subplot(1, 3, 2)
plot_complex(u_ev2, 'EV 2')
plt.subplot(1, 3, 3)
plot_complex(u_ev3, 'EV 3')

plt.show()
