from Part4_geometry import load_mesh, create_shape_fkt
from Part1_analytic import define_problem, calc_energy_ind

from scipy.constants import pi
import matplotlib.pyplot as plt
import numpy as np
"""
A.G. Part10 of Forschungspraxis1
Transmission line Model
extensions:
    Part4_geometry.py
    Part1_analytic.py
"""


# load and init mesh
cable_msh = load_mesh('wire.msh')
# get problem Parameters and generate shape Fkt
model_params = define_problem()
model_shape_fkt = create_shape_fkt(cable_msh)

print('Transmission Line Model')
print('')
print('               0===[__L__]===[__R__]=================0')
print('                                           |    |')
print('   Input                                   C    G        Output')
print('                                           |    |')
print('               0=====================================0')
print('')

# ----------------------------------------------------------------------------------------------------------------------
# ------------------------------ a) calculate lengthwise params  -------------------------------------------------------
# I) inductance L

magn_energy, inductance_l = calc_energy_ind(model_params)                               # [H/m]: inductance

print('Inductance L of the cross section model [H/m]:')
print(inductance_l)

# II) resistance R -----------------------------------------------------------------------------------------------------

copper_res = 17.86e-3
wire_surface = pi*model_params.r_1**2
resistance_r = copper_res/wire_surface                                                  # [R/m]: resistance

print('Resistance R along the transmission line z-axis [R/m]:')
print(resistance_r)

# III) capacity C ------------------------------------------------------------------------------------------------------
# using analytic formular for coax: 2*pi*eps*l/ln(r2/r1)
# [C/m]: capacity
capacity_c = 2 * pi * model_params.eps_shell * model_params.z_length / np.log(model_params.r_2 / model_params.r_1)

print('Capacity C between inner and outer conductor [C/m]:')
print(capacity_c)

# IV) conductivity R ---------------------------------------------------------------------------------------------------
# the probelm model uses a perfect isolator as SHELL, therefore the conductivity must be equal to 0

print('Conductivity G between inner and outer conductor [G/m] or [1/(R*m)]:')
print(0)                                                                                # [G/m]: conductivity


# ----------------------------------------------------------------------------------------------------------------------
# ------------------------------ b) frequency plots  -------------------------------------------------------------------

f_list = np.logspace(0, 4.0, num=40)                                                    # [Hz]: logspaced frequencies
k_list = 2*pi*f_list*np.sqrt(model_params.mu_shell*model_params.eps_shell)              # [1/m]: wavecount vector

# I) char. impedance Z(w)  ---------------------------------------------------------------------------------------------
# L_w = R + j*w*L = R + j*2*pi*f*L    with      db scaling => 20*log10()
Z_w_list = resistance_r + 1j*2*pi*f_list*inductance_l

# II) wave_length lambda(f)  -------------------------------------------------------------------------------------------
# lambda = (2*pi)/k
lambda_list = (2*pi)/k_list

# III) phase velocity v_p(f)  ------------------------------------------------------------------------------------------
# v_p = w/k = (2*pi*f)/k
v_p = (2*pi*f_list)/k_list

plt.figure()
plt.title('characteristic impedance - Bode')
plt.xlabel("freqency [Hz]")
plt.ylabel("| Z_ch |")
plt.semilogx(f_list, 20*np.log10(abs(Z_w_list)))     # Bode impedance plot
plt.figure()
plt.title('wavelength across frequencies')
plt.xlabel("freqency [Hz]")
plt.ylabel("lambda [m]")
plt.semilogx(f_list, lambda_list)                    # wavelength plot
plt.figure()
plt.title('phase-velocity across frequencies')
plt.xlabel("freqency [Hz]")
plt.ylabel("v_p [m/s]")
plt.semilogx(f_list, v_p)                            # phase-velocity plot
plt.show()

# ----------------------------------------------------------------------------------------------------------------------
# ------------------------------ c) calculate system matrices for TM  --------------------------------------------------
freq = 1000         # [Hz]
# create impedance and admittance Mat == Dim 1 for Coax
# Z = jwL + R
Z = 1j * 2 * pi * freq * inductance_l + resistance_r
# Y = jwC + G
Y = 1j * 2 * pi * freq * capacity_c + 0

# characteristic matrices. For Coax identical scalar
# CH1 = Y * Z       CH2 = Z * Y
CH1 = Y * Z
CH2 = Z * Y

# Eigenvalue decomposition not necessary because value = EV = single mode
Qi = CH1          # along cable length
Qu = CH2          # in cross-section

# modal = time domain for simple coax
Z_modal = Z
Y_modal = Y

# char. Impedance and attenuation of mode
Z_ch = np.sqrt(Z_modal / Y_modal)
beta = np.sqrt(Z_modal * Y_modal)

# Propagation matrix A of dim 2x2 for given length l
l = 2       # [m]
A = np.array([[np.cosh(beta * l), -Z_ch * np.sinh(beta * l)],
              [- 1 / Z_ch * np.sinh(beta * l), np.cosh(beta * l)]])

# Admittance Matrix B of same dim wit 2 different values
B = np.array([[np.cosh(beta * l), - 1 / (Z_ch * np.sinh(beta * l))],
             [- Z_ch * 1 / np.sinh(beta * l), np.cosh(beta * l)]])
print('Progpagation Matrix for l:')
print(A)
print('Admittance Matrix for l:')
print(B)


