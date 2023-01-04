import numpy as np
import matplotlib.pyplot as plt
from Part1_analytic import define_problem, calc_energy_ind, H_phi, A_z
from scipy.constants import pi, mu_0, epsilon_0
"""A.G. Part1 of Task1
This file plots the analytic solution of a ´magnetostatic coax cable problem

depencies:
    Part1_analytic
"""

# get Problem Constants
params = define_problem()
# init input and output
r_list = np.linspace(0, params.r_2, 50)

# get solution for all r
print('calculating the analytic solution ...')
H_list = H_phi(r_list, params)
A_list = A_z(r_list, params)
w_mag_analytic, l_analytic = calc_energy_ind(params)
print('magnetic energy (analytic) [J]:')
print(w_mag_analytic)
print('analytic model inductance [H]:')
print(l_analytic)


# plot solution for all
plt.plot(r_list, A_list)
plt.title('vector potential A in cross-section')
plt.xlabel('radius r in m', fontweight='bold')
plt.ylabel('| A |', fontweight='bold')
plt.show()

# plot solution for all
plt.plot(r_list, H_list)
plt.title('Magnetic Field H in cross-section')
plt.xlabel('radius r in m', fontweight='bold')
plt.ylabel('| H |', fontweight='bold')
plt.show()

