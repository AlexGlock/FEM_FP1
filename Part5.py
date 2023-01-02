from Part4_geometry import Mesh, ShapeFunction_N, load_mesh, create_shape_fkt
from Part1_analytic import ProblemParams, A_z, define_problem, calc_energy_ind
from Part4_knu import knu_for_elem, knu_for_mesh, j_grid_for_elem, j_grid_for_mesh

from matplotlib.tri import Triangulation
import matplotlib.pyplot as plt
import numpy as np
"""
A.G. Part5 of Task1
first Validation of the solver
extensions:
    Part4_geometry.py     - loads and analyses gmesh files
    Part4_knu.py          - create reluctivity matrix
"""

# load and init mesh
cable_msh = load_mesh('coax.geo')
# get problem Parameters and generate shape Fkt
model_params = define_problem()
model_shape_fkt = create_shape_fkt(cable_msh)

# ----------------------------------------------------------------------------------------------------------------------
# ------------------------------ a) calculate vector pot. analytic and numeric -----------------------------------------

# analytic node radius (r) is equal to sqrt(x^2 + y^2) in 2D plane
r_list = np.zeros(cable_msh.num_node)                           # [m]  : radius of nodes
for i, node_cords in enumerate(cable_msh.node):
    r_list[i] = np.sqrt(node_cords[0]**2 + node_cords[1]**2)
# calc analytic projection of vector potential onto nodes
A_analytic = model_params.z_length * A_z(r_list, model_params)  # [Tm]  : analytic projection of VP
# calc numeric solution for Knu matrix of system K*a = J :
knu = knu_for_mesh(cable_msh, model_shape_fkt, model_params)    # [1/H] : numeric reluctance matrix

# ----------------------------------------------------------------------------------------------------------------------
# ------------------------------ b) compare analytic energy to numeric solution ----------------------------------------

# calc energy and inductance based on analytic wire model:
w_mag_analytic, l_analytic = calc_energy_ind(model_params)      # [J]    : magn. energy of analytic solution
print('magnetic energy (analytic) [J]:')
print(w_mag_analytic)

# calc energy and inductance based on numeric solution W = 1/2 * vol_Int(A*J):
w_mag_num_test = 1/2*A_analytic@knu@A_analytic           # [J]    : magn. energy calculated with Knu matrix
print('magnetic energy (FE-Test) [J]:')
print(w_mag_num_test)







