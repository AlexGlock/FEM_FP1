import numpy as np

from Part6_solver import MagnetostaticSolver
from Part4_geometry import load_mesh, create_shape_fkt
from Part3_visualize import cross_section_plot
from Part1_analytic import define_problem, calc_energy_ind

"""
A.G. Part7 of Task1
Post processing
extensions:
    Part6_solver.py
    Part4_geometry.py
    Part3_visualize.py
    Part1_analytic.py
"""


# load and init mesh
cable_msh = load_mesh('wire.msh')
# get problem Parameters and generate shape Fkt
model_params = define_problem()
model_shape_fkt = create_shape_fkt(cable_msh)

# init solver
MSsolver = MagnetostaticSolver(cable_msh, model_params, model_shape_fkt)


# ----------------------------------------------------------------------------------------------------------------------
# ------------------------------ a) calc magnetic flux densities  ------------------------------------------------------
# vector pot. in z-direction only: A_z -> curl operator leads to B_x and B_y component

b_field = MSsolver.b
b_abs = np.linalg.norm(b_field, axis=1)     # [T]    : magnitude of the magnetic flux density

# ----------------------------------------------------------------------------------------------------------------------
# ------------------------------ b) plot magnitude of flux density -----------------------------------------------------

cross_section_plot(cable_msh, b_abs, 'magnetic flux density B in coax cross-section')

# ----------------------------------------------------------------------------------------------------------------------
# ------------------------------ c) calculate magnetic energy ----------------------------------------------------------

print('Magnetostatic Solver Validation:')
print('--------------------------------------------')
w_mag_fe = MSsolver.w_mag                                                              # [J] : magnetic energy
print('Magnetic energy according to FE-Solver [J]:')
print(w_mag_fe)

# ----------------------------------------------------------------------------------------------------------------------
# ------------------------------ d) integrated magnetic energy ---------------------------------------------------------

elem_rel = cable_msh.reluctivity_in_elements
w_mag_fe_integ = np.sum(1/2*elem_rel*b_abs**2*model_shape_fkt.S*model_params.z_length)  # [J] : magnetic energy (integ)
print('integrated Magnetic energy of FE-Solver [J]:')
print(w_mag_fe_integ)
print('--------------------------------------------')

# ----------------------------------------------------------------------------------------------------------------------
# ------------------------------ e) compare to analytic solution -------------------------------------------------------

w_magn_analytic, l_analytic = calc_energy_ind(model_params)                              # [J] : magnetic energy
print('Magnetic energy of analytic model [J]:')
print(w_magn_analytic)
