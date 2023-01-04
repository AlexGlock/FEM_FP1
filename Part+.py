from Part6_solver import MagnetostaticSolver
from Part4_geometry import load_mesh, create_shape_fkt
from Part3_visualize import cross_section_plot
from Part1_analytic import define_problem, calc_energy_ind

from scipy.constants import pi
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

# Transmission Line Model
#
#               0===[__L'__]===[__R'__]===============0
#                                           |    |
#   Input                                   C'   G'        Output
#                                           |    |
#               0=====================================0
#
print('Transmission Line Model')
print('')
print('               0===[__L__]===[__R__]=================0')
print('                                           |    |')
print('   Input                                   C    G        Output')
print('                                           |    |')
print('               0=====================================0')
print('')
# ----------------------------------------------------------------------------------------------------------------------
# ------------------------------ a) calculate inductance (lengthwise)  -------------------------------------------------

print('Inductance L of the cross section model [H/m]:')
magn_energy, inductance_l = calc_energy_ind(model_params)   # [H/m]: inductance
print(inductance_l)

# ----------------------------------------------------------------------------------------------------------------------
# ------------------------------ b) calculate resistance and admittance  -----------------------------------------------

copper_res = 17.86e-3
wire_surface = pi*model_params.r_1**2

print('Resistance R and corresponding conductance [R/m] bzw [m/R]:')
resistance_r = copper_res/wire_surface                      # [R/m]: resistance
conductance_g = 1/resistance_r                              # [m/R]: conductance
print(resistance_r)
print(conductance_g)

# ----------------------------------------------------------------------------------------------------------------------
# ------------------------------ b) calculate capacity  ----------------------------------------------------------------


