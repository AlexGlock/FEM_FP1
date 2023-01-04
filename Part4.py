# -*- coding: utf-8 -*-
from Part4_geometry import Mesh, ShapeFunction_N, load_mesh, create_shape_fkt
from Part4_knu import knu_for_elem, knu_for_mesh, j_grid_for_elem, j_grid_for_mesh
from Part1_analytic import define_problem

from matplotlib.tri import Triangulation
import matplotlib.pyplot as plt
import numpy as np
"""
A.G. Part4 of Task1

extensions:
    Part4_geometry.py     - loads and analyses gmesh files
    Part4_knu.py          - create reluctivity matrix
"""

# load and init mesh
cable_msh = load_mesh('wire.msh')
# get problem Parameters and generate shape FKT
model_params = define_problem()

# ----------------------------------------------------------------------------------------------------------------------
# ------------------------------ a) create shape_fkt of mesh elements --------------------------------------------------

shape_fkt_N = create_shape_fkt(cable_msh)

# ----------------------------------------------------------------------------------------------------------------------
# ------------------------------ b) reluctance for each element --------------------------------------------------------

rel_elem = cable_msh.reluctivity_in_elements                        # [m/H] : reluctivities per element

# ------------------------------ plot of reluctivities in regions ------------------------------------------------------
x = np.array(cable_msh.node[:, 0], ndmin=1).T
y = np.array(cable_msh.node[:, 1], ndmin=1).T
triang = Triangulation(x, y, cable_msh.elem_to_node)
fig = plt.figure()
ax = fig.add_subplot(111)
tpc = ax.tripcolor(triang, facecolors=rel_elem)
fig.colorbar(tpc)
ax.triplot(triang, color='black', lw=0.1)
ax.set_aspect('equal', 'box')
plt.title('reluctivities [m/H] in wire mesh')
plt.show()

# ----------------------------------------------------------------------------------------------------------------------
# ------------------------------ c) The knu matrix ---------------------------------------------------------------------

# print(knu_for_elem(cable_msh, shape_fkt_N, model_params, 0))
Knu = knu_for_mesh(cable_msh, shape_fkt_N, model_params)                          # [1/H] : reluctivities of grid

# ------------------------------ spy plot of knu matrix ----------------------------------------------------------------
plt.spy(Knu, markersize=1)
plt.title('structure of stiffness matrix Knu')
plt.show()

# ----------------------------------------------------------------------------------------------------------------------
# ------------------------------ d) create j_grid (elements and mesh) --------------------------------------------------

wire_current = 16
j_grid_elem = j_grid_for_elem(cable_msh, model_params)
j_grid = j_grid_for_mesh(cable_msh, shape_fkt_N, model_params)      # [A/m^2] : current densities of grid

# ------------------------------ plot of current densities in model ------------------------------------------
fig = plt.figure()
ax = fig.add_subplot(111)
tpc = ax.tripcolor(triang, facecolors=j_grid_elem)
fig.colorbar(tpc)
ax.triplot(triang, color='black', lw=0.1)
ax.set_aspect('equal', 'box')
plt.title('current densities [A/m^2] in mesh model')
plt.show()










