from Part4_geometry import Mesh, ShapeFunction_N, load_mesh, create_shape_fkt
from Part1_analytic import ProblemParams, A_z, define_problem, calc_energy_ind
from Part3_visualize import get_grouped_points, plot_redcrosses
from Part4_knu import knu_for_elem, knu_for_mesh, j_grid_for_elem, j_grid_for_mesh

from matplotlib.tri import Triangulation
import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse.linalg as la
"""
A.G. Part6 of Forschungspraxis1
boundary conditions and system solving
extensions:
    Part4_knu.py
    Part4_geometry.py
    Part3_visualize.py
    Part1_analytic.py
"""

# load and init mesh
cable_msh = load_mesh('wire.msh')
# get problem Parameters and generate shape Fkt
model_params = define_problem()
model_shape_fkt = create_shape_fkt(cable_msh)

# ----------------------------------------------------------------------------------------------------------------------
# ------------------------------ a) get boundary nodes with GND potential  ---------------------------------------------

index_constraint = cable_msh.boundary_nodes
# ------------------------------ plot as red crosses -------------------------------------------------------------------
Xdata_S, Ydata_S, Zdata_S, groupnames_S = get_grouped_points(2)
plot_redcrosses(Xdata_S, Ydata_S, Zdata_S, groupnames_S, index_constraint, cable_msh)

# ----------------------------------------------------------------------------------------------------------------------
# ------------------------------ b) degrees of freedom  ----------------------------------------------------------------

# every node index except the boundary ones
index_dof = np.setdiff1d(np.arange(cable_msh.num_node), index_constraint).tolist()

# ----------------------------------------------------------------------------------------------------------------------
# ------------------------------ c) apply boundary conditions & shrink system  -----------------------------------------

Knu_shrink = knu_for_mesh(cable_msh, model_shape_fkt, model_params)[index_dof, :]
Knu_shrink = Knu_shrink[:, index_dof]

j_grid_dof = j_grid_for_mesh(cable_msh, model_shape_fkt, model_params)[index_dof]
excitation = j_grid_dof

# ----------------------------------------------------------------------------------------------------------------------
# ------------------------------ d) solve reduced system with linalg  --------------------------------------------------

# problem: knu_shrink*A_shrink = J_excitation
A_shrink = la.spsolve(Knu_shrink, excitation)

# ----------------------------------------------------------------------------------------------------------------------
# ------------------------------ e) inflate system and visualize solution for vector potential A  ----------------------

A = np.zeros((cable_msh.num_node, 1))

A_shrink = np.array(A_shrink, ndmin=2).T
A[index_dof] = A_shrink
A[index_constraint] = np.zeros((len(index_constraint), 1))          # set known dirlichet boundary condition -> 0
A = A.reshape(len(A))

# ------------------------------ cross section plot  -------------------------------------------------------------------
x = np.array(cable_msh.node[:, 0], ndmin=1).T
y = np.array(cable_msh.node[:, 1], ndmin=1).T
triang = Triangulation(x, y, cable_msh.elem_to_node)
fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_aspect('equal', 'box')

surf = ax.tripcolor(triang, A, cmap='viridis')  # cmap=plt.cm.CMRmap)
fig.colorbar(surf, shrink=0.5, aspect=5)
plt.title('magnetic vector potential A in coax cross-section')
plt.show()
