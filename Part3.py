import gmsh
from Part3_visualize import get_grouped_points, plot_groups_3D


# initialize and load mesh
gmsh.initialize()
gmsh.open('wire.msh')

# get line nodes, dimension 1:
Xdata_L, Ydata_L, Zdata_L, groupnames_L = get_grouped_points(1)
# get surfaces nodes, dimension 2:
Xdata_S, Ydata_S, Zdata_S, groupnames_S = get_grouped_points(2)

# plot the nodes
plot_groups_3D(Xdata_L, Ydata_L, Zdata_L, groupnames_L, Xdata_S, Ydata_S, Zdata_S, groupnames_S)