import gmsh
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation
"""A.G. Part3 of Task1

This file loads a 2D-Model of a coaxial wire cross-section and visualizes the different physical groups in a Plot.

depencies:
    NONE
"""

# ----------------------------------------------------------------------------------------------------------------------
# ------------------------------------------- FUNCTIONS ----------------------------------------------------------------


def get_msh_points(dim, grptag):
    points, pointNodes = gmsh.model.mesh.get_nodes_for_physical_group(dim, grptag)
    Xdata=[]
    Ydata=[]
    Zdata=[]
    N=len(pointNodes)
    #print("Anzahl der gefundenen Punkte:"+str(N/3))
    for i in range(N):
        if i%3==0:
            Xdata.append(pointNodes[i])
        elif i%3==1:
            Ydata.append(pointNodes[i])
        else:
            Zdata.append(pointNodes[i])
    return Xdata,Ydata,Zdata


def get_grouped_points(dim):
    Xdata=[]
    Ydata=[]
    Zdata=[]
    groupnames=[]

    print(str(dim)+"D Objekte: "+str(gmsh.model.get_physical_groups(dim)))
    for i in range(len(gmsh.model.get_physical_groups(dim))):
        tag = gmsh.model.get_physical_groups(dim)[i][1]
        name = str(gmsh.model.get_physical_name(dim,tag))
        print(" -> Gruppe "+str(i)+' :'+str(name))
        groupnames.append(name)
        sub_Xdata, sub_Ydata, sub_Zdata = get_msh_points(dim, tag)
        Xdata.append(sub_Xdata)
        Ydata.append(sub_Ydata)
        Zdata.append(sub_Zdata)
    return Xdata, Ydata, Zdata, groupnames


def plot_groups_3D(Xdata_L,Ydata_L,Zdata_L, groupnames_L, Xdata_S,Ydata_S,Zdata_S, groupnames_S):
    # Plotting mesh as combination of surface(s) and line(s).
    ax = plt.figure().add_subplot(projection='3d')
    ax.set_title('Visualized Mesh')
    # Adding labels
    ax.set_xlabel('X-axis', fontweight='bold')
    ax.set_ylabel('Y-axis', fontweight='bold')
    ax.set_zlabel('Z-axis', fontweight='bold')
    for i in range(len(groupnames_S)):
        print("plotting surface mesh: " + groupnames_S[i])
        ax.plot_trisurf(Xdata_S[i], Ydata_S[i], Zdata_S[i], linewidth=0.2, antialiased=True, alpha=1/(i+1))
    for i in range(len(groupnames_L)):
        print("plotting closed circle: " + groupnames_L[i])
        # close the circles with overlapping
        Xdata_L[i].append(Xdata_L[i][0])
        Ydata_L[i].append(Ydata_L[i][0])
        Zdata_L[i].append(Zdata_L[i][0])
        ax.plot(Xdata_L[i], Ydata_L[i], Zdata_L[i], 'r-', linewidth=1.5, antialiased=True, )
    plt.show()


def plot_redcrosses(Xdata_S, Ydata_S, Zdata_S, groupnames_S, nodes, msh):
    # Plotting mesh as combination of surface(s) and line(s).
    ax = plt.figure().add_subplot(projection='3d')
    ax.set_title('Visualized Mesh with boundary nodes')
    # Adding labels
    ax.set_xlabel('X-axis', fontweight='bold')
    ax.set_ylabel('Y-axis', fontweight='bold')
    ax.set_zlabel('Z-axis', fontweight='bold')
    for i in range(len(groupnames_S)):
        print("plotting surface mesh: " + groupnames_S[i])
        ax.plot_trisurf(Xdata_S[i], Ydata_S[i], Zdata_S[i], linewidth=0.2, antialiased=True, alpha=1/(i+1))
    print("plotting GND boundary nodes ...")
    for n in nodes:

        node_x = msh.node[n, 0]
        node_y = msh.node[n, 1]
        ax.plot(node_x, node_y, marker="x", markersize=10, markeredgecolor="red", markerfacecolor="red")
    plt.show()


def cross_section_plot(msh, element_vals, title):
    x = np.array(msh.node[:, 0], ndmin=1).T
    y = np.array(msh.node[:, 1], ndmin=1).T
    triang = Triangulation(x, y, msh.elem_to_node)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    tpc = ax.tripcolor(triang, facecolors=element_vals)
    fig.colorbar(tpc)
    ax.triplot(triang, color='black', lw=0.1)
    ax.set_aspect('equal', 'box')
    plt.title(title)
    plt.show()











