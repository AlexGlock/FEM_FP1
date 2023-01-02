from dataclasses import dataclass
from scipy.constants import mu_0, epsilon_0
from point2d import Point2D
import numpy as np
import gmsh
import math
"""A.G. Part4 of Task1 - geometry

This file contains functions and classes for the geometry handling of a gmesh file

referenced in:
    Part4.py
"""
# ----------------------------------------------------------------------------------------------------------------------
# ------------------------------------------- CLASSES ------------------------------------------------------------------


@dataclass
class Mesh:
    node_tag_data: np.ndarray
    node_data: np.ndarray
    elementTypes: np.ndarray
    element_tags: np.ndarray
    nodeTags_elements: np.ndarray

    # treat nodes
    @property
    def num_node(self):
        # number of nodes
        return int(len(self.node_data) / 3)

    @property
    def node(self):
        # nodes
        node = np.reshape(self.node_data, (self.num_node, 3))
        # coordinates of nodes. x-coordinate in first column
        # and y-coordinate in second column
        node = node[:, 0:2]
        return node

    @property
    def node_tag(self):
        # ID of nodes
        node_tag = self.node_tag_data - np.ones(len(self.node_tag_data))
        node_tag = node_tag.astype('int')
        np.put_along_axis(self.node, np.c_[node_tag, node_tag], self.node, axis=0)
        return node_tag

        # treat elements

    @property
    def ind_elements(self):
        # index of elements
        return np.where(self.elementTypes == 2)[0]

    @property
    def elements(self):
        # elements
        return np.array(self.nodeTags_elements[self.ind_elements[0]])

    @property
    def num_elements(self):
        # number of elements
        return int(len(self.elements) / 3)

    @property
    def elem_to_node(self):
        # Associate elements (triangles) and their respective nodes.
        # Connection between elements and nodes.
        # Each line contains the indices of the contained nodes
        # OUTPUT: Array of N x 3 NODES
        elem_to_node = np.reshape(self.elements, (self.num_elements, 3)) - np.ones(
            (self.num_elements, 1))
        elem_to_node = elem_to_node.astype('int')
        return elem_to_node

    @property
    def all_edges(self):
        # Contains all edges
        return np.r_[self.elem_to_node[:, [0, 1]],
                     self.elem_to_node[:, [1, 2]],
                     self.elem_to_node[:, [0, 2]]]

    @property
    def edge_to_node(self):
        # Associate edges and their respective nodes.
        return np.unique(np.sort(self.all_edges), axis=0)

    @property
    def elem_physical_groups(self):
        # get physical groups and their associated elements
        # return:
        #   group_names = list of Physical-Groups. Name Pos in List = physical group tag [1, 2, ... , n]
        #   elem_group_list = list of size n-Elements. group Tag of each Mesh element
        group_names = []
        elem_nodes = self.elem_to_node
        elem_group_list = [0]*len(elem_nodes)

        for i in range(len(gmsh.model.get_physical_groups(2))):
            tag = gmsh.model.get_physical_groups(2)[i][1]
            name = str(gmsh.model.get_physical_name(2, tag))
            Nodes, cords = gmsh.model.mesh.get_nodes_for_physical_group(2, tag)
            for node_set_ind in range(len(elem_nodes)):
                node_set = elem_nodes[node_set_ind, :]
                if set(node_set) <= set(Nodes-1):
                    elem_group_list[node_set_ind] = tag
            group_names.append(name)
        return group_names, elem_group_list

    @property
    def boundary_nodes(self):
        line_grps = gmsh.model.get_physical_groups(1)
        line_nodes = []
        for line in line_grps:
            line_tag = line[1]
            line_name = str(gmsh.model.get_physical_name(1, line_tag))
            line_node, line_cords = gmsh.model.mesh.getNodesForPhysicalGroup(1, line_tag)
            if line_name == 'GND':
                line_nodes.append(line_node-1)
        return line_nodes


    @property
    def reluctivity_in_elements(self):
        # creates the reluctivity vector for wire.geo or similar coax problems with 2 physical groups
        groups, group_list = self.elem_physical_groups
        # swap 1 for rel of area 1 and 2 for rel of area 2
        reluc1 = 1/mu_0
        reluc2 = 1/(5 * mu_0)
        group_list = [reluc1 if x == 1 else x for x in group_list]
        return np.array([reluc2 if x == 2 else x for x in group_list])


# Handling of shape function for a given geometry
@dataclass
class ShapeFunction_N:
    a: float
    b: float
    c: float
    S: float

    def __call__(self, x: float, y: float) -> float:
        """Evaluates this shape function at the point (x,y)."""
        return (self.a + self.b * x + self.c * y) / (2 * self.S)

    @staticmethod
    def of_points(p_i: Point2D, p_j: Point2D, p_k: Point2D):
        """Creates the shape function at the triangle with corner points p_i, p_j and p_k"""

        if len(p_i) != 2 or len(p_j) != 2 or len(p_k) != 2:
            raise Exception("Points must be 2 dimensional!")
        return ShapeFunction_N.of_cords(p_i[0], p_i[1], p_j[0], p_j[1], p_k[0], p_k[1])

    @staticmethod
    def of_cords(x_i: float, y_i: float, x_j: float, y_j: float, x_k: float, y_k: float):
        """Creates the shape function at the triangle (x_i, y_i), (x_j, y_j), (x_k, y_k)."""

        return ShapeFunction_N(
            a=x_j * y_k - x_k * y_j,
            b=y_j - y_k,
            c=x_k - x_j,
            S=triag_area((x_i, y_i), (x_j, y_j), (x_k, y_k))
        )


# ----------------------------------------------------------------------------------------------------------------------
# ------------------------------------------- FUNCTIONS ----------------------------------------------------------------


# helper function for area calc
def triag_area(a: Point2D, b: Point2D, c: Point2D):

    def distance(p1, p2):
        return math.hypot(p1[0]-p2[0], p1[1]-p2[1])

    side_a = distance(a, b)
    side_b = distance(b, c)
    side_c = distance(c, a)
    s = 0.5 * ( side_a + side_b + side_c)
    return math.sqrt(s * (s - side_a) * (s - side_b) * (s - side_c))


# geometry user functions
def load_mesh(mesh_name):
    # Initialize gmsh
    gmsh.initialize()
    # open msh
    gmsh.open(mesh_name)
    gmsh.option.setNumber('Mesh.MeshSizeFactor', 0.8)  # control grid size here.
    gmsh.option.setNumber('Mesh.MshFileVersion', 2.2)  # MATLAB compatible mesh file format
    gmsh.model.occ.synchronize()
    # gmsh.fltk.run()                                  # uncomment, if you want to inspect the geometry
    gmsh.model.mesh.generate(dim=2)  # 2D mesh
    # gmsh.fltk.run()                                  # uncomment, if you want to inspect the mesh
    # gmsh.write(model_name+".msh")                    # writes the mesh to a file

    node_tag, node, _ = gmsh.model.mesh.getNodes()
    elementTypes, element_tags, nodeTags_elements = gmsh.model.mesh.getElements()

    msh = Mesh(node_tag, node, elementTypes, element_tags, nodeTags_elements)
    return msh


def create_shape_fkt(msh):
    triangles = msh.elem_to_node
    nodes = msh.node

    a = np.zeros((len(triangles), 3))
    b = np.zeros((len(triangles), 3))
    c = np.zeros((len(triangles), 3))
    S = np.zeros(len(triangles))


    for triag_ind, triag in enumerate(triangles):
        # triangle Nodes
        N0 = nodes[triag[0]]
        N1 = nodes[triag[1]]
        N2 = nodes[triag[2]]
        tri_nodes = np.array([N0, N1, N2])
        S[triag_ind] = triag_area(N0, N1, N2)

        for i in range(0, 3, 1):
            j = (i + 1) % 3
            k = (i + 2) % 3
            # calc a
            a[triag_ind, i] = tri_nodes[j][0] * tri_nodes[k][1] - tri_nodes[k][0] * tri_nodes[j][1]
            # calc b
            b[triag_ind, i] = tri_nodes[j][1] - tri_nodes[k][1]
            # calc c
            c[triag_ind, i] = tri_nodes[k][0] - tri_nodes[j][0]


    shape_fkt = ShapeFunction_N(a, b, c, S)

    return shape_fkt



