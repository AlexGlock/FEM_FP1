from Part4_geometry import Mesh, ShapeFunction_N
from Part1_analytic import ProblemParams
from scipy.constants import mu_0, pi
from scipy.sparse import csr_matrix, csr_array, spmatrix
from scipy import sparse
import numpy as np
"""A.G. Part4 of Task1 - knu

This file contains functions to initialize the knu stiffness matrix and reluctivity/ current vectors

referenced in:
    Part4.py
"""


def knu_for_elem(msh: Mesh, shape_fkt: ShapeFunction_N, model: ProblemParams, elem_ind: int):
    """creates a 3x3 knu matrix for one triangular object in the model mesh.

    :param msh: Mesh Data object of Model-Geometry
    :param shape_fkt: Shape function object of Model-Geometry
    :param model: parameters of analytic model
    :param elem_ind: integer index of the element
    :return elem_knu: 3x3 local knu matrix of element with elem_ind
    """

    elem_rel = msh.reluctivity_in_elements[elem_ind]
    elem_b = shape_fkt.b[elem_ind, :]
    elem_c = shape_fkt.c[elem_ind, :]
    elem_S = shape_fkt.S[elem_ind]

    elem_knu = np.zeros((3, 3))

    for i in range(3):
        for j in range(3):
            elem_knu[i, j] = elem_rel * (elem_b[i] * elem_b[j] + elem_c[i] * elem_c[j]) / (4 * elem_S * model.z_length)

    return elem_knu


def knu_for_mesh(msh: Mesh, shape_fkt: ShapeFunction_N, model: ProblemParams):
    """This function cumputes the NxN Knu Matrix for a mesh of N elements.

    :param msh: Mesh object of model geometry
    :param shape_fkt: shape function of model geometry
    :param model: parameters of analytic model
    :return knu: NxN stiffness matrix
    """
    n = msh.num_elements * 9  # Amount of matrix entries
    index_rows = np.zeros(n, dtype='int')
    index_cols = np.zeros(n, dtype='int')
    elementwise_entries = np.zeros(n, dtype='int')

    # Looping over the elements for assigning Knu...
    for k in range(msh.num_elements):
        elem_nodes = msh.elem_to_node[k, :]

        triple_elem_nodes = np.array([elem_nodes, elem_nodes, elem_nodes])
        index_cols[k * 9:k * 9 + 9] = np.reshape(triple_elem_nodes.T, 9)
        index_rows[k * 9:k * 9 + 9] = np.reshape(triple_elem_nodes, 9)
        elementwise_entries[k * 9:k * 9 + 9] = np.reshape(knu_for_elem(msh, shape_fkt, model, k), 9)

    # Assembly of Knu
    index_rows = index_rows.T
    index_columns = index_cols.tolist()
    elementwise_entries = elementwise_entries.tolist()
    return sparse.csr_matrix((elementwise_entries, (index_rows, index_columns))) # [1/H] : circuit-reluctance matrix


def j_grid_for_elem(msh: Mesh, model: ProblemParams):
    """This function creates a current density vector for a coax-cabel model.

    :param msh: Mesh object of model geometry
    :param model: problem parameters according to analytic model
    :return j_grid: current density vector of N model elements
    """
    elem_group_names, elem_group_vector = msh.elem_physical_groups
    # calculate densities for wire (1) and shell (2)
    j_dens1 = model.J_0             # unit: Ampere per square meter
    j_dens2 = 0
    # set density according to physical group of element
    elem_group_vector = [j_dens1 if x == 1 else x for x in elem_group_vector]
    return [j_dens2 if x == 2 else x for x in elem_group_vector]


def j_grid_for_mesh(msh: Mesh, shape_fkt: ShapeFunction_N, params: ProblemParams) -> spmatrix:
    """This function creates a current density vector for the nodes of a coax-cabel model.

    :param msh: Mesh object of model geometry
    :param shape_fkt: shape funktion of model geometry
    :param params: problem parameters according to analytic model
    :return j_grid: current density (sparse) vector of N model nodes
    """
    m = msh.num_node          # Amount of nodes

    elem_surf = shape_fkt.S                                          # [m^2]   : surface of triangles
    elem_cur_density = j_grid_for_elem(msh, params)                  # [A/m^2]     : current densities in each element

    node_currents = elem_cur_density * np.array(shape_fkt.S)/3  # [A]     : localized grid current (nodes)
    node_currents = np.tile(node_currents, (3, 1)).transpose()       # inflate for nodal accumulation

    # for j, nodes in enumerate(msh.elem_to_node):
    #    idx[j * 3:j * 3 + 3] = nodes
    #    node_currents[j * 3:j * 3 + 3] = elem_currents[j]  # / 3

    curr_v = np.zeros(m)
    for k in range(0, 3):
        for i in range(0, msh.num_elements - 1):
            idx_node = msh.elem_to_node[i, k]
            curr_v[idx_node] += node_currents[i, k]

    return sparse.csr_matrix((curr_v, (np.arange(msh.num_node), np.zeros(msh.num_node))), shape=(msh.num_node, 1))


def grid_curr_for_elem(msh: Mesh, shape_fkt: ShapeFunction_N, j_grid: np.array):
    """This function creates a current vector for a coax-cabel model.

    :param msh: Mesh object of model geometry
    :param shape_fkt: shape function of model geometry
    :param j_grid: vector of
    :return grid_current: vector of current per element
    """
    # grid currents
    grid_current = np.zeros(msh.num_elements)
    elem_surf = shape_fkt.S
    # loop over elements ...
    for i in range(msh.num_elements):
        grid_current[i] = elem_surf[i] * j_grid[i]

    return grid_current
