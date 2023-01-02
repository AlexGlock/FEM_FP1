# Imports
from dataclasses import dataclass
import numpy as np
import gmsh


@dataclass
class Mesh:

    """ Dataclass containing data of a gmsh model in matrices """

    @property
    def mat_nodes(self) -> np.ndarray:
        """ Function to create a matrix mat_nodes containing all coordinates of the nodes in the mesh msh """

        # Extrakt the data of all nodes in the mesh
        nodes_mesh = gmsh.model.mesh.get_nodes(dim=-1, tag=-1, includeBoundary=False, returnParametricCoord=True)

        # Create a matrix mat_nodes containing all coordinates of the nodes in the mesh
        num_nodes = int(len(nodes_mesh[0]))
        mat_nodes = np.zeros((num_nodes, 3))

        for i in range(0, num_nodes, 1):
            mat_nodes[i, 0] = nodes_mesh[1][3 * i]
            mat_nodes[i, 1] = nodes_mesh[1][3 * i + 1]
            mat_nodes[i, 2] = nodes_mesh[1][3 * i + 2]

        return mat_nodes

    @property
    def mat_tri(self) -> np.ndarray:

        """ Function to create the matrix mat_tri containing the indices of all nodes of a triangle from mat_nodes in msh"""

        # Extract the data of all triangles in the mesh
        tri_mesh = gmsh.model.mesh.getElementsByType(2, tag=-1, task=0, numTasks=1)

        # Creating a matrix connecting node indices in mat_nodes to triangles
        num_tri = int(len(tri_mesh[0]))
        mat_tri = np.zeros((num_tri, 3))

        # Assign the indices of the nodes to the triangles
        for i in range(0, num_tri, 1):
            mat_tri[i, 0] = tri_mesh[1][3 * i] - 1
            mat_tri[i, 1] = tri_mesh[1][3 * i + 1] - 1
            mat_tri[i, 2] = tri_mesh[1][3 * i + 2] - 1

        return mat_tri

    @property
    def tri_to_ph2d(self) -> np.ndarray:

        """ Function for creating a vector mapping which triangle belongs to which physical group in 2D """

        # Create mat_tri
        mat_tri = self.mat_tri

        # Create the "empty" vector
        tri_to_ph_2d = np.zeros(mat_tri.shape[0])

        # Extract all 2D physical groups and store their tags in l_ph_groups_2d
        l_ph_groups_2d = []
        for phgroup in gmsh.model.getPhysicalGroups(dim=2):
            l_ph_groups_2d.append(phgroup[1])

        # Assign the indices of all triangles in mat_triangles to a physical group
        for g in l_ph_groups_2d:

            # Create a set containing the tags of all nodes from physical group g (Decrease the value of every tag by one
            # because gmash starts counting from 1 and not zero)
            nodes_g = gmsh.model.mesh.getNodesForPhysicalGroup(2, g)[0]
            for i in range(0, len(nodes_g), 1):
                nodes_g[i] -= 1
            nodes_g = set(nodes_g)

            # Assign all correct triangles to the ph group g by comparing the nodes of the triangle to the nodes in g
            for i in range(0, mat_tri.shape[0], 1):

                setnodes = {mat_tri[i][0], mat_tri[i][1], mat_tri[i][2]}
                if setnodes.intersection(nodes_g) == setnodes:
                    tri_to_ph_2d[i] = g

        return tri_to_ph_2d

    @staticmethod
    def create():
        """ Creates an instance of a Mesh object """
        return Mesh()

