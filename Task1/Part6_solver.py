from Part4_geometry import Mesh, ShapeFunction_N, load_mesh, create_shape_fkt
from Part1_analytic import ProblemParams, A_z, define_problem, calc_energy_ind
from Part3_visualize import get_grouped_points, plot_redcrosses
from Part4_knu import knu_for_elem, knu_for_mesh, j_grid_for_elem, j_grid_for_mesh
from dataclasses import dataclass
from matplotlib.tri import Triangulation
import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse.linalg as la
"""
A.G. Part6_solver of Forschungspraxis1
boundary conditions and system solving
extensions:
    Part4_knu.py
    Part4_geometry.py
    Part3_visualize.py
    Part1_analytic.py
"""
@dataclass
class MagnetostaticSolver:
    mesh: Mesh
    model: ProblemParams
    shape_fkt: ShapeFunction_N


    @property
    def index_constraint(self):
        # index of all boundary nodes
        return self.mesh.boundary_nodes

    @property
    def index_dof(self):
        # index off all non boundary nodes
        return np.setdiff1d(np.arange(self.mesh.num_node), self.index_constraint).tolist()

    @property
    def knu(self):
        # stifness matrix knu of system
        return knu_for_mesh(self.mesh, self.shape_fkt, self.model)

    @property
    def j_grid(self):
        # j_grid vector of system
        return j_grid_for_mesh(self.mesh, self.shape_fkt, self.model)

    @property
    def solve_dir(self):
        Knu_shrink = (self.knu[self.index_dof, :])[:, self.index_dof]
        j_grid_dof = self.j_grid[self.index_dof]

        # problem: knu_shrink*A_shrink = J
        A_shrink = np.array(la.spsolve(Knu_shrink, j_grid_dof), ndmin=2).T

        A = np.zeros((self.mesh.num_node, 1))
        A[self.index_dof] = A_shrink
        A[self.index_constraint] = np.zeros((len(self.index_constraint), 1))
        return A.reshape(len(A))

    @property
    def b(self):
        a = self.solve_dir
        b_field = np.vstack([np.sum(self.shape_fkt.c * a[self.mesh.elem_to_node[:]]
                                / (2 * self.shape_fkt.S[:, None]), 1)
                         / self.model.z_length,
                         np.sum(self.shape_fkt.b * a[self.mesh.elem_to_node[:]]
                                / (2 * self.shape_fkt.S[:, None]), 1)
                         / self.model.z_length]).T
        return b_field

    @property
    def w_mag(self):
        # magnetic energy
        a = self.solve_dir
        return 1/2 * a @ self.knu @ a


