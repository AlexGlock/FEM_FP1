import gmsh
import sys
import numpy as np
from scipy.constants import pi, mu_0, epsilon_0
"""A.G. Part2 of Forschungspraxis1

This file creates a gmesh 2D-Model of a coaxial wire cross-section and saves it as wire.msh

depencies:
    NONE
"""

# initialize
gmsh.initialize()

# all floating point units in mm:
lc = 1e-3

# circles 1=inner cond edge, 2=outer isolator edge with curves
circle1 = gmsh.model.occ.addCircle(0, 0, 0, 2)
curve1 = gmsh.model.occ.addCurveLoop([circle1])
circle2 = gmsh.model.occ.addCircle(0, 0, 0, 3.5)
curve2 = gmsh.model.occ.addCurveLoop([circle2])

# circle surfaces 1= inner cond. , 2=isolator
surface1 = gmsh.model.occ.add_plane_surface(([curve1]))
surface2 = gmsh.model.occ.add_plane_surface([curve2, curve1])

# set number of elements for each surface
gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 25)

# from Gmsh model.
gmsh.model.occ.synchronize()

# Create the relevant Gmsh data structures
cond1 = gmsh.model.addPhysicalGroup(2, [circle1, curve1, surface1])
iso = gmsh.model.addPhysicalGroup(2, [surface2])
cond2 = gmsh.model.addPhysicalGroup(1, [circle2, curve2])
gmsh.model.setPhysicalName(2, cond1, "WIRE")
gmsh.model.setPhysicalName(2, iso, "SHELL")
gmsh.model.setPhysicalName(1, cond2, "GND")

# Generate mesh:
gmsh.model.mesh.generate()

# Write mesh data:
gmsh.write("wire.msh")

# Creates  graphical user interface
if 'close' not in sys.argv:
    gmsh.fltk.run()

# finalize the Gmsh API
gmsh.finalize()
print(f'created a coax cable mesh file with the name wire.msh')

