"""Build example sheild mesh."""
import gmsh
import io
import os


from dolfinx.cpp.io import perm_gmsh
from dolfinx.cpp.mesh import to_type
from dolfinx.mesh import create_mesh
from dolfinx.io import (
    extract_gmsh_geometry,
    extract_gmsh_topology_and_markers,
    ufl_mesh_from_gmsh,
)
from mpi4py import MPI
import numpy as np

from warnings import filterwarnings


from nova.definitions import root_dir

fname = os.path.join(root_dir, "input/geometry/ITER/sheild/EQ_RIB_SET_A_row37.stp")

gmsh.initialize()
gmsh.model.occ.importShapes(fname, format="step")
gmsh.model.occ.synchronize()

gmsh.option.setNumber("Mesh.MeshSizeMin", 10)
gmsh.option.setNumber("Mesh.MeshSizeMax", 100)
gmsh.model.mesh.generate(3)


gmsh.fltk.run()
"""
gdim = 3

filterwarnings("ignore")
if MPI.COMM_WORLD.rank == 0:
    # Get mesh geometry
    geometry_data = extract_gmsh_geometry(gmsh.model)
    # Get mesh topology for each element
    topology_data = extract_gmsh_topology_and_markers(gmsh.model)

if MPI.COMM_WORLD.rank == 0:
    # Extract the cell type and number of nodes per cell and broadcast
    # it to the other processors
    gmsh_cell_type = list(topology_data.keys())
    properties = gmsh.model.mesh.getElementProperties(gmsh_cell_type)
    name, dim, order, num_nodes, local_coords, _ = properties
    cells = topology_data[gmsh_cell_type]["topology"]
    cell_id, num_nodes = MPI.COMM_WORLD.bcast([gmsh_cell_type, num_nodes],
                                              root=0)
else:
    cell_id, num_nodes = MPI.COMM_WORLD.bcast([None, None], root=0)
    cells, geometry_data = np.empty([0, num_nodes]), np.empty([0, gdim])

# Permute topology data from MSH-ordering to dolfinx-ordering
ufl_domain = ufl_mesh_from_gmsh(cell_id, gdim)
gmsh_cell_perm = perm_gmsh(to_type(str(ufl_domain.ufl_cell())), num_nodes)
cells = cells[:, gmsh_cell_perm]

# Create distributed mesh
mesh = create_mesh(MPI.COMM_WORLD, cells, geometry_data[:, :gdim], ufl_domain)
"""
