import dolfinx
import dolfinx.io
import dolfinx.plot
import numpy as np
import ufl
from mpi4py import MPI

try:
    import pyvista
except ModuleNotFoundError:
    print("pyvista is required for this demo")
    exit(0)

# If environment variable PYVISTA_OFF_SCREEN is set to true save a png
# otherwise create interactive plot
if pyvista.OFF_SCREEN:
    pyvista.start_xvfb(wait=0.1)

# Set some global options for all plots
transparent = False
figsize = 800
pyvista.rcParams["background"] = [0.5, 0.5, 0.5]


# Interpolate a simple scalar function in 3D
def int_u(x):
    return x[0] + 3 * x[1] + 5 * x[2]


mesh = dolfinx.UnitCubeMesh(
    MPI.COMM_WORLD, 4, 3, 5, cell_type=dolfinx.cpp.mesh.CellType.tetrahedron
)
V = dolfinx.FunctionSpace(mesh, ("CG", 1))
u = dolfinx.Function(V)
u.interpolate(int_u)

# Extract mesh data from DOLFINx (only plot cells owned by the
# processor) and create a pyvista UnstructuredGrid
num_cells = mesh.topology.index_map(mesh.topology.dim).size_local
cell_entities = np.arange(num_cells, dtype=np.int32)
pyvista_cells, cell_types = dolfinx.plot.create_vtk_topology(
    mesh, mesh.topology.dim, cell_entities
)
grid = pyvista.UnstructuredGrid(pyvista_cells, cell_types, mesh.geometry.x)

# Compute the function values at the vertices, this is equivalent to a
# P1 Lagrange interpolation, and can be directly attached to the Pyvista
# mesh. Discard complex value if running DOLFINx with complex PETSc as
# backend
vertex_values = u.compute_point_values()
if np.iscomplexobj(vertex_values):
    vertex_values = vertex_values.real

# Create point cloud of vertices, and add the vertex values to the cloud
grid.point_arrays["u"] = vertex_values
grid.set_active_scalars("u")

# Create a pyvista plotter which is used to visualize the output
plotter = pyvista.Plotter()
plotter.add_text(
    "Mesh and corresponding dof values",
    position="upper_edge",
    font_size=14,
    color="black",
)

# Some styling arguments for the colorbar
sargs = dict(
    height=0.6,
    width=0.1,
    vertical=True,
    position_x=0.825,
    position_y=0.2,
    fmt="%1.2e",
    title_font_size=40,
    color="black",
    label_font_size=25,
)

# Plot the mesh (as a wireframe) with the finite element function
# visualized as the point cloud
plotter.add_mesh(grid, style="wireframe", line_width=2, color="black")

# To be able to visualize the mesh and nodes at the same time, we have
# to copy the grid
plotter.add_mesh(
    grid.copy(),
    style="points",
    render_points_as_spheres=True,
    scalars=vertex_values,
    point_size=10,
)
plotter.set_position([1.5, 0.5, 4])

# Save as png if we are using a container with no rendering
if pyvista.OFF_SCREEN:
    plotter.screenshot(
        "3D_wireframe_with_nodes.png",
        transparent_background=transparent,
        window_size=[figsize, figsize],
    )
else:
    plotter.show()

# Create a new plotter, and plot the values as a surface over the mesh
plotter = pyvista.Plotter()
plotter.add_text(
    "Function values over the surface of a mesh",
    position="upper_edge",
    font_size=14,
    color="black",
)

# Define some styling arguments for a colorbar
sargs = dict(
    height=0.1,
    width=0.8,
    vertical=False,
    position_x=0.1,
    position_y=0.05,
    fmt="%1.2e",
    title_font_size=40,
    color="black",
    label_font_size=25,
)

# Adjust camera to show the entire mesh
plotter.set_position([-2, -2, 2.1])
plotter.set_focus([1, 1, -0.01])
plotter.set_viewup([0, 0, 1])

# Add mesh with edges
plotter.add_mesh(grid, show_edges=True, scalars="u", scalar_bar_args=sargs)
if pyvista.OFF_SCREEN:
    plotter.screenshot(
        "3D_function.png",
        transparent_background=transparent,
        window_size=[figsize, figsize],
    )
else:
    plotter.show()
