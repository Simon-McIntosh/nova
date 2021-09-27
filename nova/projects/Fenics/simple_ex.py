
import dolfinx
import numpy
import ufl
from mpi4py import MPI

mesh = dolfinx.UnitSquareMesh(MPI.COMM_WORLD, 8, 8, 
                              dolfinx.cpp.mesh.CellType.quadrilateral)

V = dolfinx.FunctionSpace(mesh, ("CG", 1))

uD = dolfinx.Function(V)
uD.interpolate(lambda x: 1 + x[0]**2 + 2 * x[1]**2)
uD.x.scatter_forward()

fdim = mesh.topology.dim - 1
# Create facet to cell connectivity required to determine boundary facets
mesh.topology.create_connectivity(fdim, mesh.topology.dim)
boundary_facets = numpy.where(numpy.array(dolfinx.cpp.mesh.compute_boundary_facets(mesh.topology)) == 1)[0]

boundary_dofs = dolfinx.fem.locate_dofs_topological(V, fdim, boundary_facets)
bc = dolfinx.DirichletBC(uD, boundary_dofs)

u = ufl.TrialFunction(V)
v = ufl.TestFunction(V)

f = dolfinx.Constant(mesh, -6)

a = ufl.dot(ufl.grad(u), ufl.grad(v)) * ufl.dx
L = f * v * ufl.dx


problem = dolfinx.fem.LinearProblem(a, L, bcs=[bc])
# petsc_options={"ksp_type": "preonly", "pc_type": "lu"}
uh = problem.solve()

'''
V2 = dolfinx.FunctionSpace(mesh, ("CG", 2))
uex = dolfinx.Function(V2)
uex.interpolate(lambda x: 1 + x[0]**2 + 2 * x[1]**2)
uex.x.scatter_forward()

L2_error = ufl.inner(uh - uex, uh - uex) * ufl.dx
error_L2 = numpy.sqrt(dolfinx.fem.assemble_scalar(L2_error))

u_vertex_values = uh.compute_point_values()
u_ex_vertex_values = uex.compute_point_values()
error_max = numpy.max(numpy.abs(u_vertex_values - u_ex_vertex_values))
print(f"Error_L2 : {error_L2:.2e}")
print(f"Error_max : {error_max:.2e}")
'''