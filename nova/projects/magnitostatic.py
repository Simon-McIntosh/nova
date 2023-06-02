from fenics import *
import fenics as fe

import matplotlib.pyplot as plt


def boundary(_, on_boundary):
    return on_boundary


mesh = UnitCubeMesh(16, 16, 16)
NE = FiniteElement("N1curl", triangle, 1)
LA = FiniteElement("Lagrange", triangle, 1)
V = FunctionSpace(mesh, NE * LA)

(A, phi) = TrialFunctions(V)
(N, psi) = TestFunctions(V)

A_D = Expression(("-x[1]*x[1]", "x[0]*x[0]", "x[2]"), element=NE)
phi_D = Expression("-2*x[0] - 2*x[1]", element=LA)

bc = [
    fe.DirichletBC(V.sub(0), A_D, boundary),
    fe.DirichletBC(V.sub(1), phi_D, boundary),
]

f = Constant((0, 0, 0))

a = inner(curl(A), curl(N)) * dx + inner(grad(phi), N) * dx + inner(A, grad(psi)) * dx
b = inner(f, N) * dx

w = Function(V)
solve(a == b, w, bc)


(A_, phi_) = w.split()

# with io.BytesIO() as buf:
#    buf.write(r.content)
#    buf.seek(0)

B = project(curl(A_), FunctionSpace(mesh, "Lagrange", 1))

norm = project(inner(A_, grad(phi_)), FunctionSpace(mesh, "Lagrange", 1))

plt.figure()
plot(norm, title="L2 norm")
# print(errornorm(A_D, A_, 'Hcurl'))
# print(errornorm(B_D, B, 'L2'))
ax = plt.subplots(2, 2)[1]
plt.sca(ax[0, 0])
plot(A_, title="A")
plt.sca(ax[0, 1])
plot(B, title="B = curl(A)")
plt.sca(ax[1, 0])
plot(A_D, mesh=mesh, title="A_D")
plt.sca(ax[1, 1])
plot(B_D, mesh=mesh, title="B_D")

# interactive()
