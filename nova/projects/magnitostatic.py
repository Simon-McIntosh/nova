
import io



from fenics import *
import fenics as fe
import pyvista as pv


def boundary(_, on_boundary):
    return on_boundary

mesh = UnitCubeMesh(3, 3, 6)
NE = FiniteElement('N1curl', tetrahedron, 1)
LA = FiniteElement('Lagrange', tetrahedron, 1)
V = FunctionSpace(mesh, NE * LA)

(A, phi) = TrialFunctions(V)
(N, psi) = TestFunctions(V)

A_D = Expression(('-x[1]*x[1]', 'x[0]*x[0]', 'x[0]*x[0]'), element=NE)
B_D = Expression('2*x[0] + 2*x[1]', element=LA)

bc = [fenics.DirichletBC(V.sub(0), A_D, boundary),
      fenics.DirichletBC(V.sub(1), B_D, boundary)]

f = Constant((2, -2, 2))

a = inner(curl(A), curl(N)) * dx + inner(grad(phi), N) * dx \
  + inner(A, grad(psi)) * dx
b = inner(f, N) * dx

w = Function(V)
solve(a == b, w, bc)


(A_, _) = w.split()

with io.BytesIO() as buf:
    buf.write(r.content)
    buf.seek(0)
    
    
#B = project(curl(A_), FunctionSpace(mesh, 'Lagrange', 1))

#print(errornorm(A_D, A_, 'Hcurl'))
#print(errornorm(B_D, B, 'L2'))
plot(A_, title='A')
#plot(B, title='B = curl(A)')

#interactive()