import fipy
from amigo.pyplot import plt

L = 10.0
nx = 4
dx = L / nx
print(dx)
origin = (6,)

mesh = {}
mesh["xy"] = fipy.Grid1D(nx=nx, dx=dx) + origin
mesh["rz"] = fipy.CylindricalGrid1D(nx=nx, dx=dx, origin=origin)

for m in mesh:
    print(m, "vol", mesh[m].cellVolumes)
    print(m, "centre", mesh[m].cellCenters.value)
"""
c0 = fipy.numerix.zeros(nx, 'd')
c0[40:60] = 1.0

ax = plt.subplots(1, 1)[1]
var, eq, viewer = {}, {}, {}
for m in mesh:
    var[m] = fipy.CellVariable(name="$c_f$", mesh=mesh[m], value=c0)
    var[m].constrain(0.0, mesh[m].facesLeft)
    var[m].constrain(0.3, mesh[m].facesRight)

    eq[m] = fipy.DiffusionTerm(1.0, var=var[m])
    eq[m] -= fipy.ExponentialConvectionTerm((-0.8,),var=var[m])

    viewer[m] = fipy.Viewer(vars=(var[m]), datamax=1.0, datamin=-0.05, axes=ax)
    viewer[m].plot()


for m in mesh:
    eq[m].solve()
    viewer[m].plot()
"""
