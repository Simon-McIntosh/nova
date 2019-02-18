import fipy
import numpy as np
from amigo.pyplot import plt


C = [0.9, 0.0]
R = [0.05, 5]
nx = 501
dx = np.diff(R)[0]/nx
origin = (R[0],)

mesh = {}
mesh['rz'] = fipy.CylindricalGrid1D(nx=nx, dx=dx, origin=origin)
mesh['xy'] = fipy.Grid1D(nx=nx, dx=dx) + origin

f = fipy.CellVariable(name='$f$', mesh=mesh['xy'])  # analytic solution
f.setValue(C[1]-(C[1]-C[0]) / np.log(R[1]/R[0]) * np.log(R[1]/mesh['xy'].x))


cxy = fipy.CellVariable(name='$c_{xy}$', mesh=mesh['xy'])
cxy.constrain(C[0], mesh['xy'].facesLeft)
cxy.constrain(C[1], mesh['xy'].facesRight)

crz = fipy.CellVariable(name='$c_{rz}$', mesh=mesh['rz'])
crz.constrain(C[0], mesh['rz'].facesLeft)
crz.constrain(C[1], mesh['rz'].facesRight)

rc = fipy.CellVariable(name='$r$', mesh=mesh['xy'])
rc.setValue(mesh['xy'].cellCenters)

eq = {}
eq['rz'] = fipy.DiffusionTerm(coeff=1)  # CylindricalGrid
eq['xy'] = fipy.DiffusionTerm(coeff=1)  # grid1D
eq['xy'] += fipy.ConvectionTerm(coeff=[[1]]*mesh['xy'].x**-1)
eq['xy'] += fipy.ImplicitSourceTerm(cxy*rc**-2)

ax = plt.subplots(1, 1)[1]
viewer = fipy.Viewer((f, cxy, crz), datamin=0, datamax=1, axes=ax)


eq['rz'].solve(var=crz)
residual = 1
while residual > 1e-2:
    residual = eq['xy'].sweep(var=cxy)
    print(residual)

    viewer.plot()

for line, label in zip(ax.get_lines(), ['f', 'rz', 'xy']):
    line.set_label(label)
ax.legend()
ax.set_title('')
plt.despine()


