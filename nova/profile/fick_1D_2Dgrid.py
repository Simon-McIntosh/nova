import fipy
import numpy as np
from amigo.pyplot import plt
from scipy.interpolate import RectBivariateSpline


C = [0.9, 0.0]
R = [0.005, 5]
Z = [-2, 2]
nr = 2500
nz = 4

dr = np.diff(R)[0] / (nr-1)
dz = np.diff(R)[0] / (nz-1)
origin = ((R[0],), (-2,))

mesh = {}
mesh['rz'] = fipy.CylindricalGrid2D(nr=nr, dr=dr,
                                    nz=nz, dz=dz, origin=origin)
mesh['xy'] = fipy.Grid2D(nx=nr, dx=dr, ny=nz, dy=dz) + origin


f = fipy.CellVariable(name='$f$', mesh=mesh['xy'])  # analytic solution
f.setValue(C[1]-(C[1]-C[0]) / np.log(R[1]/R[0]) * np.log(R[1]/mesh['xy'].x))

cxy = fipy.CellVariable(name='$c_{xy}$', mesh=mesh['xy'])
cxy.constrain(C[0], mesh['xy'].facesLeft)
cxy.constrain(C[1], mesh['xy'].facesRight)

crz = fipy.CellVariable(name='$c_{rz}$', mesh=mesh['rz'])
crz.constrain(C[0], mesh['rz'].facesLeft)
crz.constrain(C[1], mesh['rz'].facesRight)

rc = fipy.CellVariable(name='$r$', mesh=mesh['xy'])
rc.setValue(mesh['xy'].cellCenters[0])

rconv = fipy.FaceVariable(name='$rconv$', mesh=mesh['xy'], rank=1)
rconv.setValue([mesh['xy'].faceCenters[0]**-1,
                np.zeros(mesh['xy'].numberOfFaces)])

eq = {}
eq['rz'] = fipy.DiffusionTerm()  # CylindricalGrid
eq['xy'] = fipy.DiffusionTerm()  # grid1D
eq['xy'] += fipy.ConvectionTerm(coeff=rconv)
eq['xy'] += fipy.ImplicitSourceTerm(rc**-2)

ax = plt.subplots(1, 3)[1]
viewer = {}
viewer['f'] = fipy.Viewer((f), datamin=0, datamax=1, axes=ax[0],
                          colorbar=None, cmap=plt.cm.inferno)
viewer['f'].plot()

eq['rz'].solve(var=crz)
viewer['rz'] = fipy.Viewer((crz), datamin=0, datamax=1, axes=ax[1],
                           colorbar=None, cmap=plt.cm.inferno)
viewer['rz'].plot()

residual = 1
while residual > 1e-4:
    residual = eq['xy'].sweep(var=cxy)
viewer['xz'] = fipy.Viewer((cxy), datamin=0, datamax=1, axes=ax[2],
                           colorbar=None, cmap=plt.cm.inferno)
viewer['xz'].plot()


nznr = (mesh['rz'].numberOfHorizontalRows-1,
        mesh['rz'].numberOfVerticalColumns-1)
r1d = mesh['rz'].x.value.reshape(nznr)[0, :]
z1d = mesh['rz'].y.value.reshape(nznr)[:, 0]

crz_interp = RectBivariateSpline(z1d, r1d, crz.value.reshape(nznr))
cxy_interp = RectBivariateSpline(z1d, r1d, cxy.value.reshape(nznr))
f_interp = RectBivariateSpline(z1d, r1d, f.value.reshape(nznr))

z_interp = 0
ax = plt.subplots(1, 1)[1]
ax.plot(r1d, crz_interp.ev(z_interp*np.ones(len(r1d)), r1d), label='rz')
ax.plot(r1d, cxy_interp.ev(z_interp*np.ones(len(r1d)), r1d), '-.', label='xy')
ax.plot(r1d, f_interp.ev(z_interp*np.ones(len(r1d)), r1d), '--',
        label='function')
ax.legend()
plt.despine()
ax.set_xlabel('$r$')
ax.set_ylabel('$c$')



