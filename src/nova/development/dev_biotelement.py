import numpy as np

from nova.frame.coilset import CoilSet
import matplotlib.pyplot as plt

cs = CoilSet(biot_instances='grid', n=2e4)
cs.add_coil(3, 0, 0.25, 0.5, dCoil=0.05, part='PF')

cs.Ic = 15e6
cs.grid.generate_grid(expand=0.2)

cs.plot(True)
'''

rs, zs = cs.grid.source._rms_, cs.grid.source._z_  # source
drs, dzs = cs.grid.source._dx_, cs.grid.source._dz_  # source filament size
r, z = cs.grid.target._x_, cs.grid.target._z_  # target
nI = cs.grid.source.nS*cs.grid.target.nT

#dLs = np.linalg.norm([drs, dzs], axis=0)/2  # filament characteristic length


dL = np.array([(r-rs), (z-zs)])
dL_mag = np.linalg.norm(dL, axis=0)
index = np.isclose(dL_mag, 0)
dL_norm = np.zeros((2, nI))
dL_norm[0, index] = 1  # radial offset
dL_norm[:, ~index] = dL[:, ~index] / dL_mag[~index]

#df = dL_mag*drs / (2*abs(dL[0]))

dL_arg = np.arctan(abs(dL_norm[1]) / abs(dL_norm[0]))
c_arg = np.arctan(dzs / drs)

df = np.zeros(nI)
c_index = dL_arg < c_arg
df[c_index] = drs[c_index]/2 / np.cos(dL_arg[c_index])
df[~c_index] = dzs[~c_index]/2 / np.cos(np.pi/2 - dL_arg[~c_index])
factor = (1 - dL_mag/df)

exponent = np.zeros(nI)
exponent[c_index] = dL_arg[c_index]/c_arg[c_index]
exponent[~c_index] = (np.pi/2 - dL_arg[~c_index]) / (np.pi/2 - c_arg[~c_index])
print(np.min(exponent), np.max(exponent))

plt.contour(cs.grid.x2d, cs.grid.z2d, exponent.reshape(*cs.grid.n2d), 40)

'''

levels = cs.grid.plot_flux(levels=201)


plt.figure()
cs.plot(True)
#levels = cs.grid.plot_field()


x = (cs.grid.x2d, cs.grid.z2d)
cs.grid.filter_sigma = 0
Bx = -cs.grid.interpolate('Psi').ev(*x, dy=1) / (2*np.pi*cs.grid.x2d)
Bz = cs.grid.interpolate('Psi').ev(*x, dx=1) / (2*np.pi*cs.grid.x2d)
B = np.linalg.norm([Bx, Bz], axis=0)

levels = plt.contour(cs.grid.x2d, cs.grid.z2d, cs.grid.B, colors='C7',
                     levels=51).levels
plt.contour(cs.grid.x2d, cs.grid.z2d, B, colors='C3', levels=levels)


'''
cs = CoilSet(biot_instances='grid', n=2e3)
cs.add_coil(3, 0, 0.5, 0.5, dCoil=0.05, part='PF')

cs.Ic = 15e6
cs.grid.generate_grid(expand=1.5)
cs.grid.plot_field(levels=3)
#cs.grid.plot_flux(levels=31, color='C1')
'''
