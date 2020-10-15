import numpy as np

from nova.electromagnetic.coilgeom import ITERcoilset
from nova.utilities.pyplot import plt

ITER = ITERcoilset(coils='pf', dCoil=0.25, dPlasma=0.25, n=1e3,
                   read_txt=False, limit=[3, 10, -6, 6],
                   plasma_n=3e4)

ITER.filename = -1
ITER.scenario = 'EOF'

sep = ITER.data['separatrix']
#sep.loc[:, 'x'] -= 0.2
#sep.loc[:, 'z'] += 2.5

ITER.separatrix = ITER.data['separatrix']


ITER.update_field()

# ITER.grid.update_psi()

"""
from scipy.interpolate import RectBivariateSpline as RBS
from scipy.optimize import minimize
interp = RBS(ITER.grid.x, ITER.grid.z, ITER.grid.B)

def minloc():
    np.argmin(ITER.subcoil.z[ITER.ionize_index])

def fun(x):
    return interp.ev(*x)

x = minimize(fun, [ITER.coil.x[-1], ITER.coil.z[-1]-5]).x
print(x)
"""

ITER.Ip = -20e6

plt.set_aspect(0.9)
ITER.plot(False, plasma=True, current='AT')
#ITER.grid.plot_flux()

ITER.plasmagrid.plot_flux()

#plt.plot(*x, 'o')

#ITER.plot_data(['firstwall', 'divertor'])
#plt.plot(*ITER.data['divertor'].iloc[1:].values.T)


# ITER.plasmagrid.plot()

