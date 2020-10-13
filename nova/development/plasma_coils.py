import numpy as np
import shapely.geometry
import shapely.ops
import pygeos

from nova.electromagnetic.coilgeom import ITERcoilset
from nova.utilities.pyplot import plt

ITER = ITERcoilset(coils='pf', dCoil=0.5, dPlasma=0.5, n=1e3,
                   read_txt=True, limit=[3, 10, -6, 6])


ITER.filename = -1
ITER.scenario = 'SOF'

'''
ITER.add_coil(ITER.d2.vector['Rcur']+2, ITER.d2.vector['Zcur'], 3, 5,
              cross_section='ellipse', name='Plasma', plasma=True)
ITER.field.solve()
ITER.grid.solve()
ITER.scenario = 'SOF'
'''

# ITER.plasma.generate_grid()
# ITER.grid.generate_grid()

sep = ITER.data['separatrix']
#sep.loc[:, 'x'] -= 0.2
#sep.loc[:, 'z'] += -1.5
ITER.separatrix = ITER.data['separatrix']

ITER.update_field()

#ITER.grid.update_psi()

from scipy.interpolate import RectBivariateSpline as RBS
from scipy.optimize import minimize
interp = RBS(ITER.grid.x, ITER.grid.z, ITER.grid.B)

def fun(x):
    return interp.ev(*x)

x = minimize(fun, [ITER.coil.x[-1], ITER.coil.z[-1]-5]).x
print(x)
plt.plot(*x, 'o')

plt.set_aspect(0.9)
#ITER.plot(False, plasma=True)
ITER.plot(True, plasma=True)
ITER.grid.plot_flux()

#ITER.plot_data(['firstwall', 'divertor'])
#plt.plot(*ITER.data['divertor'].iloc[1:].values.T)


#ITER.plasma.plot()

