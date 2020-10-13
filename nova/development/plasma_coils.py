import numpy as np
import shapely.geometry
import shapely.ops
import pygeos

from nova.electromagnetic.coilgeom import ITERcoilset
from nova.utilities.pyplot import plt

ITER = ITERcoilset(coils='pf vv trs dir', dCoil=0.25, dPlasma=0.15, n=5e3,
                   read_txt=False, limit=[3, 10, -6, 6])

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
sep.loc[:, 'x'] -= 0.2
sep.loc[:, 'z'] += -1.5
ITER.separatrix = sep#ITER.data['separatrix']

ITER.grid.update_psi()

plt.set_aspect(0.9)
#ITER.plot(False, plasma=True)
ITER.plot(True, plasma=True)
ITER.grid.plot_flux()

#ITER.plot_data(['firstwall', 'divertor'])
#plt.plot(*ITER.data['divertor'].iloc[1:].values.T)


#ITER.plasma.plot()
