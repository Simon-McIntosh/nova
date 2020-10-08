from nova.electromagnetic.coilgeom import ITERcoilset
from nova.utilities.pyplot import plt

ITER = ITERcoilset(coils='pf vv trs dir', dCoil=0.25, dPlasma=0.2, n=2e3,
                   read_txt=False, limit=[3, 10, -5, 5])

ITER.filename = -1
ITER.scenario = 'SOP'

'''
ITER.add_coil(ITER.d2.vector['Rcur']+2, ITER.d2.vector['Zcur'], 3, 5,
              cross_section='ellipse', name='Plasma', plasma=True)
ITER.field.solve()
ITER.grid.solve()
ITER.scenario = 'SOF'
'''

#ITER.plasma.generate_grid()
#ITER.grid.generate_grid()

plt.set_aspect(0.9)
ITER.plot(True, plasma=True, label='active')
ITER.grid.plot_flux()
# ITER.plot_data(['firstwall', 'divertor'])

#ITER.plasma.plot()