from nova.electromagnetic.coilgeom import ITERcoilset
from nova.utilities.pyplot import plt

ITER = ITERcoilset(coils='pf vv trs dir', dCoil=0.2, dPlasma=0.1, n=2e3, 
                   limit=[4, 8.5, -3, 3], read_txt=False)

ITER.filename = -1
ITER.scenario = 'SOF'

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
ITER.plot(True, plasma=True)
ITER.grid.plot_flux()
ITER.plot_data(['firstwall', 'divertor'])

#ITER.plasma.plot()