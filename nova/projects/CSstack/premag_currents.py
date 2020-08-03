import numpy as np

from amigo.pyplot import plt
from nova.design.inverse import Inverse
from nova.electromagnetic.coilgeom import ITERcoilset
from nova.electromagnetic.machinedata import MachineData


pmag = Inverse()

"""
#pmag.load_coilset('ITER')


ITER = ITERcoilset(coils='pf', dCoil=0.2, n=2e3, limit=[3.5, 9, -5, 5],
                   read_txt=True)
cc = ITER.cc
cc.scenario_filename = -2
cc.scenario = 'IM'
cc.save_coilset('ITER')



r, xo, zo = 1.2, 5.4, 0
x, z = np.array([[r*np.cos(t), r*np.sin(t)]
                  for t in np.linspace(0, 2*np.pi, 20, endpoint=False)]).T

pmag.add_fix(xo+x, zo+z, 0, 'psi', 1, 2)
pmag.add_fix(xo, zo, 0, 'psi', 1, 2)

#pmag.Ic = {'PF6': 0}

pmag.colocate.solve_interaction()
pmag.save_coilset('ITER')
"""



"""
pmag.fix.value = pmag.colocate.Psi

pmag.fix_flux(4)

plt.set_aspect(1.1)
pmag.colocate.plot()

pmag.plot(subcoil=False)

pmag.grid.plot_flux()

'''
pmag.grid.generate_grid(limit=[3.5, 9, -5, 5])
pmag.save_coilset('ITER')
pmag.grid.plot_flux()
'''


machine = MachineData()
machine.load_data()
machine.plot_data(['firstwall', 'divertor'], color='k')
    
pmag.label_gaps()

#machine.load_models()
#machine.plot_models()
    
#machine.load_coilset()
#machine.plot()
"""
