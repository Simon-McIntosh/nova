import numpy as np

from amigo.pyplot import plt
from nova.design.inverse import Inverse
from nova.electromagnetic.coilset import CoilSet
from nova.electromagnetic.coilgeom import ITERcoilset
from nova.electromagnetic.machinedata import MachineData


pmag = Inverse()
pmag.load_coilset('ITER')

'''
ITER = ITERcoilset(coils='pf', dCoil=0.2, n=2e3, limit=[3.5, 7.5, -3, 3],
                   read_txt=True)
cc = ITER.cc
cc.scenario_filename = -2
cc.scenario = 'IM'

pmag = Inverse()
pmag.coilset = cc.coilset

# add colocation circle
r, xo, zo = 1.2, 5.4, 0
x, z = np.array([[r*np.cos(t), r*np.sin(t)]
                  for t in np.linspace(0, 2*np.pi, 20, endpoint=False)]).T
nx, nz = x, z  # normalze

pmag.colocate.initialize_targets()
pmag.colocate.add_targets('Psi_bndry', xo+x, zo+z)
pmag.colocate.add_targets('Psi_bndry', xo, zo, 0, 1, d_dx=3, d_dz=2)  

pmag.colocate.solve_interaction()
pmag.save_coilset('ITER')
'''

#pmag.update_weight()


#print(pmag.colocate)

#pmag.colocate.targets.Psi = pmag.colocate.Psi

#pmag.fix_flux(4)



pmag.scenario_filename = -2
pmag.scenario = 'IM'

#pmag.coil.add_mpc(['PF2', 'PF3'], 0)

plt.set_aspect(1.1)
pmag.colocate.plot()
pmag.grid.plot_flux()

pmag.colocate.update()

pmag.coil.remove_mpc('CS1U')
pmag.coil.add_mpc(['PF3', 'PF4'], -2)


pmag.set_foreground()
pmag.set_background()
pmag.set_target()

pmag.solve()
print(np.linalg.norm(pmag.err))

pmag.plot(subcoil=False, current='A')
pmag.grid.plot_flux(color='C3')

"""
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