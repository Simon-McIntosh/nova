import numpy as np

from amigo.pyplot import plt
from nova.design.inverse import Inverse
from nova.electromagnetic.coilset import CoilSet
from nova.electromagnetic.coilgeom import ITERcoilset
from nova.electromagnetic.machinedata import MachineData

build_coilset = False

pmag = Inverse()

if build_coilset:
    pmag.coilset = ITERcoilset(coils='pf vv', dCoil=0.2, n=5e3, 
                               limit=[3.5, 8, -2.5, 2.5], 
                               read_txt=True).coilset
    pmag.scenario_filename = -2
    pmag.scenario = 'IM'
    pmag.add_colocation_circle(5.7, 0, 1.6, N=30)
    pmag.save_coilset('ITER')
else:
    pmag.load_coilset('ITER')
    
pmag.scenario_filename = -2
pmag.scenario = 'IM'
pmag.colocate.update_targets()

#pmag.coil.add_mpc(['PF2', 'PF3'], 0)

plt.set_aspect(1.1)
pmag.colocate.plot()
pmag.grid.plot_flux()

#pmag.coil.remove_mpc('CS1U')
#pmag.coil.add_mpc(['PF3', 'PF4'], -2)
#pmag.colocate.targets.value += 10

pmag.set_foreground()
pmag.set_background()
pmag.set_target()

pmag.add_limit(ICS3L=15)
#pmag.add_limit(IPF6=20)
#pmag.drop_limit()

pmag.scenario = 'IM'

pmag.solve()
#pmag.solve_lstsq()

print(np.linalg.norm(pmag.err))

pmag.plot(subcoil=False, current='A')
pmag.grid.plot_flux(color='C3')
#pmag.grid.plot_field()

#pmag.target.add_targets([1.409500, 4.522150])
#print(pmag.target.Bz, -pmag.target.mu_o*40e3*554/2.093)


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