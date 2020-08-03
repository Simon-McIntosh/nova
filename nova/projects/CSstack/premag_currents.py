from amigo.pyplot import plt

from nova.design.inverse import Inverse
from nova.electromagnetic.coilgeom import ITERcoilset
from nova.electromagnetic.machinedata import MachineData


premag = Inverse()
premag.load_coilset('ITER')

'''
ITER = ITERcoilset(coils='pf', dCoil=0.2, n=2e3, limit=[4, 8.5, -3, 3],
                   read_txt=False)
cc = ITER.cc
cc.scenario_filename = -2
cc.scenario = 'IM'
cc.save_coilset('ITER')
'''

#premag.Ic = {'PF6': 0}

plt.set_aspect(1.1)
premag.plot(subcoil=False)

#premag.grid.generate_grid(limit=[3.5, 9, -5, 5])
#premag.save_coilset('ITER')
#premag.grid.plot_flux()

machine = MachineData()
machine.load_data()
machine.plot_data(['firstwall', 'divertor'], color='k')
    
premag.label_gaps()

#machine.load_models()
#machine.plot_models()
    
#machine.load_coilset()
#machine.plot()
