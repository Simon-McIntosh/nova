
import numpy as np

from nova.plot import plt
from nova.design.inverse import Inverse
from nova.frame.coilset import CoilSet
from nova.frame.coilgeom import ITERcoilset
from nova.frame.machinedata import MachineData

build_coilset = True

pmag = Inverse()

if build_coilset:
    ITER = ITERcoilset(coils='pf vv trs dir', dCoil=-1, n=5e3,
                       dPlasma=0.15, plasma_n=1e2,
                       levels=31, dField=-1,
                       biot_instances='colocate', read_txt=True)
    pmag.coilset = ITER.coilset
    pmag.scenario_filename = -2
    pmag.scenario = 'SOF'
    pmag.separatrix = ITER.data['separatrix']
    pmag.add_polygon(pmag.separatrix, N=30)
    pmag.save_coilset('ITER')
else:
    pmag.load_coilset('ITER')

pmag.scenario_filename = -1
pmag.scenario = 'EOB'

pmag.current_update = 'coil'

#pmag.colocate.update_target()
pmag.set_foreground()
pmag.set_background()
pmag.set_target()

#pmag.wT = pmag.colocate.Psi.mean() - pmag.BG

pmag.solve()

plt.set_aspect(0.7)
pmag.plot(subcoil=True, label='active')
pmag.grid.plot_flux()
