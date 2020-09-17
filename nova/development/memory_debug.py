from guppy import hpy
import numpy as np

from nova.electromagnetic.coilgeom import ITERcoilset
from nova.electromagnetic.IO.read_scenario import field_data

#hp = hpy()  # initialize memory manager



# load ITER coilset
ITER = ITERcoilset(coils='pf vv dir trs', dCoil=0.25, n=5e3, 
                   expand=0.0, levels=31,
                   read_txt=True, current_update='full',
                   biot_instances={'field': 'target'})

#ITER.add_coil(5, [-2, 4], 1.2, [0.3, 2.4], name='C', delim='',
#              optimize=True)

#ff.source.nT = 5

#hp.setrelheap()  # set relative heap

#ITER.forcefield.calculate()


#ITER.grid.generate_grid()

ITER.filename = -1
ITER.scenario = 'IM'

#ITER.current_update = 'passive'
#ITER.It = 20e4

ITER.grid.plot_flux()

ITER.plot()
#ITER.forcefield.plot()

d3 = field_data(read_txt=False)
d3.load_file(-1)
d3.to = ITER.t

ITER.target.add_target(ITER.coil.loc['CS1U', 'x'] - 
                       ITER.coil.loc['CS1U', 'dx'] / 2,
                       ITER.coil.loc['CS1U', 'z'] + np.linspace(
                        -ITER.coil.loc['CS1U', 'dz']/2,
                        ITER.coil.loc['CS1U', 'dz']/2, 30))
ITER.target.plot()
ITER.target.update_biotset()
print(ITER.target.B.max())
    
print(12.938000)

'''
B_cs3u         11.615000
B_cs2u         12.789000
B_cs1u         12.938000
B_cs1l         12.960000
B_cs2l         12.956000
B_cs3l         12.724000
'''

'''
#ff.assemble()

#ITER.plot(False)
#ITER.mutual.points
'Total size = 28731699 bytes'

h = hp.heap()  # get heap
byrcs= h.byrcs

print(byrcs)
print(byrcs[0].byvia)
'''

