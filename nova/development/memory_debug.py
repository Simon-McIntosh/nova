from guppy import hpy
import numpy as np

from nova.electromagnetic.coilgeom import ITERcoilset
from nova.electromagnetic.IO.read_scenario import field_data
from amigo.pyplot import plt

#hp = hpy()  # initialize memory manager



# load ITER coilset
ITER = ITERcoilset(coils='pf vv dir trs', dCoil=-1, dField=0.1,
                   n=1e3, expand=0.0, levels=31,
                   read_txt=False, current_update='full')

ITER.add_coil(6, 0, 0.2, 0.2, name='plasma', delim='',
              Ic=1000, cross_section='circle')

ITER.field.solve()
ITER.forcefield.solve()

#ff.source.nT = 5

#hp.setrelheap()  # set relative heap

#ITER.forcefield.calculate()


#ITER.grid.generate_grid()


ITER.filename = -1
ITER.scenario = 'IM'

print(1e-6*ITER.forcefield.Fx)

#ITER.current_update = 'passive'
#ITER.Ic = 0

plt.set_aspect(1.0)
#ITER.grid.plot_flux()

ITER.plot()
#ITER.field.plot()
#ITER.forcefield.plot()

'''
d3 = field_data(read_txt=False)
d3.load_file(-1)
d3.to = ITER.t
print(d3.vector)


ITER.target.add_target(ITER.coil.loc['CS1U', 'x'] - 
                       ITER.coil.loc['CS1U', 'dx'] / 2,
                       ITER.coil.loc['CS1U', 'z'] + np.linspace(
                        -ITER.coil.loc['CS1U', 'dz']/2,
                        ITER.coil.loc['CS1U', 'dz']/2, 50))
ITER.target.plot()
ITER.target.update_biotset()
print(ITER.target.B.max())
print(ITER.target.B.argmax())  
print(12.938000)
'''

'''
Fx_cs3u      1095.000000
Fx_cs2u      1361.300000
Fx_cs1u      1386.000000
Fx_cs1l      1391.100000
Fx_cs2l      1394.400000
Fx_cs3l      1344.500000

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

