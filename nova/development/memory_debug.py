from guppy import hpy

from nova.electromagnetic.coilgeom import ITERcoilset
from nova.electromagnetic.coilset import CoilSet

hp = hpy()  # initialize memory manager


#cs = CoilSet(dCoil=3, current_update='coil', turn_fraction=0.5,
#             cross_section='circle', biot_instances={'fd': 'forcefield'},
#             coildata_attributes={'rms_offset': False})

# load ITER coilset
ITER = ITERcoilset(coils='pf', dCoil=0.25, n=1e3, 
                   limit=[3, 10, -5.75, 5.75], levels=61, 
                   read_txt=True, current_update='passive')

ITER.plot()



#ITER.add_coil(5, [-2, 4], 1.2, [0.3, 2.4], name='C', delim='',
#              optimize=True)

#ff.source.nT = 5

hp.setrelheap()  # set relative heap

ITER.forcefield.calculate()

ITER.grid.generate_grid()
#ITER.forcefield.plot()

#ff.assemble()

#ITER.plot(False)
#ITER.mutual.points
'Total size = 28731699 bytes'

h = hp.heap()  # get heap
byrcs= h.byrcs

print(byrcs)
print(byrcs[0].byvia)

