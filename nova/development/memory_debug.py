from guppy import hpy

from nova.electromagnetic.coilgeom import ITERcoilset
from nova.electromagnetic.biotmethods import ForceField

#hp = hpy()  # initialize memory manager
#hp.setrelheap()  # set relative heap

# load ITER coilset
ITER = ITERcoilset(coils='pf', dCoil=0.25, n=1e3, 
                   limit=[3, 10, -5.75, 5.75], levels=61, read_txt=True)

# ITER.coil.coilframe_metadata = {'_coilframe_attributes': ['x', 'z']}

ff = ForceField(ITER.subcoil)

ITER.add_coil(5, [-2, 4], 1.2, [0.3, 2.4], name='C', delim='',
              optimize=True)

print(ff.source)

ff.source.update_coilframe()

print(ff.source)

#ff.assemble()

ITER.plot()
#ITER.mutual.points
'Total size = 28731699 bytes'

#h = hp.heap()  # get heap
#byrcs= h.byrcs

#print(byrcs)
#print(byrcs[0].byvia)