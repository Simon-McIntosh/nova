from guppy import hpy

from nova.electromagnetic.coilgeom import ITERcoilset


hp = hpy()  # initialize memory manager
hp.setrelheap()  # set relative heap

# load ITER coilset
ITER = ITERcoilset(coils='pf', dCoil=0.25, n=1e3, 
                   limit=[3, 10, -5.75, 5.75], levels=61, read_txt=True)

h = hp.heap()  # get heap
byrcs= h.byrcs

print(byrcs)
print(byrcs[0].byvia)