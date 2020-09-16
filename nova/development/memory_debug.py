from guppy import hpy

from nova.electromagnetic.coilgeom import ITERcoilset

#hp = hpy()  # initialize memory manager



# load ITER coilset
ITER = ITERcoilset(coils='pf vv vs', dCoil=0.25, n=5e4, 
                   expand=0.0, levels=31,
                   read_txt=False, current_update='full')

#ITER.add_coil(5, [-2, 4], 1.2, [0.3, 2.4], name='C', delim='',
#              optimize=True)

#ff.source.nT = 5

#hp.setrelheap()  # set relative heap

#ITER.forcefield.calculate()


#ITER.grid.generate_grid()

ITER.filename = -1
ITER.scenario = 'IM'

ITER.current_update = 'passive'
ITER.It = 20e4

ITER.grid.plot_flux()

ITER.plot(True)
#ITER.forcefield.plot()

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

