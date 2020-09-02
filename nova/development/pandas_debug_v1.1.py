from nova.electromagnetic.coilset import CoilSet
from nova.electromagnetic.coilframe import CoilFrame


cs = CoilSet()

cs.add_coil(1, 2, 1, 1)
cs.plot()