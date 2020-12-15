from nova.electromagnetic.coilgeom import ITERcoilset
from nova.utilities.pyplot import plt

ITER = ITERcoilset(coils='pf', dCoil=0.25, dPlasma=0.15, dField=0.25,
                   plasma_expand=0.4, plasma_n=2e4,
                   n=1e3, read_txt=False)

ITER.filename = -1
ITER.scenario = 'SOF'

#ITER.data['separatrix'].z += 0.1
ITER.separatrix = ITER.data['separatrix']
ITER.coil.P

ITER.plot()
ITER.plasmagrid.plot_flux(levels=31)