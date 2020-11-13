import numpy as np
from scipy.optimize import minimize

from nova.electromagnetic.coilgeom import ITERcoilset
from nova.utilities.pyplot import plt

ITER = ITERcoilset(coils='pf', dCoil=0.25, dPlasma=0.15, dField=0.25,
                   plasma_expand=0.6, plasma_n=2e3,
                   n=1e3, read_txt=False)

ITER.filename = -1
ITER.scenario = 'EOF'

ITER.separatrix = ITER.data['separatrix']


plt.set_aspect(0.8)

#ITER.plot_null()

#ITER.plasmagrid.cluster = True
#ITER.plasmagrid.plot_topology(True)

for __ in range(1):
    ITER.update_separatrix(alpha=1, plot=True)

ITER.plot(True)
ITER.plasmagrid.plot_topology(True)
ITER.plasmagrid.plot_flux()


#ITER.plot_data(['firstwall', 'divertor'])
#plt.plot(*ITER.data['divertor'].iloc[1:].values.T)

#ITER.field.plot()
#ITER.plasmafilament.plot()



