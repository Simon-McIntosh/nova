

from nova.electromagnetic.coilgeom import ITERcoilset
from nova.utilities.pyplot import plt

ITER = ITERcoilset(coils='pf', dCoil=0.25, dPlasma=0.15, dField=0.25,
                   plasma_expand=0.6, plasma_n=2e3,
                   n=1e3, read_txt=False)

ITER.filename = -1
ITER.scenario = 'EOF'

ITER.data['separatrix'].z += 0.1
ITER.separatrix = ITER.data['separatrix']


plt.set_aspect(0.8)

for __ in range(8):
    ITER.update_separatrix(alpha=1, plot=True)

ITER.plot(True)
ITER.plot_null()
#ITER.plasmagrid.plot_topology(True)
ITER.plasmagrid.plot_flux()


#ITER.plot_data(['firstwall', 'divertor'])
#plt.plot(*ITER.data['divertor'].iloc[1:].values.T)

#ITER.field.plot()
#ITER.plasmafilament.plot()



