

from nova.electromagnetic.coilgeom import ITERcoilset
from nova.utilities.pyplot import plt

ITER = ITERcoilset(coils='pf', dCoil=0.25, dPlasma=0.15, dField=0.25,
                   plasma_expand=0.2, plasma_n=2e3,
                   n=1e3, read_txt=False)

ITER.filename = -1
ITER.scenario = 'EOF'

#ITER.data['separatrix'].z += 0.1
ITER.separatrix = ITER.data['separatrix']

ITER.current_update = 'feedback'
ITER.Ic = 0
ITER.Ip = -16.5e6

#ITER.current_update = 'full'
#ITER.Ic = 0

ITER.plasmagrid.optimize = 'newton'


plt.set_aspect(0.8)

zo = 1
k = 2e6

for __ in range(25):
    err = (ITER.Opoint[1]-zo)
    ITER.update_separatrix(alpha=1)

    print(err)
    ITER.Ic = k*err

ITER.plot(True, feedback=False)
ITER.plot_null()
ITER.plasmagrid.plot_topology(True)
ITER.plasmagrid.plot_flux()
plt.plot(*ITER.separatrix.boundary.xy)

#ITER.plot_data(['firstwall', 'divertor'])
#plt.plot(*ITER.data['divertor'].iloc[1:].values.T)

#ITER.field.plot()
#ITER.plasmafilament.plot()


