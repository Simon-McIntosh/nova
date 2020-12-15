import time

from nova.electromagnetic.coilgeom import ITERcoilset
from nova.utilities.pyplot import plt

ITER = ITERcoilset(coils='pf', dCoil=0.25, dPlasma=0.15, dField=0.25,
                   plasma_expand=0.4, plasma_n=2e4,
                   n=1e3, read_txt=False)

ITER.filename = -1
ITER.scenario = 'EOF'

#ITER.data['separatrix'].z += 0.1
ITER.separatrix = ITER.data['separatrix']

ITER.current_update = 'feedback'
ITER.Ic = 0

#ITER.current_update = 'full'
#ITER.Ic = 0

ITER.plasmagrid.optimize = 'newton'
ITER.plasmagrid.filter_sigma = 3
#ITER.plasmagrid.cluster = True
ITER.plasmagrid.ftol_rel = 1e-12
ITER.plasmagrid.xtol_rel = 1e-12

plt.set_aspect(0.8)

zo = 0
k = 8e6

tic = time.perf_counter()
for i in range(1):
    err = (ITER.Opoint[1]-zo)
    ITER.Ic = k*err
    #ITER.update_separatrix(alpha=1, plot=True)

    print(err)

toc = time.perf_counter()
print(f'time {toc-tic:1.3f}s')

ITER.plot(True, feedback=False)
ITER.plot_null()
ITER.plasmagrid.plot_topology(True)
ITER.plasmagrid.plot_flux(levels=201)
plt.plot(*ITER.separatrix.boundary.xy)

#ITER.plot_data(['firstwall', 'divertor'])
#plt.plot(*ITER.data['divertor'].iloc[1:].values.T)

#ITER.field.plot()
#ITER.plasmafilament.plot()


