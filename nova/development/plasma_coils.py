import numpy as np
from scipy.optimize import minimize

from nova.electromagnetic.coilgeom import ITERcoilset
from nova.utilities.pyplot import plt

ITER = ITERcoilset(coils='pf', dCoil=0.2, dPlasma=0.2, dField=0.2,
                   plasma_n=2e2, n=1e3, read_txt=True)

ITER.filename = -1
ITER.scenario = 'EOF'

ITER.separatrix = ITER.data['separatrix']


#def minloc():
#    np.argmin(ITER.subcoil.z[ITER.ionize_index])

'''
def fun(x):
    return ITER.plasmagrid.B_rbs.ev(*x)

x = minimize(fun, [ITER.coil.x[-1], ITER.coil.z[-1]+5],
             bounds=(ITER.plasmagrid.grid_boundary[:2],
                     ITER.plasmagrid.grid_boundary[2:])).x

print(x)
'''

#ITER.Ic *= -1

#ITER.Ip = -15e6

plt.set_aspect(0.8)
ITER.plot()
#levels = ITER.grid.plot_flux()
ITER.plasmagrid.plot_flux()

plt.plot(*ITER.plasmagrid.Opoint, 'ko')

#opt = ITER.plasmagrid.get_Xpoint([5, 0])
#plt.plot(*ITER.plasmagrid.Xpoint, 'kX')

ITER.grid.contour(ITER.plasmagrid.Opsi + 70, plot=True)

#ITER.plasmagrid.plot()

ITER.plasmagrid.get_global_null(plot=True)

print(*ITER.plasmagrid.Opoint)
print(ITER.plasmagrid._Opoint)

#ITER.plot_data(['firstwall', 'divertor'])
#plt.plot(*ITER.data['divertor'].iloc[1:].values.T)

#ITER.field.plot()
#ITER.plasmafilament.plot()


