import numpy as np
from scipy.optimize import minimize

from nova.electromagnetic.coilgeom import ITERcoilset
from nova.utilities.pyplot import plt

ITER = ITERcoilset(coils='pf', dCoil=0.25, dPlasma=0.25, dField=0.25,
                   plasma_expand=1.1, plasma_n=1e3, n=1e3, read_txt=False)

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

#levels = ITER.grid.plot_flux()
ITER.plasmagrid.plot_flux()

#plt.plot(*ITER.plasmagrid.Opoint, 'ko')

#opt = ITER.plasmagrid.get_Xpoint([5, 0])
#plt.plot(*ITER.plasmagrid.Xpoint, 'kX')

#ITER.grid.contour(ITER.plasmagrid.Opsi + 70, plot=True)

#ITER.plasmagrid.plot()

#ITER.plasmagrid.field_quantile = 0.05

ITER.plasmagrid.ftol_rel = 1e-9
ITER.plasmagrid.xtol_rel = 1e-9


Psi = ITER.Xpsi + 0.5*(ITER.Opsi-ITER.Xpsi)


contour = ITER.plasmagrid.contour(Psi, plot=True)
closed_contour = []
for c in contour:
    if np.isclose(np.linalg.norm(c[0]-c[-1]), 0):
        closed_contour.append(c)

plt.plot(*ITER.Opoint, 'C3d')
plt.plot(*ITER.Xpoint, 'C3d')


ITER.plot()

Psi = ITER.plasmagrid.Xpsi[0] + 1e-3*(ITER.plasmagrid.Opsi[0] -
                                      ITER.plasmagrid.Xpsi[0])
ITER.plasmagrid.plot_topology(True, color='C3')
#plt.plot(*ITER.plasmagrid.Opoint[0], 'C1o', ms=15)
#ITER.separatrix = c[1]

#print(*ITER.plasmagrid.Opoint)
#print(ITER.plasmagrid._Opoint)

#ITER.plot_data(['firstwall', 'divertor'])
#plt.plot(*ITER.data['divertor'].iloc[1:].values.T)

#ITER.field.plot()
#ITER.plasmafilament.plot()



