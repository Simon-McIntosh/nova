import numpy as np
from scipy.optimize import minimize_scalar
from collections.abc import Iterable
from numba import jit


def gmr_offset(x_gmr, *args):
    x, dx = args  # coil geometry
    g = np.exp(1/dx * ((x_gmr + dx/2) * np.log(x_gmr + dx/2)
                       - dx - (x_gmr - dx/2) * np.log(x_gmr - dx/2)))
    offset = np.abs(g - x)
    return offset


@jit
def coil_gmr(x, dx):
    '''
    Attributes:
        x (float or itterable): geometric coil center
        dx (float): coil radial width
    '''
    if not isinstance(x, Iterable):
        x = [x]

    x_gmr = np.zeros(len(x))
    for i, x_ in enumerate(x):
        res = minimize_scalar(gmr_offset, method='bounded',
                              bounds=(x_, x_+2*dx), args=(x_, dx))
        x_gmr[i] = res.x
    return x_gmr


x, dx = 5.5, 2.5

Nx = 5
Ndx = 40
xm = np.linspace(1.0, 5, Nx)
dxm = np.linspace(0.05, 20, Ndx)

for i in range(Nx):
    x = np.zeros(Ndx)
    for j in range(Ndx):
        x[j] = coil_gmr(xm[i], dxm[j])[0]
    plt.plot(dxm, x)

