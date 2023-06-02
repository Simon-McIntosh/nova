import xarray
import numpy as np
import scipy.integrate
import scipy.interpolate

from nova.frame.acloss import DataIO
import matplotlib.pyplot as plt

coil = "CS3L"

ac = DataIO()
with xarray.open_dataset(ac.catalog["B_2016_dev"].urlpath) as dataset:
    indices = np.append(dataset.B.indices, dataset.B.nT)
    coil_index = dataset.B.coil.index(coil)
    turn_index = slice(indices[coil_index], indices[coil_index + 1])
    B = dataset.B[:, turn_index]
    Bx = dataset.Bx[:, turn_index]
    Bz = dataset.Bz[:, turn_index]
    t = dataset.t
    x, z = dataset.B.x[turn_index], dataset.B.z[turn_index]

# create field interpolant
Be = scipy.interpolate.interp1d(t, Bx, axis=0, assume_sorted=True)


def Bdot(t, Bi, tau):
    r"""
    Return gradient of induced field.

    Formulae
    --------
    .. math:: \frac{d\mathbf{B_i}}{dt} = \frac{\mathbf{B_e}-\mathbf{B_i}}{\tau}


    Parameters
    ----------
    t : float
        Evaluation time.
    Bi : array-like
        Induced field.
    tau : float
        Time constant.

    Returns
    -------
    Bdot : array-like
        Time derivitive of induced field.

    """
    return (Be(t) - Bi) / tau


t_max = 2
iloc = np.argmin(abs(t.values - t_max))

t_span = (t[0], t[iloc])
t_eval = t[:iloc]
sol = scipy.integrate.solve_ivp(Bdot, t_span, Bx[0], args=(9,), t_eval=t_eval)


plt.plot(sol.t, sol.y.T, "C0")
plt.plot(t[:iloc], Bx[:iloc], "C3")

"""
plt.plot(t, B)

plt.figure()
plt.plot(x, z, 'C3o')
plt.axis('equal')
"""
