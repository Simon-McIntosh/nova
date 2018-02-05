import numpy as np
from scipy.integrate import odeint
from amigo.pyplot import plt
from nova.coils import PF
from nova.inverse import INV



pf = PF()  # PF coil object


pf.add_coil(5.3, 3.2, 0.4, 0.6, 1)
pf.add_coil(5.8, 4.9, 0.1, 0.1, 1)  # vs upper
pf.mesh_coils()
pf.plot()


inv = INV(pf, Iscale=1)
inv.update_coils()

Nf = np.ones(inv.nC)
turns = np.ones(inv.nC)
for i, coil in enumerate(inv.adjust_coils):
    x, z = pf.coil[coil]['x'], pf.coil[coil]['z']
    inv.add_psi(1, point=(x, z))

inv.set_foreground()
t2 = np.dot(turns.reshape((-1, 1)), turns.reshape((1, -1)))
fillaments = np.dot(np.ones((len(turns), 1)), Nf.reshape(1, -1))
M = 2 * np.pi * inv.G * t2 * fillaments

print(M)


#M = np.matrix([[1, 0.5],
#               [0.5, 1]])

Rpl = M[0,0]/16e-3

R = np.array([Rpl, 17.66e-3])  # 17.66e-3

Io = [15e6, 0]
t = np.linspace(0, 0.3, 1000)


def dIdt(I, t):
    Idot = np.linalg.solve(-M, I*R)
    return Idot


Iode = odeint(dIdt, Io, t)

ax = plt.subplots(2, 1, sharex=True)[1]
for i, I in enumerate(Iode.T):
    if i == 0:
        ax[0].plot(t, 1e-6*I)
    else:
        ax[1].plot(t, 1e-3*I)
