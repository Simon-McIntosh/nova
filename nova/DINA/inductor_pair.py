from nep.DINA.coupled_inductors import inductance
import numpy as np
from scipy.integrate import odeint
from amigo.pyplot import plt
from read_dina import timeconstant


ind = inductance()

Io = 60
R = np.array([1e-3, 1e-3])

ind.add_coil(5, 0.2, 0.1, 0.1, Io, R=R[0], nt=1)  # primary turn
ind.add_coil(5, -0.2, 0.1, 0.1, 0, R=R[1], nt=1)  # secondary turn

ind.assemble()
ind.assemble_cp_nodes()
ind.constrain()

t = np.linspace(0, 0.3, 3000)

ind.Minv = np.linalg.inv(ind.M)  # inverse for odeint
Iode = odeint(ind.dIdt, ind.Ic, t).T

M = ind.M/(np.ones((2, 1))*R.reshape(1, 2))
print(np.linalg.eig(M))

for i in range(len(Iode)):
    plt.plot(t, Iode[i])

    if i == 0:
        tc = timeconstant(t, Iode[i], trim_fraction=0.15)  # discharge
        tdis, ttype, tfit, Ifit = tc.fit(plot=False)
        plt.plot(tfit, Ifit, '--')
        print(ttype)
        print('tau_o={:1.1f}ms'.format(ind.M[0, 0]/R[0]*1e3))
        print('tau_d={:1.1f}ms'.format(tdis*1e3))
