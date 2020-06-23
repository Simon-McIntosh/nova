from nep.DINA.coupled_inductors import inductance
import numpy as np
from scipy.integrate import odeint
from amigo.pyplot import plt
from read_dina import timeconstant


Io = 60
R = np.array([1e-3, 9e-3])
t = np.linspace(0, 0.8, 5000)

ind = inductance()
ind.add_coil(4, 0.2, 0.1, 0.1, 15, R=1e-1, nt=20)
ind.add_coil(5, 0.2, 0.1, 0.1, 0, R=R[0], nt=1)  # primary turn
Io = ind.solve(t)

#for i in range(ind.nM):
#    plt.plot(t, Isingle[i])

ind = inductance()
ind.add_coil(4, 0.2, 0.1, 0.1, 15, R=1e-1, nt=20)
ind.add_coil(5, 0.2, 0.1, 0.1, 0, R=R[0], nt=1)  # primary turn
ind.add_coil(4.5, 0.2, 0.1, 0.1, 0, R=R[1], nt=3)  # secondary turn
Ic = ind.solve(t)

for i in range(ind.nM):
    plt.plot(t, Ic[i])
plt.plot(t, Io[1], '--')

'''
tc = timeconstant(t, Ipair[0], trim_fraction=0.05)
Io_o, tau_o, tfit_o, Ifit_o = tc.nfit(2)
plt.plot(tfit_o, Ifit_o, '--')
txt_o = timeconstant.ntxt(Io_o/Ipair[0, 0], tau_o)
print(txt_o)
'''


#ind.add_coil(5, 0.4, 0.1, 0.1, 0, R=R[1], nt=1)  # third turn



#Iode_30 = odeint(ind.dIdt, [30, 0], t).T
# M = ind.M/(np.ones((2, 1))*R.reshape(1, 2))



#plt.plot(t, Iode_30[1], 'C2-')
#plt.plot(t, Iode[1]/2, 'C3--')


'''
if i == 0:
    tc = timeconstant(t, Iode[i], trim_fraction=0.05)  # discharge
    tdis, ttype, tfit, Ifit = tc.fit(plot=False, Io=60)
    plt.plot(tfit, Ifit, '--')
    print(ttype)
    print('tau_o={:1.1f}ms'.format(ind.M[0, 0]/R[0]*1e3))
    print('tau_d={:1.1f}ms'.format(tdis*1e3))
'''
