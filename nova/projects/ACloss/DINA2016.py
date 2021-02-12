import time

import numpy as np

from nova.electromagnetic.coilgeom import ITERcoilset
from nova.electromagnetic.timeseries import DataArray, DataSet
from nova.utilities.pyplot import plt
from nova.utilities.time import clock

ITER = ITERcoilset(coils='pf trs vv', dCoil=0.25, dPlasma=0.25, dField=0.25,
                   plasma_expand=0.2, plasma_n=2e3,
                   n=1e3, read_txt=False)

#ITER.biot_instances = ['probe']
#ITER.probe.add_target(7.5, -2.7)


#ITER.filename = '15MA DT-DINA2016-01_v1.1'
ITER.filename = '15MA DT-DINA2017-04_v1.2'
ITER._update_plasma = True
ITER.scenario = 'SOB'

Nt = 10000
t, dt = np.linspace(ITER.d2.t[0], 8, Nt, retstep=True) #ITER.d2.t[-1]

def dIdt(self, It, t, *args):  # current rate (function for odeint)
    vfun = args[0]
    if vfun is None:
        vbg = np.zeros(len(It))  # background field
    else:
        vbg = np.array([vf(t) for vf in vfun])
    Idot = np.dot(self.Minv, vbg - It*self.Rc)
    return Idot

def solve(self, t, **kwargs):
    self.Minv = np.linalg.inv(self.M)  # inverse for odeint
    vfun = kwargs.get('vfun', None)
    Iode = odeint(self.dIdt, self.It, t, (vfun,)).T
    return Iode

acloss = DataSet((t, ITER.acloss.target, ['Bx', 'Bz', 'Psi']))
background = DataSet((t, ITER.background.target, ['Psi', 'dPsi']))

tick = clock(Nt, print_rate=500, print_width=20, header='computing B')

Psi = np.zeros((ITER.background.target._nC, 2))
#phi_dot =
for i, t in enumerate(acloss.time):
    ITER.scenario = t

    if i == 0:
        Psi[:, 0] = ITER.background.Psi
    Psi[:, 1] = Psi[:, 0]
    Psi[:, 0] = ITER.background.Psi
    dPsi = (Psi[:, 1] - Psi[:, 0]) / dt

    background['Psi'].data[i] = ITER.background.Psi
    background['dPsi'].data[i] = dPsi

    #acloss['Bx'].data[i] = ITER.acloss.Bx
    #acloss['Bz'].data[i] = ITER.acloss.Bz
    #acloss['Psi'].data[i] = ITER.acloss.Psi
    tick.tock()

#ITER.plot()
#plt.plot(acloss['Psi'].time, acloss['Psi'].data)

#index = background['Psi'].coil.tolist().index('trs')
#index = slice(background['Psi'].attrs['indices'][index],
#              background['Psi'].attrs['indices'][index+1])

plt.plot(background['Psi'].time, background['dPsi'])

plt.figure()
ITER.plot()
ITER.grid.plot_flux()



'''

#ITER.plot(True)
#ITER.plasmagrid.plot_flux()

plt.figure()
iloc = int(np.where(ITER.coil.index == 'CS3L')[0])
index = slice(*ITER.subcoil._reduction_index[iloc:iloc+2])
plt.plot(cds['Bx'].t, cds['Bx'].data[:, index])
'''
