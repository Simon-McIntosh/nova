import time

import numpy as np

from nova.electromagnetic.coilgeom import ITERcoilset
from nova.electromagnetic.timeseries import DataArray, DataSet
from nova.utilities.pyplot import plt
from nova.utilities.time import clock

ITER = ITERcoilset(coils='pf vv', dCoil=0.25, dPlasma=0.25, dField=0.25,
                   plasma_expand=0.2, plasma_n=2e3,
                   n=1e3, read_txt=True)


ITER.filename = '15MA DT-DINA2016-01_v1.1'
ITER._update_plasma = True
ITER.scenario = 'SOB'

ITER._update_plasma = False

Nt = 500
t = np.linspace(ITER.d2.t[0], ITER.d2.t[-1], Nt)

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

dataset = DataSet((t, ITER.acloss.target, ['Bx', 'Bz']))

tick = clock(Nt, print_rate=500, print_width=20, header='computing B')

#phi_dot =
for i, t in enumerate(dataset.time):
    #ITER.scenario = t
    ITER.Ic = 5e3
    dataset['Bx'].data[i] = ITER.acloss.Bx
    dataset['Bz'].data[i] = ITER.acloss.Bz
    #dataset['B'].data[i] = ITER.acloss.B
    tick.tock()

ITER.plot()

'''

#ITER.plot(True)
#ITER.plasmagrid.plot_flux()

plt.figure()
iloc = int(np.where(ITER.coil.index == 'CS3L')[0])
index = slice(*ITER.subcoil._reduction_index[iloc:iloc+2])
plt.plot(cds['Bx'].t, cds['Bx'].data[:, index])
'''
