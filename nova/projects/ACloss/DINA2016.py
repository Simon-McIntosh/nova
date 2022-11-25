
import os
from dataclasses import dataclass, field
from typing import Union

import numpy as np
import scipy

from nova.frame.coilset import CoilSet
from nova.frame.coilgeom import ITERcoilset
from nova.frame.timeseries import DataArray, DataSet
from nova.utilities.localdata import LocalData
from nova.plot import plt
from nova.utilities.time import clock


@dataclass
class Transient:
    """Manage transient calculations."""

    filename: str
    _coilset: Union[CoilSet, dict] = field(repr=False, default_factory=dict)
    psi_inverse: np.array = field(init=False, repr=False)

    def __post_init__(self):
        """Configure paths and load coilset."""
        self.local = LocalData('', 'Nova', source='coilsets', data='transient')
        if self.coilset:
            self.coilset = self._coilset
        else:
            self._load_coilset()

    @property
    def coilset(self):
        """Manage coilset property."""
        return self._coilset

    @coilset.setter
    def coilset(self, coilset):
        self._coilset = CoilSet(**coilset)
        self._update_biot_instances()
        self.coilset.save_coilset(self.filename, self.local.source_directory)
        self.psi_inverse = np.linalg.inv(self.coilset.passive._psi)

    def _load_coilset(self):
        """Load coilset from file."""
        if os.path.isfile(os.path.join(
                self.local.source_directory, f'{self.filename}.pk')):
            self.coilset = CoilSet().load_coilset(self.filename)
        else:
            self._coilset = CoilSet()

    def _update_biot_instances(self):
        biot_instances = [instance for instance in ['passive', 'background']
                          if instance not in self.coilset.biot_instances]
        if biot_instances:
            self.coilset.biot_instances = biot_instances

    def rebuild(self):
        """Reduild coilset."""
        print('rebuild')
        coilset = ITERcoilset(
            name='trans',
            coils='pf trs dir vv', dCoil=0.25, dPlasma=0.25, dField=0.25,
            plasma_expand=0.2, plasma_n=2e3, n=1e3, read_txt=True).coilset
        print(coilset['coilset_metadata']['name'])


if __name__ == '__main__':

    #waveform = WaveForm('15MA DT-DINA2017-04_v1.2', 30)

    trans = Transient('transient')
    trans.rebuild()
    #trans.coilset.plot()





'''
#ITER.biot_instances = ['probe']
#ITER.probe.add_target(7.5, -2.7)


#ITER.filename = '15MA DT-DINA2016-01_v1.1'
ITER.filename = '15MA DT-DINA2017-04_v1.2'
ITER._update_plasma = True
ITER.scenario = 'SOB'

Nt = 1000
t, dt = np.linspace(ITER.d2.t[0], 0.5, Nt, retstep=True) #ITER.d2.t[-1]

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

order = 1
coefficents = scipy.special.binom(order, np.arange(order+1))
kernel = (-1)**np.arange(order+1) * coefficents

ITER.scenario = acloss.time[0]
Psi = np.ones((ITER.background.target._nC, order+1))
Psi *= ITER.background.Psi.reshape(-1, 1)
timestep = acloss.time[0] - np.arange(1, order+2) * np.diff(acloss.time[:2])[0]

for i, t in enumerate(acloss.time):
    ITER.scenario = t

    timestep[1:] = timestep[:-1]
    timestep[0] = t
    Psi[:, 1:] = Psi[:, :-1]
    Psi[:, 0] = ITER.background.Psi

    dPsi = np.sum(Psi*kernel, axis=1) / np.prod(np.diff(timestep[::-1]))

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

'''

#ITER.plot(True)
#ITER.plasmagrid.plot_flux()

plt.figure()
iloc = int(np.where(ITER.coil.index == 'CS3L')[0])
index = slice(*ITER.subcoil._reduction_index[iloc:iloc+2])
plt.plot(cds['Bx'].t, cds['Bx'].data[:, index])
'''
