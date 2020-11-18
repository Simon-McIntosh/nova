import time

import numpy as np

from nova.electromagnetic.coilgeom import ITERcoilset
from nova.electromagnetic.acloss import CoilDataArray, CoilDataSet
from nova.utilities.pyplot import plt
from nova.utilities.time import clock

ITER = ITERcoilset(coils='pf', dCoil=-1, dPlasma=0.15, dField=0.25,
                   plasma_expand=0.2, plasma_n=2e3,
                   n=1e3, read_txt=False)

ITER.filename = '15MA DT-DINA2016-01_v1.1'
ITER._update_plasma = True
ITER.scenario = 'SOB'

ITER._update_plasma = False

Nt = 50000
t = np.linspace(ITER.d2.t[0], ITER.d2.t[-1], Nt)
cda = CoilDataArray(t, ITER.forcefield.target, 'B')
cds = CoilDataSet()
for var in ['Bx', 'Bz', 'B']:
    cds[var] = CoilDataArray(t, ITER.forcefield.target, var).data

tick = clock(Nt, print_rate=100, print_width=20, header='computing B')
for i, t in enumerate(cda.time):
    ITER.scenario = t
    cds['Bx'].data[i] = ITER.forcefield.Bx
    cds['Bz'].data[i] = ITER.forcefield.Bz
    cds['B'].data[i] = ITER.forcefield.B
    tick.tock()

cds.save('B_2016_dev', 'Poloidal field values from DINA 2016-01 v1.1',
         code='Nova', coilname=ITER.coilname,
         dCoil=ITER.dCoil, dPlasma=ITER.dPlasma, Nt=Nt,
         replace=True)

cds.data.close()

#ITER.plot(True)
#ITER.plasmagrid.plot_flux()

plt.figure()
iloc = int(np.where(ITER.coil.index == 'CS3L')[0])
index = slice(*ITER.subcoil._reduction_index[iloc:iloc+2])
plt.plot(cds['Bx'].t, cds['Bx'].data[:, index])
