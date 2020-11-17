import time

import numpy as np

from nova.electromagnetic.coilgeom import ITERcoilset
from nova.electromagnetic.acloss import CoilDataArray

from nova.utilities.pyplot import plt

ITER = ITERcoilset(coils='pf', dCoil=-1, dPlasma=0.15, dField=0.25,
                   plasma_expand=0.2, plasma_n=2e3,
                   n=1e3, read_txt=False)

ITER.filename = '15MA DT-DINA2016-01_v1.1'
ITER._update_plasma = True
ITER.scenario = 'SOB'

ITER._update_plasma = False

Nt = 1000
tick = time.perf_counter()


t = np.linspace(ITER.d2.t[0], ITER.d2.t[-1], Nt)
cda = CoilDataArray(t, ITER.forcefield.target, 'B')

for i, t in enumerate(cda.time):
    ITER.scenario = t
    cda.data[i] = ITER.forcefield.B

cda.save('B_2016', 'Poloidal field values from DINA 2016-01 v1.1',
         code='Nova', coilname=ITER.coilname,
         dCoil=ITER.dCoil, dPlasma=ITER.dPlasma,
         replace=False)

print(f'elapsed {time.perf_counter() - tick:1.3f}s')

ITER.plot(True)
ITER.plasmagrid.plot_flux()


plt.figure()

iloc = int(np.where(ITER.coil.index == 'CS3L')[0])
index = slice(*ITER.subcoil._reduction_index[iloc:iloc+2])
plt.plot(cda.time, cda.data[:, index])
