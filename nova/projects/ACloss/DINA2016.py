import time

import numpy as np

from nova.electromagnetic.coilgeom import ITERcoilset
from nova.utilities.pyplot import plt

ITER = ITERcoilset(coils='pf', dCoil=0.5, dPlasma=0.15, dField=0.25,
                   plasma_expand=0.4, plasma_n=2e3,
                   n=1e3, read_txt=False)

ITER.filename = '15MA DT-DINA2016-01_v1.1'
ITER.scenario = 'SOB'

Nt = 100
tick = time.perf_counter()
t_array = np.linspace(ITER.d2.t[0], ITER.d2.t[-1], Nt)
B = np.zeros((Nt, ITER.forcefield.nT))
for i, t in enumerate(t_array):
    ITER.scenario = t
    B[i] = ITER.forcefield.B
print(f'elapsed {time.perf_counter() - tick:1.3f}s')

ITER.plot(True)
ITER.plasmagrid.plot_flux()


plt.figure()

iloc = int(np.where(ITER.coil.index == 'CS3L')[0])
index = slice(*ITER.subcoil._reduction_index[iloc:iloc+2])
plt.plot(t_array, B[:, index])
