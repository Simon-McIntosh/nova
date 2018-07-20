import os
import nep
from amigo.png_tools import data_mine, data_load
from amigo.IO import class_dir
from amigo.pyplot import plt
import numpy as np

path = os.path.join(class_dir(nep), '../Data/EM/')
#data_mine(path, 'VS3_risetime', [5, 6], [0, 3e5])


points = data_load(path, 'VS3_risetime', date='2018_07_20')[0]


t = points[0]['x'] - 5
Ic = points[0]['y']/4

plt.plot(1e3*t, 1e-3*Ic)

plt.despine()
plt.xlabel('$t$ ms')
plt.ylabel('$I_{vs3}$ kA')
plt.legend()
plt.xlim(180, 250)

Imax = np.max(Ic)
t10 = t[Ic > 0.1*Imax][0]
I10 = Ic[Ic > 0.1*Imax][0]
t90 = t[Ic > 0.9*Imax][0]
I90 = Ic[Ic > 0.9*Imax][0]
plt.plot(1e3*t10, 1e-3*I10, 'o')
plt.plot(1e3*t90, 1e-3*I90, 'o')

dt_rise = t90 - t10
print('risetime {:1.1f}ms'.format(1e3*(t90 - t10)))


