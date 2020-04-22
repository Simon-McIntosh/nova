import os
import nep
from amigo.png_tools import data_mine, data_load
from amigo.IO import class_dir
from amigo.pyplot import plt
from scipy.interpolate import interp1d
import numpy as np
from os.path import join
import pickle

path = os.path.join(class_dir(nep), '../Data/LTC/')
# data_mine(path, 'Fr_upper_vs', [0, 0.08], [-200e2, 1800e2])
# data_mine(path, 'Fr_lower_vs', [0, 0.08], [-2250e2, 250e2])
# data_mine(path, 'Fz_upper_vs', [0, 0.08], [-400e2, 1600e2])
# data_mine(path, 'Fz_lower_vs', [0, 0.08], [-600e2, 1400e2])

LTC = {'upperVS': {}, 'lowerVS': {}}
LTC['upperVS'] = {}

# load current
current = data_load(path, 'VS3_current', date='2018_03_15')[0][0]
t = np.linspace(current['x'][0], current['x'][-1])
Ivs3 = -1 * interp1d(current['x'], current['y'])(t)  # correct LTC sign
LTC['lowerVS']['I'] = Ivs3
LTC['lowerVS']['t'] = t
LTC['upperVS']['I'] = -Ivs3  # current in global coordinates
LTC['upperVS']['t'] = t

# load line force
Fx_upper = data_load(path, 'Fr_upper_vs', date='2018_06_25')[0][0]
LTC['upperVS']['Fx'] = interp1d(Fx_upper['x'], Fx_upper['y'],
                                fill_value='extrapolate')(t)
Fz_upper = data_load(path, 'Fz_upper_vs', date='2018_06_25')[0][0]
LTC['upperVS']['Fz'] = interp1d(Fz_upper['x'], Fz_upper['y'],
                                fill_value='extrapolate')(t)
Fx_lower = data_load(path, 'Fr_lower_vs', date='2018_06_25')[0][0]
LTC['lowerVS']['Fx'] = interp1d(Fx_lower['x'], Fx_lower['y'],
                                fill_value='extrapolate')(t)
Fz_lower = data_load(path, 'Fz_lower_vs', date='2018_06_25')[0][0]
LTC['lowerVS']['Fz'] = interp1d(Fz_lower['x'], Fz_lower['y'],
                                fill_value='extrapolate')(t)

for coil in ['upperVS', 'lowerVS']:
    LTC[coil]['Bz'] = LTC[coil]['Fx'] / (4 * LTC[coil]['I'])
    LTC[coil]['Bx'] = -LTC[coil]['Fz'] / (4 * LTC[coil]['I'])

with open(join(path, 'LTC_components'), 'wb') as output:
            pickle.dump(LTC, output, -1)

with open(join(path, 'LTC_components'), 'rb') as intput:
            LTC = pickle.load(intput)

for i, coil in enumerate(['upperVS', 'lowerVS']):
    Fmag = np.sqrt(LTC[coil]['Fx']**2 + LTC[coil]['Fz']**2)
    plt.plot(1e3*LTC[coil]['t'], Fmag, 'k-')
    plt.plot(1e3*LTC[coil]['t'], LTC[coil]['Fx'], '--', color='C{}'.format(i))
    plt.plot(1e3*LTC[coil]['t'], LTC[coil]['Fz'], '-', color='C{}'.format(i))


