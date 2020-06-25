import os
import nep
from amigo.png_tools import data_mine, data_load
from amigo.IO import class_dir
from amigo.pyplot import plt

path = os.path.join(class_dir(nep), '../Data/LTC/')
#data_mine(path, 'VS3_field', [0, 0.08], [0.32, 1.12])

points = data_load(path, 'VS3_field', date='2018_03_15')[0]

LTC = {}
LTC['lowerVS'] = {'t': points[0]['x'], 'Bmag': points[0]['y']}
LTC['upperVS'] = {'t': points[1]['x'], 'Bmag': points[1]['y']}

for coil in LTC:
    plt.plot(1e3*LTC[coil]['t'], LTC[coil]['Bmag'])

plt.despine()
plt.xlabel('$t$ ms')
plt.ylabel('$|B_p|$ T')
plt.legend()

