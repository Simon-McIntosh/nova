import os

import numpy as np

from nova.definitions import root_dir
from nova.utilities.png_tools import data_mine, data_load
from nova.utilities.pyplot import plt

path = os.path.join(root_dir, 'input/Assembly/')
#data_mine(path, 'peaktopeak_case1', xlim=(0, 2*np.pi), ylim=(0, 6))
#data_mine(path, 'peaktopeak_case2', xlim=(0, 2*np.pi), ylim=(-4, 0))
#data_mine(path, 'peaktopeak_case3', xlim=(0, 2*np.pi), ylim=(-3, 3))

data = data_load(path, 'peaktopeak_case3', date='2022_04_01')[0][0]
plt.plot(data['x'], data['y'])
