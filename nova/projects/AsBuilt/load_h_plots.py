import os

import numpy as np

from nova.definitions import root_dir
from nova.utilities.png_tools import data_mine, data_load
from nova.utilities.pyplot import plt

path = os.path.join(root_dir, 'input/Assembly/')
#data_mine(path, 'peaktopeak_case1', xlim=(0, 2*np.pi), ylim=(0, 6))
#data_mine(path, 'peaktopeak_case2', xlim=(0, 2*np.pi), ylim=(-4, 0))
#data_mine(path, 'peaktopeak_case3', xlim=(0, 2*np.pi), ylim=(-3, 3))
data_mine(path, 'peaktopeak_a1', xlim=(0, 2*np.pi), ylim=(-1.5, 0.5))
data_mine(path, 'peaktopeak_a2', xlim=(0, 2*np.pi), ylim=(-1.5, 0.5))
#data_mine(path, 'peaktopeak_k1', xlim=(0, 2*np.pi), ylim=(-1.5, 0.5))
#data_mine(path, 'peaktopeak_k2', xlim=(0, 2*np.pi), ylim=(-1.5, 0.5))
#data_mine(path, 'peaktopeak_k3', xlim=(0, 2*np.pi), ylim=(-1.5, 0.5))
#data_mine(path, 'peaktopeak_k4', xlim=(0, 2*np.pi), ylim=(-1.5, 0.5))
#data_mine(path, 'peaktopeak_k5', xlim=(0, 2*np.pi), ylim=(-1.5, 0.5))
#data_mine(path, 'peaktopeak_k6', xlim=(0, 2*np.pi), ylim=(-1.5, 0.5))
#data_mine(path, 'peaktopeak_k7', xlim=(0, 2*np.pi), ylim=(-1.5, 0.5))
#data_mine(path, 'peaktopeak_k8', xlim=(0, 2*np.pi), ylim=(-1.5, 0.5))
#data_mine(path, 'peaktopeak_k9', xlim=(0, 2*np.pi), ylim=(-1.5, 0.5))

#data = data_load(path, 'peaktopeak_k2', date='2022_04_15')[0][0]
#plt.plot(data['x'], data['y'])
