from amigo.pyplot import plt
import os
import nep
from amigo.png_tools import data_mine, data_load, image_plot
from amigo.IO import class_dir


path = os.path.join(class_dir(nep), '../Data/geom/')
# data_mine(path, 'vv_full', [2, 10], [-6, 6])
vv_shell = data_load(path, 'vv_full', date='2018_08_22')[0]
trs = data_load(path, 'lower_triangular_support_extended',
                date='2018_07_17')[0]

ax = plt.subplots(1, 1, figsize=(8, 10))[1]
image_plot(path, 'vv_full', date='2018_08_22', ax=ax)

for cl in vv_shell[1:]:
    x, z = cl['x'], cl['y']
    plt.plot(x, z, 'C7')

plt.plot(trs[2]['x'], trs[2]['y'], 'C7')
plt.axis('equal')
