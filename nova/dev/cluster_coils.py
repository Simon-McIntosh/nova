from amigo.pyplot import plt

import numpy as np
from nep.coil_geom import VVcoils, ITERcoilset
from nova.coil_class import CoilClass

pf = ITERcoilset().cc
vv = VVcoils(model='local', read_txt=False).cc

cc = CoilClass(pf, vv)

cc.scenario_filename = -1
cc.scenario = 'EOF'

cc.cluster()

cc.plot(subcoil=True, color_label='cluster_index')


'''
for i in range(vv.coil.cluster_index.max()+1):
    print(i)
    index = vv.coil.index[vv.coil.cluster_index == i]
    plt.plot(vv.coil.loc[index, 'x'], vv.coil.loc[index, 'z'], f'C{i}o')
'''