from amigo.pyplot import plt

import numpy as np
from nep.coil_geom import VVcoils, ITERcoilset, VSgeom
from nova.coil_class import CoilClass

pf = ITERcoilset().cc
#vv = VVcoils(model='full').cc
vs = VSgeom(jacket=True).cc

cc = CoilClass(pf, vs)

cc.scenario_filename = -2
cc.scenario = 'EOF'

cc.cluster(5)

cc.merge(['LVS', 'UVS'])
cc.merge(['CS1L', 'CS1U'])

cc.plot(subcoil=False, label=True)

#cc.solve_grid(plot=True)


'''
for i in range(vv.coil.cluster_index.max()+1):
    print(i)
    index = vv.coil.index[vv.coil.cluster_index == i]
    plt.plot(vv.coil.loc[index, 'x'], vv.coil.loc[index, 'z'], f'C{i}o')
'''