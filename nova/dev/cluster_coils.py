from amigo.pyplot import plt

import numpy as np
from nep.coil_geom import VVcoils, ITERcoilset, VSgeom
from nova.coil_class import CoilClass

pf = ITERcoilset().cc
#vv = VVcoils(model='full').cc
vs = VSgeom(jacket=True).cc

cc = CoilClass(vs, pf)

cc.scenario_filename = -2
cc.scenario = 'EOF'

cc.cluster(5)

cc.merge(['LVS', 'UVS'])
# cc.merge(['CS1L', 'CS1U'])

# cc.plot(subcoil=False, label=True)

cc.calculate_inductance()

L = np.diag(cc.inductance['Mc'])
LL = np.dot(L.reshape(-1, 1), L.reshape(1, -1))
m = cc.inductance['Mc'] / np.sqrt(LL)

#cc.inductance['Mc'][m>0.5] = 0.5
print(cc.inductance['Mc'])
print(m)
#cc.solve_grid(plot=True)

'''
for i in range(vv.coil.cluster_index.max()+1):
    print(i)
    index = vv.coil.index[vv.coil.cluster_index == i]
    plt.plot(vv.coil.loc[index, 'x'], vv.coil.loc[index, 'z'], f'C{i}o')
'''