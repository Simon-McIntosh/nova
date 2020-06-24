import numpy as np
import pylab as pl
from nova.finite_element import FE
from nova.properties import second_moment

fe = FE(frame='3D')
sm = second_moment()
sm.add_shape('circ', r=0.2, ro=0.1)
C, I, A = sm.report()
section = {'C': C, 'I': I, 'A': A, 'J': I['xx'], 'pnt': sm.get_pnt()}
fe.add_mat('bar', ['steel_cast'], [section])

fe.add_nodes([0, 0, 0])

R = 2
nTF = 16

for i, theta in enumerate(np.linspace(0, 2 * np.pi, nTF, endpoint=False)):
    fe.add_nodes([R * np.cos(theta), 0, R * np.sin(theta)])
    fe.add_elements(n=[0, i+1], part_name='s{:d}'.format(i), nmat='bar')
    fe.add_bc(['fix'], [0], part='s{:d}'.format(i), ends=0)
    if i > 0:
        fe.add_cp([1, i+1], dof='fix', rotate=True, axis='y')

#fe.add_cp([1, 2], dof='fix', rotate=False)
#fe.add_cp([1, 5], dof='fix', rotate=True)

# fe.d(['u'],[1])

# fe.add_elements(n=[4,5],part_name='s3')
# fe.d(['fix'],[0],part='s3',ends=1)


fe.add_nodal_load(1, 'fz', 2e4)
fe.add_nodal_load(1, 'fx', 2e4)

fe.add_weight()  # add weight to all elements
# fe.add_tf_load(config,tf,sf.Bpoint,method='function')  # burst and topple

# fe.cp.add([1,2],dof=['u','v'])  # couple nodes


# fe.add_cp([1,2],dof='fix',rotate=True, axis='y')
# fe.add_cp([1,3],dof='fix',rotate=True)

fe.solve()
fe.deform(scale=1e4)



fe.plot_nodes()
fe.plot_F(scale=5e-5)

fe.plot_displacment()
pl.axis('off')


# fe.plot_twin()
# fe.plot_curvature()
