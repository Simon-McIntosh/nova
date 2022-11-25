import numpy as np

from nova.structural.finiteframe import finiteframe, scale
from nova.plot import plt

ff = finiteframe(frame='3D')
ff.add_shape('circ', r=0.2, ro=0.1)
ff.add_mat('bar', ['steel_cast'], [ff.section])

ff.add_nodes([0, 0, 0])  # central node

R = 2
nTF = 18

for i, theta in enumerate(np.linspace(0, 2 * np.pi, nTF, endpoint=False)):
    ff.add_nodes([R * np.cos(theta), 0, R * np.sin(theta)])
    ff.add_elements(n=[0, i+1], part_name='s{:d}'.format(i), nmat='bar')
    ff.add_bc(['fix'], [0], part='s{:d}'.format(i), ends=0)
    if i > 0:
        ff.add_cp([1, i+1], dof='fix', rotate=True, axis='y')  # rotation cp

#ff.add_cp([1, 2], dof='fix', rotate=False)
#ff.add_cp([1, 5], dof='fix', rotate=True)

# ff.d(['u'],[1])

# ff.add_elements(n=[4,5],part_name='s3')
# ff.d(['fix'],[0],part='s3',ends=1)


ff.add_nodal_load(1, 'fz', 2e4)
ff.add_nodal_load(1, 'fx', 2e4)

ff.add_weight()  # add weight to all elements
# ff.add_tf_load(config,tf,sf.Bpoint,method='function')  # burst and topple

# ff.cp.add([1,2],dof=['u','v'])  # couple nodes


# ff.add_cp([1,2],dof='fix',rotate=True, axis='y')
# ff.add_cp([1,3],dof='fix',rotate=True)

ff.solve()

with scale(ff.deform, -0.5):
    ff.plot_nodes()
    ff.plot_F(factor=0.5)
    
    ff.plot_displacment()
    plt.axis('off')


# ff.plot_twin()
# ff.plot_curvature()
