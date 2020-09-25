
from nova.structural.finite_element import FE
from nova.utilities.pyplot import plt


fe = FE(frame='2D')
fe.add_material('beam', data)

fe.add_mat(0, E=1e-1, I=1e0, A=1, G=5, J=1, rho=5e-2)

fe.add_nodes([-1, 0, 0])
fe.add_nodes([0, 0, 0])
fe.add_nodes([0, 0, 0])
fe.add_nodes([1, 0, 0])

fe.add_elements(n=[0, 1], part_name='s1')
fe.add_elements(n=[2, 3], part_name='s2')


fe.add_bc(['fix'], 0, part='s1', ends=0)
fe.add_bc(['fix'], 0, part='s2', ends=1)

# fe.add_cp([1,2],dof='fix',rotate=False)
fe.add_cp([1, 2], dof='nz', rotate=False)


fe.add_nodal_load(1, 'fy', 0.25)

fe.solve()



fe.plot_nodes()
fe.plot_F(scale=5e-1)

fe.plot_displacment()
plt.axis('off')


# fe.plot_twin()
# fe.plot_curvature()
