
from nova.structural.finiteframe import finiteframe, scale
from nova.plot import plt


ff = finiteframe(frame='2D')

#ff.add_material('beam', data)
#ff.add_mat(0, E=1e-1, I=1e0, A=1, G=5, J=1, rho=5e-2)

ff.add_shape('circ', r=0.02, ro=0.01)
ff.add_mat('tube', ['steel_cast'], [ff.section])

ff.add_nodes([-1, 0, 0])
ff.add_nodes([0, 0, 0])
ff.add_nodes([0, 0, 0])
ff.add_nodes([1, 0, 0])

ff.add_elements(n=[0, 1], part_name='s1', nmat='tube')
ff.add_elements(n=[2, 3], part_name='s2', nmat='tube')


ff.add_bc(['fix'], 0, part='s1', ends=0)
ff.add_bc(['fix'], 0, part='s2', ends=1)

#ff.add_cp([1,2], dof='fix',rotate=False)
ff.add_cp([1, 2], dof='ny', rotate=False)
#ff.add_cp([1, 2], dof='nz', rotate=False)

ff.add_nodal_load(1, 'fy', 0.25)


ff.solve()


with scale(ff.deform, -0.5):
    ff.plot_nodes()
    ff.plot_F(projection='xy', factor=0.25)
    
    ff.plot_displacment(projection='xy')
    plt.axis('off')


# ff.plot_twin()
# ff.plot_curvature()
