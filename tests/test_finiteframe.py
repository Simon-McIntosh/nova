from nova.structural.finiteframe import finiteframe, scale
from nova.utilities.pyplot import plt

def test_couple(plot=False):
    ff = finiteframe(frame='2D')
    ff.add_shape('circ', r=0.02, ro=0.01)
    ff.add_mat('tube', ['steel_cast'], [ff.section])
    # nodes
    ff.add_nodes([-1, 0, 0])
    ff.add_nodes([0, 0, 0])
    ff.add_nodes([0, 0, 0])
    ff.add_nodes([1, 0, 0])
    # elements
    ff.add_elements(n=[0, 1], part_name='s1', nmat='tube')
    ff.add_elements(n=[2, 3], part_name='s2', nmat='tube')
    # boundarys
    ff.add_bc(['fix'], 0, part='s1', ends=0)
    ff.add_bc(['fix'], 0, part='s2', ends=1)
    ff.add_cp([1, 2], dof='ny', rotate=False)
    # load
    ff.add_nodal_load(1, 'fy', 0.25)
    # solve 
    ff.solve()
    # check
    assert ff.D['y'][1] == ff.D['y'][2]
    # plot
    if plot:
        with scale(ff.deform, -0.5):
            ff.plot_nodes()
            ff.plot_F(projection='xy', factor=0.25)
            ff.plot_displacment(projection='xy')
            plt.axis('off')
    

if __name__ == '__main__':
    test_couple(True)