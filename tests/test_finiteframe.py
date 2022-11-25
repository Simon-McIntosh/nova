import numpy as np

from nova.structural.finiteframe import finiteframe, scale
from nova.structural.catenary import catenary
from nova.plot import plt


def test_couple(plot=False):
    'test node to node coupling (add_cp)'
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
            
def test_rotational_couple(plot=False):
    ff = finiteframe(frame='3D')
    ff.add_shape('circ', r=0.2, ro=0.1)
    ff.add_mat('bar', ['steel_cast'], [ff.section])
    # central node
    ff.add_nodes([0, 0, 0])  
    # radius and beam number
    R = 2
    nTF = 18
    # space beams around azimuth
    for i, theta in enumerate(np.linspace(0, 2 * np.pi, nTF, endpoint=False)):
        ff.add_nodes([R * np.cos(theta), 0, R * np.sin(theta)])
        ff.add_elements(n=[0, i+1], part_name='s{:d}'.format(i), nmat='bar')
        ff.add_bc(['fix'], [0], part='s{:d}'.format(i), ends=0)
        if i > 0:
            # add rotation cp linking back to primary
            ff.add_cp([1, i+1], dof='fix', rotate=True, axis='y')  
    # specify dummy loads
    ff.add_nodal_load(1, 'fx', 4e4)
    ff.add_nodal_load(2, 'fz', 8e4)
    ff.solve()
    # check tangential dissplacment
    tangential = np.array([f @ dz for f, dz in zip(
            ff.el['dz'], 
            np.array([ff.D['x'], ff.D['y'], ff.D['z']])[:, 1:].T)])
    assert np.isclose(tangential, tangential[0]).all()
    if plot:
        with scale(ff.deform, -0.35):
            ff.plot_nodes()
            ff.plot_F(factor=0.5)
            ff.plot_displacment()
            plt.axis('off')
    return ff
            
def test_catenary(plot=False):
    'test catenary with constant horizontal tension - low curvature'
    L, Lo = 1, 1.5
    cat = catenary(N=51)
    cat.solve('elastic', L, Lo)
    if plot:
        cat.plot(scale_factor=-0.2, projection='xy')
        cat.plot_moment()
    assert np.max(np.abs(cat.part['chain']['d2u'][:, 1])) < 1e-6
            
def test_xy_plane_3D(plot=False):
    'ensure symetric behavior in xy plane with 3D frame elements'
    ff = finiteframe(frame='3D')
    ff.add_shape('circ', r=0.02, ro=0.01)
    ff.add_mat('tube', ['steel_cast'], [ff.section])
    # nodes
    X = np.zeros((21, 3))
    X[:, 0] = np.linspace(-1, 1, len(X))
    ff.add_nodes(X)
    # elements
    ff.add_elements(nmat='tube', part_name='tube')
    # boundarys
    ff.add_bc(['fix'], 0, part='tube', ends=0)
    ff.add_bc(['fix'], -1, part='tube', ends=1)
    # load
    ff.add_weight([0, -1, 0])  # gravity in y-dir
    # solve 
    ff.solve()
    # check
    assert np.allclose(ff.D['y'][:10], ff.D['y'][11:][::-1])
    # plot
    if plot:
        with scale(ff.deform, -0.5):
            ax = plt.subplots(1, 1)[1]
            ff.plot_nodes(ax=ax)
            ff.plot_F(projection='xy', factor=0.25)
            ff.plot_displacment(projection='xy')
            plt.axis('off')

def test_displacment_constraints(plot=False):
    'test displacment constraints'
    ff = finiteframe(frame='3D')
    ff.add_shape('circ', r=0.02, ro=0.01)
    ff.add_mat('tube', ['steel_cast'], [ff.section])
    # nodes
    X = np.zeros((5, 3))
    X[:, 0] = np.linspace(-1, 1, len(X))
    ff.add_nodes(X)
    # elements
    ff.add_elements(nmat='tube', part_name='tube')
    # boundarys
    ff.add_bc(['fix'], 0, part='tube', ends=0)
    ff.add_bc(['fix'], -1, part='tube', ends=1)
    # add displacment constraints in x,y,z
    ff.add_bc(['u'], 2, part='tube', ends=0, d=-0.005)
    ff.add_bc(['v'], 1, part='tube', ends=0, d=0.01)
    ff.add_bc(['w'], 3, part='tube', ends=0, d=-0.03)
    # load
    ff.add_nodal_load(2, 'fx', 50)
    ff.add_nodal_load(3, 'fy', 100)
    ff.add_weight([0, -1, 0])
    # solve 
    ff.solve()
    # check
    assert np.allclose(np.array([-0.005, 0.01, -0.03]), 
                       np.array([ff.D['x'][2], ff.D['y'][1], ff.D['z'][3]]))
    # plot
    if plot:
        with scale(ff.deform, -0.5):
            ax = plt.subplots(1, 1)[1]
            ff.plot_nodes(ax=ax)
            ff.plot_F(projection='xy', factor=0.25)
            ff.plot_displacment(projection='xy')
            plt.axis('off')
    
    

if __name__ == '__main__':
    #test_couple(True)
    #test_catenary(True)
    #test_xy_plane_3D(True)
    #test_displacment_constraints(True)
    ff = test_rotational_couple(True)