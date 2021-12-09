import pytest
import shapely.geometry
import numpy as np

from nova.electromagnetic.coilset import CoilSet


def test_centroid_x():
    coilset = CoilSet()
    coilset.plasma.insert([[1, 2, 2, 1, 1], [1, 1, 3, 3, 1]])
    poly = coilset.frame.poly[0]
    assert poly.centroid.x == coilset.frame.x[0] == 1.5


def test_centroid_z():
    coilset = CoilSet()
    coilset.plasma.insert([[1, 2, 1.5, 1], [0, 0, 3, 0]])
    poly = coilset.frame.poly[0]
    assert poly.centroid.y == 1 == coilset.frame.z[0]


def test_circle():
    coilset = CoilSet()
    polygon = shapely.geometry.Point(1, 1).buffer(0.5)
    coilset.plasma.insert(polygon)
    assert np.isclose(coilset.subframe.area.sum(), np.pi*0.5**2, 5e-3)


def test_plasma_turns():
    coilset = CoilSet(dplasma=0.25)
    coilset.plasma.insert({'ellipse': [1.7, 1, 0.5, 0.85]})
    assert coilset.frame.nturn[0] == 1
    assert np.isclose(coilset.subframe.nturn.sum(), 1)


def test_plasma_part():
    coilset = CoilSet(dplasma=0)
    coilset.plasma.insert({'o': [1.7, 1, 0.5]})
    assert coilset.frame.part[0] == 'plasma'


def test_polygon_separatrix():
    coilset = CoilSet(dplasma=-5000)
    coilset.plasma.insert([[1, 5, 5, 1, 1], [1, 1, 5, 5, 1]])

    from nova.geometry.polygon import Polygon
    loop = Polygon(dict(ellip=(3, 3, 4, 3))).points[:, ::2]
    coilset.plasma.update(loop)
    coilset.plasma.plot()
    coilset.plot()
    assert np.isclose(
        coilset.subframe.area[coilset.subframe.ionize].sum(), np.pi*2**2, 0.05)
    assert False
test_polygon_separatrix()

def test_array_separatrix():
    coilset = CoilSet(dplasma=0.1)
    coilset.plasma.insert([[1, 2, 2, 1, 1], [1, 1, 2, 2, 1]])
    coilset.plasma.update(np.array([[1, 2, 1.5, 1], [0, 0, 2, 0]]).T)
    assert np.isclose(
        coilset.subframe.area[coilset.subframe.ionize].sum(), 0.5**2, 0.1)


def test_separatrix_nturn():
    coilset = CoilSet(dplasma=0.5)
    coilset.plasma.insert([[1, 5, 5, 1, 1], [1, 1, 5, 5, 1]])
    coilset.plasma.update(shapely.geometry.Point(3, 3).buffer(2))
    assert coilset.loc['plasma', 'nturn'].sum() == 1


def test_polarity():
    coilset = CoilSet(dplasma=-10, dcoil=-10)
    coilset.coil.insert(4.65, [-0.3, 0.3], 0.1, 0.5)
    coilset.plasma.insert({'ellip': [5, 0, 0.5, 0.75]}, It=-15e6)
    coilset.plasma.update({'disc': [5, 0, 0.3]})
    assert coilset.plasma.polarity == -1


def test_multiframe_area():
    coilset = CoilSet(dplasma=0.5)
    coilset.plasma.insert(dict(sk=(5, 0, 2, 0.2)), name='PLedge', delta=0.2)
    coilset.plasma.insert(dict(o=(5, 0, 1.6)), name='PLcore')
    assert np.isclose(coilset.loc['plasma', 'area'].sum(), 1/4*np.pi*2**2,
                      atol=1e-3)


def test_multiframe_nturn():
    coilset = CoilSet(dplasma=0.5)
    coilset.plasma.insert(dict(sk=(5, 0, 2, 0.2)), name='PLedge', delta=0.2)
    coilset.coil.insert(6.5, 0, 0.2, 0.8)
    coilset.plasma.insert(dict(o=(5, 0, 1.6)), name='PLcore')
    assert np.isclose(coilset.loc['plasma', 'nturn'].sum(), 1)


if __name__ == '__main__':

    pytest.main([__file__])

    coilset = CoilSet(dplasma=-50)
    coilset.plasma.insert(dict(o=(5, 0, 1.6)))

    import scipy
    import sklearn
    import sklearn.mixture

    points = coilset.loc['plasma', ['x', 'z']].to_numpy()

    center = coilset.Loc['plasma', ['x', 'z']].to_numpy()
    delta = points - center
    radius = np.linalg.norm(delta, axis=1)
    angle = np.arctan2(delta[:, 1], delta[:, 0])

    radius = radius**4
    points[:, 0] = radius * np.cos(angle)
    points[:, 1] = radius * np.sin(angle)



    #norm = np.linalg.norm(points - [5, 0], axis=1).reshape(-1, 1)
    #norm = np.exp(2*norm / np.std(norm))
    #points = np.append(points, norm, axis=1)

    #

    '''
    weight = norm @ norm.T
    weight = np.exp(-beta * weight / np.std(weight))
    #weight = np.exp(weight)
    #weight /= weight.max()
    '''

    '''
    beta = 1
    distance = scipy.spatial.distance_matrix(points, points)
    distance * 18*weight# np.exp(weight)
    affinity = np.exp(-beta * distance / np.std(distance))
    '''
    #cluster = sklearn.cluster.AgglomerativeClustering(
    #    n_clusters=20)
    cluster = sklearn.cluster.KMeans(n_clusters=15, random_state=0)


    cluster.fit(points)
    labels = cluster.labels_
    '''

    cluster = sklearn.cluster.KMeans(n_clusters=5, random_state=0)

    cluster.fit(points)
    labels = cluster.labels_
    '''

    '''

    cluster = sklearn.mixture.GaussianMixture(
        n_components=3, covariance_type="full")
    cluster.fit(weight)
    labels = cluster.predict(weight)
    '''
    #coilset.link(['PLedge', 'PLcore'])

    for label in range(len(np.unique(labels))):
        coilset.subframe.polyplot(labels==label, facecolor=f'C{label%10}')
