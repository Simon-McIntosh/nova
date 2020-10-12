"""A library of geometric helper functions derived from Amigo."""

import collections
from itertools import count

import numpy as np
import scipy as sp
from scipy.interpolate import interp1d
from scipy.interpolate import UnivariateSpline as spline
from matplotlib.collections import PolyCollection
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d import art3d
from shapely.geometry import Polygon, LineString, MultiPoint
from scipy.signal import savgol_filter
from scipy.spatial import ConvexHull

from rdp import rdp
from pyquaternion import Quaternion

from nova.utilities.pyplot import plt


def rms(x, Nt):
    """Return turn weighted root mean squared distance."""
    return np.sqrt(np.sum((x*Nt)**2) / np.sum(abs(Nt)))


def gmd(x, Nt):
    """Return turn weighted geometric mean distance."""
    return np.exp(np.sum(abs(Nt) * np.log(x)) / np.sum(abs(Nt)))


def amd(x, Nt):
    """Return turn weighted arithmetic mean distance."""
    return np.sum(abs(Nt) * x) / np.sum(abs(Nt))


def rdp_extract(x, y, eps=1e-2, dx=None, dx_window=None,
                scale=None, plot=False):
    """
    Extract features from a 2d curve via the rdp algorithm.

    Atributes
    ---------
        x (np.array): x-coordinate data
        y (np.array): y-corrdinate data
        eps (float): acceptible deviation expressed as fraction of y extent
        dx (float): step width for x,y subsampling
        scale (float): xo = x / scale (modify aspect of input data)
    """
    if dx_window is not None:
        step = x[1] - x[0]
        y = lowpass(y, step, dt_window=dx_window, polyorder=2)
    if scale is None:
        scale = (np.max(x)-np.min(x)) / (np.max(y)-np.min(y))
    if dx is None:
        xo, yo = x, y
    else:
        n = int((x[-1]-x[0]) / dx) + 1
        xo = [x[np.argmin(abs(x-x_))] for x_ in np.linspace(x[0], x[-1], n)]
        xo = np.unique(xo)
        yo = interp1d(x, y)(xo)
    epsilon = eps * (np.max(y)-np.min(y))
    mask = rdp(np.vstack((xo/scale, yo)).T, epsilon=epsilon,
               return_mask=True).T
    x_rdp, y_rdp = xo[mask], yo[mask]
    if plot:
        plt.plot(xo, yo, 'C0')
        for i in range(len(x_rdp)-1):
            plt.plot(x_rdp[i:i+2], y_rdp[i:i+2], f'C{6+i%2}o-')
    return x_rdp, y_rdp


def reflect(v, x):
    # compute Householder matrix from plane's normal
    # x shape [:, n]
    n = len(v)  # dimension
    vhat = v / np.linalg.norm(v)  # ensure unit vector
    P = np.identity(n) - 2*np.outer(vhat, vhat)
    xr = np.dot(x, P)
    return xr


def turning_points(array):
    # https://stackoverflow.com/questions/19936033/
    # finding-turning-points-of-an-array-in-python
    ''' turning_points(array) -> min_indices, max_indices
    Finds the turning points within an 1D array
    and returns the indices of the minimum and
    maximum turning points in two separate lists.
    '''
    idx_max, idx_min = [], []
    if (len(array) < 3):
        return idx_min, idx_max

    neutral, rising, falling = range(3)

    def get_state(a, b):
        if a < b:
            return rising
        if a > b:
            return falling
        return neutral

    ps = get_state(array[0], array[1])
    begin = 1
    for i in range(2, len(array)):
        s = get_state(array[i - 1], array[i])
        if s != neutral:
            if ps != neutral and ps != s:
                if s == falling:
                    idx_max.append((begin + i - 1) // 2)
                else:
                    idx_min.append((begin + i - 1) // 2)
            begin = i
            ps = s
    return idx_min, idx_max


def trapz2d(z, x=None, y=None, dx=1., dy=1.):
    ''' Integrate a regularly spaced 2D grid using the composite trapezium rule
    IN:
       z : 2D array
       x : (optional) grid values for x (1D array)
       y : (optional) grid values for y (1D array)
       dx: if x is not supplied, set it to the x grid interval
       dy: if y is not supplied, set it to the x grid interval
    '''
    if x is not None:
        dx = (x[-1]-x[0])/(np.shape(x)[0]-1)
    if y is not None:
        dy = (y[-1]-y[0])/(np.shape(y)[0]-1)

    s1 = z[0, 0] + z[-1, 0] + z[0, -1] + z[-1, -1]
    s2 = np.sum(z[1:-1, 0]) + np.sum(z[1:-1, -1]) +\
        np.sum(z[0, 1:-1]) + np.sum(z[-1, 1:-1])
    s3 = np.sum(z[1:-1, 1:-1])
    return 0.25*dx*dy*(s1 + 2*s2 + 4*s3)


class shape:
    '''
    flatten / reshape leading dimensions of an ndarray
    '''
    def __init__(self, input_array):
        self.input_shape = np.shape(input_array)
        self.ndim = len(self.input_shape)
        self.n = np.prod(self.input_shape)
        self.n2d = np.prod(self.input_shape[:-1])

    def flatten(self, array):
        array = array.reshape(-1, self.input_shape[-1])
        return array

    def shape(self, array, shape):
        array = array.reshape(shape)
        return array

    def reshape(self, array):
        array = array.reshape(self.input_shape)
        return array


def unique2D(Xo, eps=1e-1, bound=None):
    # returns list and index of unique points
    Xu, index = [], []
    for i, xo in enumerate(Xo):
        append = True
        if bound is None:
            inbound = True
        else:
            inbound = np.array([xo[0] > (bound[0] + eps/2),
                                xo[0] < bound[1] - eps/2,
                                xo[1] > bound[2] + eps/2,
                                xo[1] < bound[3] - eps/2]).all()
        if inbound:
            for xu in Xu:
                dr = np.sqrt((xu[0]-xo[0])**2 + (xu[1]-xo[1])**2)
                if dr < eps:
                    append = False
                    break
        if append and inbound:
            Xu.append(list(xo))
            index.append(i)
    Xu = np.array(Xu)
    return index, Xu


def grid(n, limit, eqdsk=False):
    if len(np.shape(limit)) > 1:
        limit = np.array(limit).flatten()
    xo, zo = limit[:2], limit[2:]
    try:  # n ([nx, nz])
        nx, nz = n
    except TypeError:  # n (int)
        dxo, dzo = (xo[-1] - xo[0]), (zo[-1] - zo[0])
        ar = dxo / dzo
        nz = np.max([int(np.sqrt(n / ar)), 3])
        nx = np.max([int(n / nz), 3])
    x = np.linspace(xo[0], xo[1], nx)
    z = np.linspace(zo[0], zo[1], nz)
    x2d, z2d = np.meshgrid(x, z, indexing='ij')
    if eqdsk:
        return {'x2d': x2d, 'z2d': z2d, 'x': x, 'z': z, 'nx': nx, 'nz': nz}
    else:
        return x2d, z2d, x, z, nx, nz


def patch_corner(X):
    # find corners for a 2D patch of data points shape (N, 2)
    ch = ConvexHull(X)  # rail's convex hull
    nedge = len(ch.vertices)  # number of edge points
    verts = np.append(ch.vertices, ch.vertices[0])  # first
    verts = np.append(ch.vertices[-2], verts)  # last
    turn = np.zeros(nedge)  # turning angle (cos(theta))
    for i in range(nedge):
        no = X[verts[i+1]] - X[verts[i]]  # previous edge
        n1 = X[verts[i+2]] - X[verts[i+1]]  # next edge
        turn[i] = np.dot(no, n1) / (np.linalg.norm(no) * np.linalg.norm(n1))
    corner_index = np.arange(nedge)[turn < 0.9]
    c_index = ch.vertices[corner_index]  # convert back to X index
    return c_index


def patch(X, plot=False):
    ch = ConvexHull(X)  # rail's convex hull
    mask = rdp(X[ch.vertices, :], epsilon=1e-4, return_mask=True)  # simplify
    index = ch.vertices[mask]
    if plot:
        plt.plot(X[:, 0], X[:, 1], '.')
        plt.plot(X[ch.vertices, 0], X[ch.vertices, 1], 'C2.')
        plt.plot(X[index, 0], X[index, 1], 'C1o-')
    return index


def Jcalc(dx, dy):  # torsional constant J = beta dx*dy**3 (dx > dy)
    edge = np.array([dx, dy])
    index = dy > dx
    edge[:, index] = edge[::-1, index]  # flip
    beta = 1/3 - 0.21*edge[1]/edge[0] * (1 - edge[1]**4/(12*edge[0]**4))
    return beta*dx*dy**3


def Kcalc(R, col):  # K coefficents for rectangular beam torsion
    r = np.array([1, 1.2, 1.5, 2, 2.5, 3, 4, 5, 10, 100, 101])  # dx/dy, dx>dy
    if col == 0:
        k = np.array([0.675, 0.759, 0.848, 0.93, 0.968, 0.985,
                      0.997, 0.999, 1, 1, 1])
    elif col == 1:
        k = np.array([0.1406, 0.166, 0.196, 0.229, 0.249, 0.263,
                      0.281, 0.291, 0.312, 0.333, 0.333])
    elif col == 2:
        k = np.array([0.208, 0.219, 0.231, 0.246, 0.258, 0.267,
                      0.282, 0.291, 0.312, 0.333, 0.333])
    else:
        raise ValueError('k not in range(3)')
    K = interp1d(r, k, fill_value='extrapolate')(R)
    return K


def three_point_arc(p1, p2, p3):
    '''
    return center and radius for arc formed from three points
    http://stackoverflow.com/questions/20314306/
    find-arc-circle-equation-given-three-points-in-space-3d
    '''
    a = np.linalg.norm(p3 - p2)
    b = np.linalg.norm(p3 - p1)
    c = np.linalg.norm(p2 - p1)
    s = (a + b + c) / 2
    radius = a*b*c / 4 / np.sqrt(s * (s - a) * (s - b) * (s - c))
    b1 = a*a * (b*b + c*c - a*a)
    b2 = b*b * (a*a + c*c - b*b)
    b3 = c*c * (a*a + b*b - c*c)
    centre = np.column_stack((p1, p2, p3)).dot(np.hstack((b1, b2, b3)))
    centre /= np.array(b1 + b2 + b3)
    return centre, radius


def elipsoid(theta, xo, zo, A, k, delta):
    X = xo + xo / A * np.cos(theta + delta * np.sin(theta))
    Z = zo + xo / A * k * np.sin(theta)
    return X, Z


def loop_area(X, Z, plot=False):
    print('You are using geom.loop_area. Consider using geom.polyarea /'
          'instead :).')
    X, Z = theta_sort(X, Z, origin='top')
    Z -= np.min(Z)
    imin, imax = np.argmin(X), np.argmax(X)
    Rb = np.append(X[imin:], X[:imax + 1])
    Zb = np.append(Z[imin:], Z[:imax + 1])
    Rt = X[imax:imin + 1][::-1]
    Zt = Z[imax:imin + 1][::-1]
    At = np.trapz(Zt, Rt)
    Ab = np.trapz(Zb, Rb)
    A = At - Ab
    if plot:
        plt.plot(X[0], Z[0], 'bo')
        plt.plot(X[20], Z[20], 'bd')
        plt.plot(Rt, Zt, 'bx-')
        plt.plot(Rb, Zb, 'gx-')
    return A


def loop_vol(X, Z):
    A = polyarea(X, Z)
    c = get_centroid(X, Z)
    return A*2*np.pi*c[0]


def get_centroid(X, Z):
    # Non-self-intersecting counterclockwise polygon
    A = polyarea(X, Z)
    cx, cy = 0, 0
    for i in range(len(X)-1):
        a = X[i]*Z[i+1]-X[i+1]*Z[i]
        cx += (X[i]+X[i+1])*a
        cy += (Z[i]+Z[i+1])*a
    cx /= 6*A
    cy /= 6*A
    return cx, cy


def vol_calc(X, Z):
    print('Warning: geom.vol_calc is deprecated.')
    print('Use geom.loop_vol inplace')
    dX = np.diff(X)
    dZ = np.diff(Z)
    V = 0
    for x, dx, dz in zip(X[:-1], dX, dZ):
        V += np.abs((x + dx / 2)**2 * dz)
    V *= np.pi
    return V


def poly_inloop(loop, point, plot=False, ax=None):
    poly = Polygon(np.array([loop['x'], loop['z']]).T)
    points = MultiPoint(points=list(zip(point['x'], point['z'])))
    if poly.intersects(points):
        interior_multi_point = poly.intersection(points)
        multi_point = np.asarray(interior_multi_point)
        if len(multi_point) == 2:
            interior_points = {'x': multi_point[0], 'z': multi_point[1]}
        else:
            interior_points = {'x': [point[0] for point in multi_point],
                               'z': [point[1] for point in multi_point]}
    else:
        interior_points = {'x': None, 'z': None}
    if plot:
        if ax is None:
            ax = plt.subplots(1, 1, figsize=(8, 6))[1]
        ax.plot(loop['x'], loop['z'], zorder=10)
        ax.plot(point['x'], point['z'], '.')
        ax.plot(interior_points['x'], interior_points['z'], '.')
        plt.axis('equal')
        plt.axis('off')
    return interior_points['x'], interior_points['z']


def polyline(loop, point, plot=False):
    # repurposed from M. Colemann - mattitools
    '''
    loop = x,z dict
    points = x,z dict len > 1
    '''
    poly = Polygon(np.array([loop['x'], loop['z']]).T)
    points = np.array([[point['x'][0], point['z'][0]],
                       [point['x'][-1], point['z'][-1]]])
    cross = poly.exterior.intersection(LineString(points))[1]
    if plot:
        plt.plot(loop['x'], loop['z'])
        plt.plot(points.T[0], points.T[1])
        plt.plot(cross.x, cross.y, 'o')
    return cross.x, cross.y


def unique(X, Z):
    L = length(X, Z)
    io = np.append(np.diff(L) > 0, True)  # remove duplicates
    return X[io], Z[io], L[io]


def order(X, Z, anti=True):
    rc, zc = (np.mean(X), np.mean(Z))
    theta = np.unwrap(np.arctan2(Z - zc, X - rc))
    if theta[-2] < theta[0]:
        X, Z = X[::-1], Z[::-1]
    if not anti:
        X, Z = X[::-1], Z[::-1]
    return X, Z


def clock(X, Z, reverse=True):  # order loop points in anti-clockwise direction
    rc, zc = (np.mean(X), np.mean(Z))
    radius = ((X - rc)**2 + (Z - zc)**2)**0.5
    theta = np.arctan2(Z - zc, X - rc)
    index = theta.argsort()[::-1]
    radius, theta = radius[index], theta[index]
    X, Z = rc + radius * np.cos(theta), zc + radius * np.sin(theta)
    X, Z = np.append(X, X[0]), np.append(Z, Z[0])
    X, Z = xzSLine(X, Z, npoints=len(X) - 1)
    if reverse:
        X, Z = X[::-1], Z[::-1]
    return X, Z


def theta_sort(X, Z, origin='lfs', **kwargs):
    xo = kwargs.get('xo', (np.mean(X), np.mean(Z)))
    anti = kwargs.get('anti', True)  # changed from False
    if origin == 'lfs':
        theta = np.arctan2(Z - xo[1], X - xo[0])
    elif origin == 'top':
        theta = np.arctan2(xo[0] - X, Z - xo[1])
    elif origin == 'bottom':
        theta = np.arctan2(X - xo[0], xo[1] - Z)
    if kwargs.get('unwrap', False):
        theta = np.unwrap(theta)
    index = np.argsort(theta)
    X, Z = X[index], Z[index]
    if not anti:
        X, Z = X[::-1], Z[::-1]
    return X, Z


def rt(X, Z, xo):
    theta = np.unwrap(np.arctan2(Z - xo[1], X - xo[0]))
    radius = np.sqrt((Z - xo[1])**2 + (X - xo[0])**2)
    index = np.argsort(theta)
    radius, theta = radius[index], theta[index]
    return radius, theta


def xz(radius, theta, xo):
    X = xo[0] + radius * np.cos(theta)
    Z = xo[1] + radius * np.sin(theta)
    return X, Z


def xzSpline(X, Z, xo, npoints=500, w=None, s=0.005):
    radius, theta = rt(X, Z, xo)
    Ts = np.linspace(theta[0], theta[-1], npoints)
    if w is None:
        radius = spline(theta, radius, s=s)(Ts)
    else:
        radius = spline(theta, radius, w=w, s=s)(Ts)
    Rs, Zs = xz(radius, Ts, xo)
    return Rs, Zs, Ts


def xzSLine(X, Z, npoints=500, s=0, Hres=False):
    L = length(X, Z)
    if Hres:
        npoints *= 10
    Linterp = np.linspace(0, 1, int(npoints))
    if s == 0:
        X = interp1d(L, X)(Linterp)
        Z = interp1d(L, Z)(Linterp)
    else:
        X = spline(L, X, s=s)(Linterp)
        Z = spline(L, Z, s=s)(Linterp)
    return X, Z


def xzInterp(X, Z, npoints=500, ends=True):
    L = length(X, Z)
    Linterp = np.linspace(0, 1, npoints, endpoint=ends)
    X = interp1d(L, X)(Linterp)
    Z = interp1d(L, Z)(Linterp)
    return X, Z


def xyzInterp(X, npoints=500, ends=True):
    L = vector_length(X)
    Linterp = np.linspace(0, 1, npoints, endpoint=ends)
    if npoints == 1:
        Linterp = 0.5
    Xinterp = np.zeros((npoints, 3))
    for i, x in enumerate(X.T):
        Xinterp[:, i] = interp1d(L, x)(Linterp)
    return Xinterp


def xzfun(X, Z):  # return interpolation functions
    L = length(X, Z)
    X = interp1d(L, X)
    Z = interp1d(L, Z)
    return X, Z


def xzCirc(X, Z):
    radius, theta = rt(X, Z)
    X, Z = xz(radius, theta)
    return X, Z


def length(X, Z, norm=True):
    L = np.append(0, np.cumsum(np.sqrt(np.diff(X)**2 + np.diff(Z)**2)))
    if norm:
        L = L / L[-1]
    return L


def vector_length(X, norm=True):
    L = np.append(0, np.cumsum(np.sqrt(np.diff(X[:, 0])**2 +
                                       np.diff(X[:, 1])**2 +
                                       np.diff(X[:, 2])**2)))
    if norm:
        L = L / L[-1]
    return L


def vector_SLine(X, s=0, Hres=False):
    L = vector_length(X)
    npoints = len(X)
    # Linterp = np.linspace(0, 1, npoints)
    Xi = np.zeros((npoints, 3))
    if s == 0:
        for i in range(3):
            Xi[:, i] = interp1d(L, X[:, i])(L)
    else:
        for i in range(3):
            Xi[:, i] = spline(L, X[:, i], s=s)(L)
    return Xi


def lowpass(x, dt, dt_window=1, polyorder=3):
    nwindow = int(dt_window/dt)
    if nwindow % 2 == 0:
        nwindow += 1
    x_lp = savgol_filter(x, nwindow, polyorder=polyorder,
                         mode='mirror')  # lowpass
    return x_lp


def vector_lowpass(X, window_length=5, ployorder=3):
    # apply moving window filter to nD line data
    mn = np.shape(X)
    if len(mn) == 1:
        m = mn[0]
        n = 1
        X = X.reshape(-1, 1)
    else:
        m, n = mn
    Xf = np.zeros((m, n))
    for i in range(n):
        Xf[:, i] = savgol_filter(X[:, i], window_length, ployorder,
                                 mode='mirror')
    return Xf


def space(X, Z, npoints):
    L = length(X, Z)
    le = np.linspace(0, 1, npoints)
    X, Z = interp1d(L, X)(le), interp1d(L, Z)(le)
    return X, Z


def rotate(theta, axis='z'):
    if axis == 'z':
        X = np.array([[np.cos(theta), -np.sin(theta), 0],
                      [np.sin(theta), np.cos(theta), 0],
                      [0, 0, 1]])
    elif axis == 'y':
        X = np.array([[np.cos(theta), 0, -np.sin(theta)],
                      [0, 1, 0],
                      [np.sin(theta), 0, np.cos(theta)]])
    elif axis == 'x':
        X = np.array([[1, 0, 0],
                      [0, np.cos(theta), -np.sin(theta)],
                      [0, np.sin(theta), np.cos(theta)]])
    else:
        errtxt = 'incorrect roation axis {}'.format(axis)
        errtxt += ', select from [\'x\',\'y\',\'z\']'
        raise ValueError(errtxt)
    return X


def rotate2D(theta, xo=0, yo=0, dx=0, dy=0):
    X = np.array([[np.cos(theta), -np.sin(theta), 0],
                  [np.sin(theta), np.cos(theta), 0],
                  [0, 0, 1]])
    To = np.array([[1, 0, -xo], [0, 1, -yo], [0, 0, 1]])
    T1 = np.array([[1, 0, xo + dx], [0, 1, yo + dy], [0, 0, 1]])
    TRT = np.dot(T1, np.dot(X, To))[:2, :]
    return X[:2, :2], TRT


def rotate_vector2D(V, theta):
    Vmag = np.sqrt(V[0]**2 + V[1]**2)
    X = rotate2D(theta)[0]
    Vhat = (X * np.matrix([[V[0]], [V[1]]])).getA1() / Vmag
    return Vhat


def qrotate(point, **kwargs):
    '''
    rotate point cloud by angle theta around vector
    right-hand coordinates
    kwargs defines rotation vector using:
        pair of points, p1=[x,y,z] and p2=[x,y,z]
        origin and vector, xo=[x,y,z] and dx=[x,y,z]
        vector dx=[x,y,z], assumes xo=[0,0,0]
        quaternion quart=Quaternion object

    points as numpy.array size=(:,3) or dict with entries 'x','y','z'
    theta=float (radian)

    requires ether:
    numpy quaternion dtype:
    conda install -c moble quaternion
    https://github.com/moble/quaternion/blob/master/
    or
    pyquaternion
    pip install pyquaternion

    From quaternion readme:
    Euler angles are awful
    Euler angles are pretty much the worst things ever and it makes me feel
    bad even supporting them. Quaternions are faster, more accurate,
    basically free of singularities, more intuitive, and generally
    easier to understand.

    Matti - amigo.geom curently uses Euler angles. This makes me feel sad.
    Not enough time now, but will need to be addresed in the future...
    '''
    if 'quart' in kwargs:
        quart = kwargs['quart']
        xo = kwargs.get('xo', np.zeros(3))
    else:
        theta = kwargs['theta']
        if 'p1' in kwargs and 'p2' in kwargs:
            p1, p2 = kwargs['p1'], kwargs['p2']
            if not isinstance(p1, np.ndarray) or\
                    not isinstance(p1, np.ndarray):
                p1, p2 = np.array(p1), np.array(p2)
            xo = p1
            dx = p2-p1
            dx = tuple(dx)
        elif 'xo' in kwargs and 'dx' in kwargs:
            xo, dx = kwargs['xo'], kwargs['dx']
        elif 'dx' in kwargs:
            dx = kwargs['dx']
            if isinstance(dx, str):
                index = ['x', 'y', 'z'].index(dx)
                dx = np.zeros(3)
                dx[index] = 1
            xo = np.zeros(3)
        else:
            errtxt = 'error in kwargs input\n'
            errtxt += 'rotation vector input as ether:\n'
            errtxt += '\tpair of points, p1=[x,y,z] and p2=[x,y,z]\n'
            errtxt += '\torigin and vector, xo=[x,y,z] and dx=[x,y,z]\n'
            raise ValueError(errtxt)
        dx /= np.linalg.norm(dx)  # normalise rotation axis
        quart = Quaternion(axis=dx, angle=theta)
    if isinstance(point, dict):
        isdict = True
        p = np.zeros((len(point['x']), 3))
        for i, var in enumerate(['x', 'y', 'z']):
            p[:, i] = point[var]
        point = p
    else:
        isdict = False
    if np.ndim(point) == 1 and len(point) == 3:
        point = np.array([point])
    if np.shape(point)[1] != 3:
        errtxt = 'point vector required as numpy.array size=(:,3)'
        raise ValueError(errtxt)
    trans = np.ones((len(point), 1))*xo  # expand vector origin
    p = point - trans  # translate to rotation vector's origin (xo)
    rpoint = np.zeros(np.shape(point))
    for i, po in enumerate(p):
        rpoint[i, :] = quart.rotate(po)
    rpoint += trans  # translate from rotation vector's origion (xo)
    if isdict:  # return to dict
        p = {}
        for i, var in enumerate(['x', 'y', 'z']):
            p[var] = rpoint[:, i]
        rpoint = p
    return rpoint


def tangent(X, Z):  # loop tangents in 2D plane
    dX, dZ = np.gradient(X), np.gradient(Z)
    mag = np.sqrt(dX**2 + dZ**2)
    index = mag > 0
    dX, dZ, mag = dX[index], dZ[index], mag[index]  # clear duplicates
    X, Z = X[index], Z[index]
    tX, tZ = dX / mag, dZ / mag
    return tX, tZ


def normal(X, Z):  # loop normals in 2D plane
    X, Z = np.array(X), np.array(Z)
    tX, tZ = tangent(X, Z)
    t = np.zeros((len(tX), 3))
    t[:, 0], t[:, 1] = tX, tZ
    n = np.cross(t, [0, 0, 1])
    nR, nZ = n[:, 0], n[:, 1]
    return nR, nZ, X, Z


def cross_vectors(po, to, p1, t1):
    '''
    calculates the crossing point of two 2D vectors
    if lines are coincident then returns the midpoint of p1-po
    if lines are parralel and non-coincident then raises error
    '''
    to_hat = to / np.linalg.norm(to)
    t1_hat = t1 / np.linalg.norm(t1)

    c1o = np.cross(t1_hat, to_hat)
    c1o_norm = np.linalg.norm(c1o)
    if c1o_norm < 1e-3:  # lines parralel
        if np.linalg.norm(np.cross(p1-po, to_hat)) < 1e-3:  # vectors colinear
            pc = po + 0.5*(p1-po)
        else:  # parrallel non-coincedent
            raise ValueError('vectors are parallel and not colinear')
    else:  # lines cross
        f1 = np.dot(np.cross(po-p1, to_hat), c1o) / c1o_norm**2
        pc = p1 + f1*t1_hat
    return pc


def inloop(Xloop, Zloop, X, Z, side='in'):
    if side == 'in':
        sign = 1
    elif side == 'out':
        sign = -1
    else:
        raise ValueError('define side, \'in\' or \'out\'')
    Xloop, Zloop = clock(Xloop, Zloop)
    nXloop, nZloop, Xloop, Zloop = normal(Xloop, Zloop)
    Xin, Zin = np.array([]), np.array([])
    if isinstance(X, collections.Iterable):
        ''' return subset of points and points status (booliean) '''
        status = np.zeros(len(X), dtype=bool)
        for i, (x, z) in enumerate(zip(X, Z)):
            imin = np.argmin((x - Xloop)**2 + (z - Zloop)**2)
            dx = [Xloop[imin] - x, Zloop[imin] - z]
            dn = [nXloop[imin], nZloop[imin]]
            status[i] = sign * np.dot(dx, dn) > 0
            if status[i]:
                Xin, Zin = np.append(Xin, x), np.append(Zin, z)
        return Xin, Zin, status
    else:
        ''' return boolean '''
        imin = np.argmin((X - Xloop)**2 + (Z - Zloop)**2)
        dx = [Xloop[imin] - X, Zloop[imin] - Z]
        dn = [nXloop[imin], nZloop[imin]]
        return sign * np.dot(dx, dn) > 0


def max_steps(dX, dr_max):
    dRbar = np.mean(dX)
    nr = int(np.ceil(dRbar / dr_max))
    if nr < 2:
        nr = 2
    dx = dX / nr
    return dx, nr


def offset(X, Z, dX, min_steps=5, close_loop=False, s=0):
    X, Z = order(X, Z)  # enforce anti-clockwise
    dr_max = np.mean(dX) / min_steps  # maximum step size
    if np.mean(dX) != 0:
        dx, nr = max_steps(dX, dr_max)
        for i in range(nr):
            nR, nZ, X, Z = normal(X, Z)
            X = X + dx * nR
            Z = Z + dx * nZ
            X, Z = xzSLine(X, Z, npoints=len(X), s=s/nr)
            if close_loop:
                X[0], Z[0] = np.mean([X[0], X[-1]]), np.mean([Z[0], Z[-1]])
                X[-1], Z[-1] = X[0], Z[0]
    return X, Z


def cut(x, z):  # trim gaps
    for _ in range(len(x) - 1):
        dl = np.diff(length(x, z))
        if np.max(dl) > 2 * np.median(dl):
            icut = np.argmax(dl) - 1
            if icut > len(dl) / 2:
                x, z = x[:icut], z[:icut]
            else:
                x, z = x[icut:], z[icut:]
        else:
            break
    return x, z


class Loop(object):

    def __init__(self, X, Z, **kwargs):
        self.X = X
        self.Z = Z
        self.xo = kwargs.get('xo', (np.mean(X), np.mean(Z)))

    def xzPut(self):
        self.Xstore, self.Zstore = self.X, self.Z

    def xzGet(self):
        self.X, self.Z = self.Xstore, self.Zstore

    def fill(self, trim=None, dX=0, dt=0, ref_o=4 / 8 * np.pi, dref=np.pi / 4,
             edge=True, ends=True, color='k', label=None, alpha=0.8,
             referance='theta', part_fill=True, loop=False, s=0, gap=0,
             plot=False):
        dt_max = 0.1  # 2.5
        if not part_fill:
            dt_max = dt
        if isinstance(dt, list):
            dt = self.blend(dt, ref_o=ref_o, dref=dref, referance=referance,
                            gap=gap)
        dt, nt = max_steps(dt, dt_max)
        Xin, Zin = offset(self.X, self.Z, dX)  # gap offset
        for i in range(nt):
            self.part_fill(trim=trim, dt=dt, ref_o=ref_o, dref=dref,
                           edge=edge, ends=ends, color=color, label=label,
                           alpha=alpha, referance=referance, loop=loop,
                           s=s, plot=False)
        Xout, Zout = self.X, self.Z
        if plot:
            polyparrot({'x': Xin, 'z': Zin}, {'x': Xout, 'z': Zout},
                       color=color, alpha=1)  # fill
        return Xout, Zout

    def part_fill(self, trim=None, dt=0, ref_o=4/8*np.pi, dref=np.pi / 4,
                  edge=True, ends=True,
                  color='k', label=None, alpha=0.8, referance='theta',
                  loop=False, s=0, plot=False):
        Xin, Zin = self.X, self.Z
        if loop:
            Napp = 5  # Nappend
            X = np.append(self.X, self.X[:Napp])
            X = np.append(self.X[-Napp:], X)
            Z = np.append(self.Z, self.Z[:Napp])
            Z = np.append(self.Z[-Napp:], Z)
            X, Z = xzSLine(X, Z, npoints=len(X), s=s)
            if isinstance(dt, (np.ndarray, list)):
                dt = np.append(dt, dt[:Napp])
                dt = np.append(dt[-Napp:], dt)
            Xout, Zout = offset(X, Z, dt)
            print('part fill')
            Xout, Zout = Xout[Napp:-Napp], Zout[Napp:-Napp]
            Xout[-1], Zout[-1] = Xout[0], Zout[0]
        else:
            X, Z = xzSLine(self.X, self.Z, npoints=len(self.X), s=s)
            Xout, Zout = offset(X, Z, dt)
        self.X, self.Z = Xout, Zout  # update
        if trim is None:
            Lindex = [0, len(Xin)]
        else:
            Lindex = self.trim(trim)
        if plot:
            flag = 0
            for i in np.arange(Lindex[0], Lindex[1] - 1):
                Rfill = np.array([Xin[i], Xout[i], Xout[i + 1], Xin[i + 1]])
                Zfill = np.array([Zin[i], Zout[i], Zout[i + 1], Zin[i + 1]])
                if flag == 0 and label is not None:
                    flag = 1
                    plt.fill(Rfill, Zfill, facecolor=color, alpha=alpha,
                             edgecolor='none', label=label)
                else:
                    plt.fill(Rfill, Zfill, facecolor=color, alpha=alpha,
                             edgecolor='none')

    def blend(self, dt, ref_o=4 / 8 * np.pi, dref=np.pi / 4, gap=0,
              referance='theta'):
        if referance == 'theta':
            theta = np.arctan2(self.Z - self.xo[1], self.X - self.xo[0]) - gap
            theta[theta > np.pi] = theta[theta > np.pi] - 2 * np.pi
            tblend = dt[0] * np.ones(len(theta))  # inner
            tblend[(theta > -ref_o) & (theta < ref_o)] = dt[1]  # outer
            if dref > 0:
                for updown in [-1, 1]:
                    blend_index = (updown * theta >= ref_o) &\
                        (updown * theta < ref_o + dref)
                    tblend[blend_index] = dt[1] + (dt[0] - dt[1]) / dref *\
                        (updown * theta[blend_index] - ref_o)
        else:
            L = length(self.X, self.Z)
            tblend = dt[0] * np.ones(len(L))  # start
            tblend[L > ref_o] = dt[1]  # end
            if dref > 0:
                blend_index = (L >= ref_o) & (L < ref_o + dref)
                tblend[blend_index] = dt[0] + (dt[1] - dt[0]) /\
                    dref * (L[blend_index] - ref_o)
        return tblend

    def trim(self, trim, X, Z):
        L = length(X, Z, norm=True)
        index = []
        for t in trim:
            index.append(np.argmin(np.abs(L - t)))
        return index


def split_loop(x, z, xo, half):
    if 'upper' in half:
        index = z >= xo[1]
    elif 'lower' in half:
        index = z <= xo[1]
    else:
        errtxt = '\n'
        errtxt += 'specify loop segment [\'upper\',\'lower\']\n'
        raise ValueError(errtxt)
    x, z = x[index], z[index]
    x, z = theta_sort(x, z, xo=xo)
    return x, z


def trim_loop(x, z):
    n = len(x)
    index = np.ones(n, dtype=bool)
    for i in range(n - 2):
        if x[i + 1] <= x[i]:
            index[i + 1] = False
        else:
            index[i + 1] = True
        if index[i + 1] and not index[i]:  # keep corner
            index[i] = True
    x, z = x[index], z[index]
    if x[0] > x[1]:
        x[0] = x[1] - 1e-6
    if x[-1] < x[-2]:
        x[-1] = x[-2] + 1e-6
    return x, z


def process_loop(x, z):
    xo = (np.mean(x), z[np.argmax(x)])
    r1, z1 = split_loop(x, z, xo, 'upper')
    xo, zo = split_loop(x, z, xo, 'lower')
    r1 = np.append(xo[-1], r1)  # join upper
    r1 = np.append(r1, xo[0])
    z1 = np.append(zo[-1], z1)
    z1 = np.append(z1, zo[0])
    xo, zo = trim_loop(xo, zo)
    r1, z1 = trim_loop(r1[::-1], z1[::-1])
    return (xo, zo), (r1, z1)


def read_loop(part, loop, npoints=100, close=True):
    x, z = part[loop]['x'], part[loop]['z']
    if len(x) > 0:
        x, z = theta_sort(x, z)  # sort azimuth
        if close:
            try:
                x, z = np.append(x, x[0]), np.append(z, z[0])  # close loop
            except IndexError:  # empty entry
                pass
    return x, z


def polyloop(xin, xout, color=0.5 * np.ones(3), alpha=1):  # pair to single
    x = {}
    for var in ['x', 'z']:
        x[var] = np.append(xin[var], xout[var][::-1])
        x[var] = np.append(x[var], xin[var][0])
    return x['x'], x['z']


def polyarea(x, y, d3=None):
    '''
    Returns the area inside a closed polygon with x, y coordinate vectors. \n
    Shoelace method: https://en.wikipedia.org/wiki/Shoelace_formula \n
    Inputs: x, y list/array coordinates \n
    Outputs: area in m^2
    UPDATE 08/09/17: Now handles 3D polygons if 3rd coordinate is used
    TODO: catch edge case of n_hat with 3 points in a line
    '''
    if d3 is not None:
        p1 = np.array([x[0], y[0], d3[0]])
        p2 = np.array([x[1], y[1], d3[1]])
        p3 = np.array([x[2], y[2], d3[2]])
        v1, v2 = p3-p1, p2-p1
        v3 = np.cross(v1, v2)
        v3 = v3/np.linalg.norm(v3)
        a = np.zeros(3)
        m = np.array([x, y, d3])
        for i in range(len(d3)):
            a += np.cross(m[:, i], m[:, (i+1) % len(d3)])
        a /= 2
        return abs(np.dot(a, v3))
    else:
        if len(x) != len(y):
            raise Exception('Coordinate vectors must have same length.')
        A = np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1))) / 2
        return A


def polyfill(x, z, color=0.5*np.ones(3), alpha=1):
    verts = np.array([x, z])
    verts = [np.swapaxes(verts, 0, 1)]
    coll = PolyCollection(verts, edgecolors='none', color=color, alpha=alpha)
    ax = plt.gca()
    ax.add_collection(coll)
    ax.autoscale_view()


def polyfill3D(x, y, z, ax=None, color=0.5*np.ones(3), alpha=1, lw=2):
    if ax is None:
        ax = Axes3D(plt.figure())
    verts = np.array([x, y, z])
    verts = [np.swapaxes(verts, 0, 1)]
    coll = art3d.Poly3DCollection(
            verts, edgecolors='none', color=color, alpha=alpha)
    coll.set_edgecolor('k')
    coll.set_linewidth(lw)
    ax.add_collection3d(coll)
    ax.plot(x, y, z, '.', alpha=0)
    ax.autoscale_view()
    ax.autoscale()


def polyparrot(xin, xout, color=0.5 * np.ones(3), alpha=1):  # polyloopfill
    x, z = polyloop(xin, xout)
    polyfill(x, z, color=color, alpha=alpha)


def pointloop(x, z, ref='max'):  # orders points, selects breadcrumb neighbours
    n = len(x)
    x_, z_ = np.zeros(n), np.zeros(n)
    if ref == 'max':
        i = np.argmax(sp.linalg.norm([x, z], axis=0))
    else:
        i = np.argmin(z)
    x_[0], z_[0] = x[i], z[i]
    x, z = np.delete(x, i), np.delete(z, i)
    for i in range(n - 1):
        dx = sp.linalg.norm([x - x_[i], z - z_[i]], axis=0)
        j = np.argmin(dx)
        x_[i + 1], z_[i + 1] = x[j], z[j]
        x, z = np.delete(x, j), np.delete(z, j)
    x, z = x_, z_
    x, z = np.append(x, x[0]), np.append(z, z[0])
    return x, z


class String(object):
    '''
    simplify 2D or 3D string of points P
    P input as  numpy array, dimension (:,2) or (:,3)
    space function calculates simplfied line based on minimum turning angle
    and minimum and maximum segment lenghts
    '''

    def __init__(self, P, angle=10, dx_min=0, dx_max=np.inf, verbose=False):
        self.P = P  # string of points
        self.ndim = np.shape(self.P)[1]
        self.angle = angle  # maximum turning angle [degrees]
        self.dx_min = dx_min  # minimum pannel length
        self.dx_max = dx_max  # maximum pannel length
        self.space(verbose=verbose)

    def space(self, verbose=False, **kwargs):
        ''' option to override init defaults with kwargs '''
        self.angle = kwargs.get('angle', self.angle)
        self.dx_min = kwargs.get('dx_min', self.dx_min)
        self.dx_max = kwargs.get('dx_max', self.dx_max)

        T = self.P[1:] - self.P[:-1]  # tangent vector
        dT = np.linalg.norm(T, axis=1)
        dT[dT == 0] = 1e-36  # protect zero division
        dT_m = np.median(dT)  # average step length
        T /= dT.reshape(-1, 1)*np.ones((1, np.shape(T)[1]))
        self.points = np.zeros(np.shape(self.P))
        self.index = np.zeros(len(self.P), dtype=int)
        delta_x, delta_turn = np.zeros(len(self.P)), np.zeros(len(self.P))
        self.points[0] = self.P[0]
        to, po = T[0], self.P[0]
        k = count(1)
        for i, (p, t) in enumerate(zip(self.P[1:], T)):
            c = np.cross(to, t)
            c_mag = np.linalg.norm(c)
            dx = np.linalg.norm(p-po)  # segment length
            if (c_mag > np.sin(self.angle*np.pi/180) and dx > self.dx_min) or \
                    dx+dT_m > self.dx_max:  # store
                j = next(k)
                self.points[j] = self.P[i]  # pivot point
                self.index[j] = i+1  # pivot index
                delta_x[j-1] = dx  # pannel length
                delta_turn[j-1] = np.arcsin(c_mag)*180/np.pi
                to, po = t, p  # update
        if dx > self.dx_min:
            j = next(k)
            delta_x[j-1] = dx  # last segment length
        else:
            delta_x[j-1] += dx  # last segment length
        self.points[j] = p  # replace / append last point
        self.index[j] = i+1  # replace / append last point
        self.n = j+1  # reduced point number
        self.points = self.points[:j+1]  # trim
        self.index = self.index[:j+1]  # trim
        delta_x = delta_x[:j]  # trim
        delta_turn = delta_turn[:j]  # trim

        if verbose:
            print('\nturning angle: {:1.2f}'.format(self.angle))
            print('minimum pannel length: {:1.2f}, set: {:1.2f}'
                  .format(np.min(delta_x), self.dx_min))
            print('maximum pannel length: {:1.2f}, set: {:1.2f}'
                  .format(np.max(delta_x), self.dx_max))
            print('points input: {}, simplified: {}\n'
                  .format(len(self.P), len(self.points)))

    def plot(self, projection='2D', aspect=1):
        fig = plt.gcf()
        fig_width = fig.get_figwidth()
        fig_height = fig_width/aspect
        fig.set_figheight(fig_height)

        if projection == '3D':
            ax = fig.gca(projection='3d')
            ax.plot(self.points[:, 0], self.points[:, 1],
                    self.points[:, 2], 'o-')
            ax.plot(self.P[:, 0], self.P[:, 1], self.P[:, 2], '-')
        else:
            ax = fig.gca()
            ax.plot(self.P[:, 0], self.P[:, 1], '-')
            ax.plot(self.points[:, 0], self.points[:, 1], 'o-', ms=15)

        ax.set_axis_off()
        ax.set_aspect('equal')

        if projection == '3D':
            bb, zo = 7, 5
            ax.set_xlim([-bb, bb])
            ax.set_ylim([-bb, bb])
            ax.set_zlim([-bb+zo, bb+zo])
