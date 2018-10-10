import numpy as np
from amigo.pyplot import plt
from scipy.special import ellipk, ellipe
from amigo import geom
from scipy.interpolate import interp1d
from scipy.linalg import norm

mu_o = 4 * np.pi * 1e-7  # magnetic constant [Vs/Am]


def green(X, Z, Xc, Zc, dXc=0, dZc=0):
    x = np.array((X - Xc)**2 + (Z - Zc)**2)
    m = 4 * X * Xc / ((X + Xc)**2 + (Z - Zc)**2)
    g = np.array((Xc * X)**0.5 *
                 ((2 * m**-0.5 - m**0.5) *
                  ellipk(m) - 2 * m**-0.5 * ellipe(m)) / (2 * np.pi))
    if np.min(np.sqrt(x)) < dXc / 2:  # self inductance
        rho = np.mean([dXc + dZc]) / 2
        Xc = Xc * np.ones(np.shape(X))
        index = np.sqrt(x) < dXc / 2  # self inductance index
        g_s = Xc[index] * (np.log(8 * Xc[index] / rho) - 2) / (2 * np.pi)
        g[index] = g_s
    return g


def green_field(X, Z, Xc, Zc):
    field = np.zeros(2)
    a = np.sqrt((X + Xc)**2 + (Z - Zc)**2)
    m = 4 * X * Xc / a**2
    I1 = 4 / a * ellipk(m)
    I2 = 4 / a**3 * ellipe(m) / (1 - m)
    A = (Z - Zc)**2 + X**2 + Xc**2
    B = -2 * X * Xc
    field[0] = Xc / (4 * np.pi) * (Z - Zc) / B * (I1 - A * I2)
    field[1] = Xc / (4 * np.pi) * ((Xc + X * A / B) * I2 - X / B * I1)
    return field


def add_Pcoil(x, z, coil):
    xc, zc, If = coil['x'], coil['z'], coil['If']
    dx, dz = coil['dx'], coil['dz']
    return mu_o * If * green(x, z, xc, zc, dXc=dx, dZc=dz)


def add_Bcoil(x, z, coil):
    xc, zc, If = coil['x'], coil['z'], coil['If']
    return mu_o * If * green_field(x, z, xc, zc)


def get_plasma_coil(coilset, **kwargs):
    plasma_coil = kwargs.get('plasma_coil', coilset['plasma_coil'])
    return plasma_coil


def Ppoint(point, coilset, **kwargs):
    plasma_coil = get_plasma_coil(coilset, **kwargs)
    psi = 0
    for name in coilset['subcoil']:
        psi += add_Pcoil(point[0], point[1], coilset['subcoil'][name])
    for name in plasma_coil:
        psi += add_Pcoil(point[0], point[1], plasma_coil[name])
    return psi


def Bpoint(point, coilset, **kwargs):
    plasma_coil = get_plasma_coil(coilset, **kwargs)
    field = np.zeros(2)
    for name in coilset['subcoil']:
        field += add_Bcoil(point[0], point[1], coilset['subcoil'][name])
    for name in plasma_coil:
        field += add_Bcoil(point[0], point[1], plasma_coil[name])
    return field


def Bmag(point, *args):
    pf = args[0]  # pf object
    if len(args) == 2:
        kwargs = {'plasma_coil': args[1]}  # specify seperate plasma_coil
    else:
        kwargs = {}  # use pf.plasma_coil
    field = Bpoint(point, pf, **kwargs)
    B = np.sqrt(field[0]**2 + field[1]**2)
    return B


def get_coil_psi(x2d, z2d, pf_subcoil, plasma_coil, **kwargs):
    # plasma_coil = get_plasma_coil(pf, **kwargs)
    set_pf = kwargs.get('set_pf', True)
    set_plasma = kwargs.get('set_plasma', True)
    if len(np.shape(x2d)) > 0:
        psi = np.zeros(np.shape(x2d))
    else:
        psi = 0
    if set_pf:
        for name in pf_subcoil.keys():
            psi += add_Pcoil(x2d, z2d, pf_subcoil[name])
    if set_plasma:
        for name in plasma_coil.keys():
            psi += add_Pcoil(x2d, z2d, plasma_coil[name])
    return psi


class GreenFieldLoop(object):
    def __init__(self, loop, smooth=True, Nss=100, rc=0.5):
        self.rc = rc
        if np.sum((loop[0, :] - loop[-1, :])**2) != 0:  # close loop
            loop = np.append(loop, np.reshape(loop[0, :], (1, 3)), axis=0)
        self.dL, self.loop_ss = cut_corners(loop, smooth=smooth, Nss=Nss)
        self.loop_cl = np.copy(loop)
        self.loop = self.loop_cl[:-1, :]  # re-open loop
        self.npoints = len(self.loop)

    def transform(self, theta, dy, dr):  # translate / rotate
        loop, dL = np.copy(self.loop), np.copy(self.dL)
        if dr != 0:  # offset loop
            loop[:, 0], loop[:, 2] = geom.offset(loop[:, 0], loop[:, 2], dr)
        if dy != 0:  # translate in y
            loop[:, 1] += dy
        if theta != 0:  # rotate about z-axis
            loop = np.dot(loop, geom.rotate(theta))
            dL = np.dot(dL, geom.rotate(theta))
        return loop, dL

    def A(self, point, theta=0, dy=0, dr=0):  # vector potential
        loop, dL = self.transform(theta, dy, dr)
        point = np.array(point) * np.ones((self.npoints, 3))  # point array
        x = point - loop  # point-segment vectors
        r_mag = np.tile(norm(x, axis=1), (3, 1)).T
        r_mag[r_mag < 1e-16] = 1e-16
        core = r_mag / self.rc
        core[r_mag > self.rc] = 1
        Apot = np.sum(core * dL / r_mag, axis=0) / (4 * np.pi)
        return Apot

    def B(self, point, theta=0, dy=0, dr=0):  # 3D field from arbitrary loop
        loop, dL = self.transform(theta, dy, dr)
        point = np.array(point) * np.ones((self.npoints, 3))  # point array
        x = point - loop  # point-segment vectors
        r1 = x - dL / 2
        r1_hat = r1 / np.tile(norm(r1, axis=1), (3, 1)).T
        r2 = x + dL / 2
        r2_hat = r2 / np.tile(norm(r2, axis=1), (3, 1)).T
        dL_hat = np.tile(norm(dL, axis=1), (3, 1)).T
        ds = np.cross(dL, x) / dL_hat
        ds_mag = np.tile(norm(ds, axis=1), (3, 1)).T
        ds = np.cross(dL, ds) / dL_hat
        ds_mag[ds_mag < 1e-16] = 1e-16
        core = ds_mag**2 / self.rc**2
        core[ds_mag > self.rc] = 1
        Bfield = sum(core * np.cross(ds, r2_hat - r1_hat) /
                     ds_mag**2) / (4 * np.pi)
        return Bfield

    def plot(self):
        plt.figure()
        plt.plot(self.loop_cl[:, 0], self.loop_cl[:, 1])
        plt.plot(self.loop_ss[:, 0], self.loop_ss[:, 1])
        plt.axis('equal')
        plt.axis('off')


def green_field_circle(coil, point, N=20):  # 3D field from arbitrary loop
    theta, dtheta = np.linspace(
        0, 2 * np.pi, N, endpoint=False, retstep=True)  # angle
    c = np.transpose(np.array([coil['x'] * np.cos(theta),
                               coil['x'] * np.sin(theta),
                               np.array([coil['z']] * N)]))  # position
    dL = coil['x'] * np.transpose(np.array([-np.sin(theta), np.cos(theta),
                                            np.array([0] * len(theta))])) *\
        dtheta  # segment
    x = point - c  # point-segment vectors
    r_mag = np.transpose(np.sum(x * x, axis=1)**0.5 * np.ones((3, N)))
    field = np.sum(np.cross(dL, x) / r_mag**3, axis=0) / (4 * np.pi)  # Bfield
    return field


def cut_corners(loop, smooth=True, Nss=100):
    if smooth:  # round edges of segmented coil
        if Nss < len(loop):
            Nss = len(loop)
        N = np.shape(loop)[0]
        loop_ss = np.zeros((Nss, 3))
        le = geom.vector_length(loop)
        lss = np.linspace(0, 1, Nss)
        npad = 2  # mirror loop for cubic interpolant
        for i in range(3):
            loop_m = np.pad(loop[:-1, i], npad, 'wrap')
            l_m = np.pad(le[:-1], npad, 'linear_ramp',
                         end_values=[-npad * le[1], le[-2] +
                                     npad * (le[-1] - le[-2])])
            loop_ss[:, i] = interp1d(l_m, loop_m, kind='cubic')(lss)
        Lss = geom.vector_length(loop_ss, norm=False)
        L = interp1d(lss, Lss)(le)  # cumulative length
        dL_seg = L[1:] - L[:-1]  # segment length
        dL_seg = np.append(dL_seg[-1], dL_seg)  # prepend
        dL_mag = (dL_seg[1:] + dL_seg[:-1]) / 2  # average segment length
        dLss = np.gradient(loop_ss, axis=0)
        dLss /= np.dot(np.reshape(norm(dLss, axis=1), (-1, 1)),
                       np.ones((1, 3)))  # unit tangent
        dL = np.zeros((N - 1, 3))
        for i in range(3):
            dL[:, i] = interp1d(lss, dLss[:, i])(le[:-1]) * dL_mag
    else:
        dL = loop[1:] - loop[:-1]
        dL = np.append(np.reshape(dL[-1, :], (1, 3)), dL, axis=0)  # prepend
        dL = (dL[1:] + dL[:-1]) / 2  # central diffrence average segment length
        loop_ss = loop
    return dL, loop_ss


def get_green_field(x, z, xi, zi, rc):
    r_mag = np.sqrt((x - xi)**2 + (z - zi)**2)
    if r_mag > rc:  # outside coil core
        dfield = green_field(x, z, xi, zi)
    else:  # inside coil core
        dfield, B = np.zeros(2), np.zeros(2)
        dz = (rc**2 - (x - xi)**2)**0.5  # Br
        for i, zc in enumerate([zi - dz, zi + dz]):
            B[i] = green_field(x, z, xi, zc)[0]
        dfield[0] = sum(B) / 2 + (z - zi) * (B[1] - B[0]) / (2 * dz)
        dr = (rc**2 - (z - zi)**2)**0.5  # Bz
        for i, rc in enumerate([xi - dr, xi + dr]):
            B[i] = green_field(x, z, rc, zi)[1]
        dfield[1] = sum(B) / 2 + (x - xi) * (B[1] - B[0]) / (2 * dr)
    return dfield


def Gtorque(coil, subcoil, source, sink, multi_filament):  # source-sink

    if multi_filament:
        Nbundle = 1
        Nsource = coil[source]['Nf']
        Nsink = coil[sink]['Nf']
        coil = subcoil

        def name_source(i):
            return source + '_{:1.0f}'.format(i)

        def name_sink(j):
            return sink + '_{:1.0f}'.format(j)

    else:  # single-filament
        Nbundle = coil[source]['Nf'] * coil[sink]['Nf']
        Nsource, Nsink = 1, 1

        def name_source(i):
            return source

        def name_sink(j):
            return sink

    field = np.zeros(2)
    for i in range(Nsource):
        source_strand = name_source(i)  # source
        xi = coil[source_strand]['x']
        zi = coil[source_strand]['z']
        for j in range(Nsink):
            sink_strand = name_sink(j)  # sink
            x = coil[sink_strand]['x']
            z = coil[sink_strand]['z']
            rc = coil[sink_strand]['rc']
            dfield = get_green_field(x, z, xi, zi, rc)
            field += Nbundle * x * dfield  # field couple, rG
    return field


def Btorque(coil, subcoil, plasma_coil, passive_coils, sink):
    Csink = subcoil
    Nsink = coil[sink]['Nf']
    field = np.zeros(2)
    for source in passive_coils:
        if source == 'Plasma':
            Csource = plasma_coil
            Nsource = len(Csource)
        else:
            Csource = subcoil
            Nsource = coil[source]['Nf']
        for i in range(Nsource):
            source_strand = source + '_{:1.0f}'.format(i)
            xi = Csource[source_strand]['x']  # source
            zi = Csource[source_strand]['z']
            Ii = Csource[source_strand]['If']
            for j in range(Nsink):
                sink_strand = sink + '_{:1.0f}'.format(j)
                x = Csink[sink_strand]['x']  # sink
                z = Csink[sink_strand]['z']
                rc = Csink[sink_strand]['rc']
                dfield = get_green_field(x, z, xi, zi, rc)
                field += Ii * x * dfield
    return field


def Gfield(coil, plasma_coil, point):
    field = np.zeros(2)
    for coil_set in [coil, plasma_coil]:
        field += Btorque(coil_set, point)
    return field


def tangent(X, Z, norm=True):
    dR, dZ = np.diff(X), np.diff(Z)
    X, Z = (X[:-1] + X[1:]) / 2, (Z[:-1] + Z[1:]) / 2
    if norm:
        mag = np.sqrt(dR**2 + dZ**2)
    else:
        mag = np.ones(len(X))
    index = mag > 0
    dR, dZ, mag = dR[index], dZ[index], mag[index]  # clear duplicates
    X, Z = X[index], Z[index]
    return dR / mag, dZ / mag, X, Z


def normal(X, Z, norm=True):
    tX, tZ, X, Z = tangent(X, Z, norm=norm)
    t = np.zeros((len(X), 3))
    t[:, 0], t[:, 1] = tX, tZ
    n = np.cross(t, [0, 0, 1])
    nX, nZ = n[:, 0], n[:, 1]
    return nX, nZ, X, Z
