import numpy as np
from scipy.special import ellipk, ellipe
from amigo import geom
from scipy.interpolate import interp1d
from scipy.linalg import norm
import pylab as pl

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
        g_s = 4 * np.pi * \
            Xc[index] * (np.log(8 * Xc[index] / rho) -
                         1.75) / (2 * np.pi)
        # g[index] = g_s
        g[index] = 0
    return g


def green_feild(X, Z, Xc, Zc):
    feild = np.zeros(2)
    a = np.sqrt((X + Xc)**2 + (Z - Zc)**2)
    m = 4 * X * Xc / a**2
    I1 = 4 / a * ellipk(m)
    I2 = 4 / a**3 * ellipe(m) / (1 - m)
    A = (Z - Zc)**2 + X**2 + Xc**2
    B = -2 * X * Xc
    feild[0] = Xc / (4 * np.pi) * (Z - Zc) / B * (I1 - A * I2)
    feild[1] = Xc / (4 * np.pi) * ((Xc + X * A / B) * I2 - X / B * I1)
    return feild


class GreenFeildLoop(object):
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

    def B(self, point, theta=0, dy=0, dr=0):  # 3D feild from arbitrary loop
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
        Bfeild = sum(core * np.cross(ds, r2_hat - r1_hat) /
                     ds_mag**2) / (4 * np.pi)
        return Bfeild

    def plot(self):
        pl.figure()
        pl.plot(self.loop_cl[:, 0], self.loop_cl[:, 1])
        pl.plot(self.loop_ss[:, 0], self.loop_ss[:, 1])
        pl.axis('equal')
        pl.axis('off')


def green_feild_circle(coil, point, N=20):  # 3D feild from arbitrary loop
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
    feild = np.sum(np.cross(dL, x) / r_mag**3, axis=0) / (4 * np.pi)  # Bfeild
    return feild


def cut_corners(loop, smooth=True, Nss=100):
    if smooth:  # round edges of segmented coil
        if Nss < len(loop):
            Nss = len(loop)
        N = np.shape(loop)[0]
        loop_ss = np.zeros((Nss, 3))
        l = geom.vector_length(loop)
        lss = np.linspace(0, 1, Nss)
        npad = 2  # mirror loop for cubic interpolant
        for i in range(3):
            loop_m = np.pad(loop[:-1, i], npad, 'wrap')
            l_m = np.pad(l[:-1], npad, 'linear_ramp',
                         end_values=[-npad * l[1], l[-2] +
                                     npad * (l[-1] - l[-2])])
            loop_ss[:, i] = interp1d(l_m, loop_m, kind='cubic')(lss)
        Lss = geom.vector_length(loop_ss, norm=False)
        L = interp1d(lss, Lss)(l)  # cumulative length
        dL_seg = L[1:] - L[:-1]  # segment length
        dL_seg = np.append(dL_seg[-1], dL_seg)  # prepend
        dL_mag = (dL_seg[1:] + dL_seg[:-1]) / 2  # average segment length
        dLss = np.gradient(loop_ss, axis=0)
        dLss /= np.dot(np.reshape(norm(dLss, axis=1), (-1, 1)),
                       np.ones((1, 3)))  # unit tangent
        dL = np.zeros((N - 1, 3))
        for i in range(3):
            dL[:, i] = interp1d(lss, dLss[:, i])(l[:-1]) * dL_mag
    else:
        dL = loop[1:] - loop[:-1]
        dL = np.append(np.reshape(dL[-1, :], (1, 3)), dL, axis=0)  # prepend
        dL = (dL[1:] + dL[:-1]) / 2  # central diffrence average segment length
        loop_ss = loop
    return dL, loop_ss


def Gtorque(eq_coil, pf_coil, source, sink, multi_filament):  # source-sink
    if multi_filament:
        coil = eq_coil
        Nbundle = 1
        Nsource = coil[source + '_0']['Nf']
        Nsink = coil[sink + '_0']['Nf']

        def name_source(i): return source + '_{:1.0f}'.format(i)

        def name_sink(j): return sink + '_{:1.0f}'.format(j)
    else:  # single-filament
        coil = pf_coil
        Nbundle = eq_coil[source + '_0']['Nf'] * eq_coil[sink + '_0']['Nf']
        Nsource, Nsink = 1, 1

        def name_source(i): return source

        def name_sink(j): return sink
    feild = np.zeros(2)
    for i in range(Nsource):
        source_strand = name_source(i)  # source
        ri = coil[source_strand]['x']
        zi = coil[source_strand]['z']
        for j in range(Nsink):
            sink_strand = name_sink(j)  # sink
            x = coil[sink_strand]['x']
            z = coil[sink_strand]['z']
            rc = coil[sink_strand]['rc']
            r_mag = np.sqrt((x - ri)**2 + (z - zi)**2)
            if r_mag > rc:  # outside coil core
                dfeild = green_feild(x, z, ri, zi)
            else:  # inside coil core
                dfeild, B = np.zeros(2), np.zeros(2)
                dz = (rc**2 - (x - ri)**2)**0.5  # Br
                for i, zc in enumerate([zi - dz, zi + dz]):
                    B[i] = green_feild(x, z, ri, zc)[0]
                dfeild[0] = sum(B) / 2 + (z - zi) * (B[1] - B[0]) / (2 * dz)
                dr = (rc**2 - (z - zi)**2)**0.5  # Bz
                for i, rc in enumerate([ri - dr, ri + dr]):
                    B[i] = green_feild(x, z, rc, zi)[1]
                dfeild[1] = sum(B) / 2 + (x - ri) * (B[1] - B[0]) / (2 * dr)
            feild += Nbundle * x * dfeild  # feild couple, rG
    return feild


def Btorque(eq_coil, plasma_coil, passive_coils, sink):
    Csink = eq_coil
    Nsink = Csink[sink + '_0']['Nf']
    feild = np.zeros(2)
    for source in passive_coils:
        if source == 'Plasma':
            Csource = plasma_coil
            Nsource = len(Csource)
        else:
            Csource = eq_coil
            Nsource = Csource[source + '_0']['Nf']
        for i in range(Nsource):
            source_strand = source + '_{:1.0f}'.format(i)
            ri = Csource[source_strand]['x']  # source
            zi = Csource[source_strand]['z']
            Ii = Csource[source_strand]['I']
            for j in range(Nsink):
                sink_strand = sink + '_{:1.0f}'.format(j)
                x = Csink[sink_strand]['x']  # sink
                z = Csink[sink_strand]['z']
                feild += Ii * x * green_feild(x, z, ri, zi)
    return feild


def Gfeild(coil, plasma_coil, point):
    feild = np.zeros(2)
    for coil_set in [coil, plasma_coil]:
        feild += Btorque(coil_set, point)
    return feild


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
