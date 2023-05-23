import nova.cross_coil as cc
import pylab as pl
import numpy as np
import seaborn as sns
from nova.config import Setup
from nova.streamfunction import SF
from amigo import geom
from scipy.linalg import norm
from scipy.optimize import minimize_scalar
from nova import geqdsk


rc = {'figure.figsize': [8, 8], 'savefig.dpi': 100,  # *12/16
      'savefig.jpeg_quality': 100, 'savefig.pad_inches': 0.1,
      'lines.linewidth': 1.5}
sns.set(context='poster', style='white', font='sans-serif', palette='Set2',
        font_scale=7 / 8, rc=rc)
color = sns.color_palette('Set2')

# mu_o = 4*np.pi*1e-7  # magnetic constant [Vs/Am]


class ripple(object):
    def __init__(self, nTF=18, **kwargs):
        self.nTF = nTF
        if 'plasma' in kwargs:
            self.get_seperatrix(alpha=1 - 1e-3, **kwargs['plasma'])
        if 'coil' in kwargs:
            self.set_TFcoil(**kwargs['coil'])

    def get_seperatrix(self, nplasma=80, alpha=1 - 1e-3, **kwargs):
        self.nplasma = nplasma
        self.plasma_loop = np.zeros((self.nplasma, 3))  # initalise loop array
        if 'sf' in kwargs:  # seperatrix directly from sf object
            r, z = kwargs['sf'].get_boundary(alpha=alpha)
        elif 'config' in kwargs:
            setup = Setup(kwargs.get('config'))
            r, z = ripple.get_boundary(setup.filename, alpha=alpha)
        elif 'eqdsk' in kwargs:  # seperatrix from eqdsk
            eqdsk = geqdsk.read(kwargs.get('eqdsk'))
            r, z = eqdsk['rbdry'], eqdsk['zbdry']
        elif 'r' in kwargs and 'z' in kwargs:  # separatrix from input
            r, z = kwargs.get('r'), kwargs.get('z')
        else:
            errtxt = '\n'
            errtxt += 'Require plasma={} input of following types:\n'
            errtxt += '1) configuration flag, {\'config\':\'SN\'} \n'
            errtxt += '2) eqdsk file, {\'eqdsk\':\'\'} \n'
            errtxt += '3) seperatrix profile, {\'r\':[],\'z\':[]}'
            raise ValueError(errtxt)
        r, z = geom.clock(r, z)
        (self.plasma_loop[:, 0], self.plasma_loop[:, 2]) = \
            geom.rzSLine(r, z, npoints=self.nplasma)
        self.plasma_length = geom.length(self.plasma_loop[:, 0],
                                         self.plasma_loop[:, 2])
        rfun, zfun = geom.rzfun(r, z)
        self.plasma_interp = {'r': rfun, 'z': zfun}

    def get_boundary(filename, alpha=1e-3):
        sf = SF(filename)
        r, z = sf.get_boundary(alpha=alpha)
        return r, z

    def set_TFcoil(self, cl, npoints=80, smooth=False):
        r, z = geom.clock(cl['r'], cl['z'])
        self.coil_loop = np.zeros((npoints, 3))
        self.coil_loop[:, 0], self.coil_loop[:, 2] = \
            geom.rzSLine(r, z, npoints=npoints)  # coil centerline
        self.gfl = cc.GreenFeildLoop(self.coil_loop, smooth=smooth)

    def point(self, s, variable='ripple'):  # s==3D point vector
        B = np.zeros(2)
        n = np.array([0, 1, 0])
        if variable == 'ripple':
            planes = [0, np.pi / self.nTF]  # rotate (inline, ingap)
        elif variable == 'feild':
            planes = [0]
        else:
            errtxt = '\n'
            errtxt += 'point variable error, require \'ripple\' or \'feild\'\n'
            raise ValueError(errtxt)
        for j, theta in enumerate(planes):
            sr = np.dot(s, geom.rotate(theta))
            nr = np.dot(n, geom.rotate(theta))
            Bo = np.zeros(3)
            for tcoil in np.linspace(0, 2 * np.pi, self.nTF, endpoint=False):
                Bo += cc.mu_o * self.gfl.B(sr, theta=tcoil)
            B[j] = np.dot(nr, Bo)
        if variable == 'ripple':
            ripple = 1e2 * (B[0] - B[1]) / np.sum(B)
            return ripple
        else:
            return Bo

    def loop_ripple(self):
        self.ripple = np.zeros(self.nplasma)
        for i, plasma_point in enumerate(self.plasma_loop):
            self.ripple[i] = self.point(plasma_point, variable='ripple')

    def ripple_opp(self, x):
        s = np.zeros(3)
        s[0], s[2] = self.plasma_interp['r'](x), self.plasma_interp['z'](x)
        ripple = self.point(s, variable='ripple')
        return -ripple

    def get_ripple(self):
        self.res = minimize_scalar(self.ripple_opp, method='bounded',
                                   bounds=[0, 1], options={'xatol': 0.1})
        self.res['fun'] *= -1  # negate minimum (max ripple)
        return self.res['fun']

    def plot_loops(self, scale=1.5):
        self.get_ripple()
        pl.plot(self.res['x'], self.res['fun'], 'o', color=0.8 * np.ones(3))
        rpl, zpl = self.plasma_loop[:, 0], self.plasma_loop[:, 2]
        rr, zr = geom.offset(rpl, zpl, scale * self.ripple)
        for i in range(self.nplasma):  # ripple sticks
            pl.plot([rpl[i], rr[i]], [zpl[i], zr[i]],
                    lw=1, color=0.5 * np.ones(3))
        pl.plot(self.plasma_loop[:, 0], self.plasma_loop[:, 2], '-')
        pl.plot(self.coil_loop[:, 0], self.coil_loop[:, 2])
        x = self.res['x']
        pl.plot(self.plasma_interp['r'](x), self.plasma_interp['z'](x), '.',
                ms=10)
        pl.axis('equal')
        pl.axis('off')

    def plot_ripple_contours(self, n=3e3, offset=-1):
        xlim, dx = np.zeros((3, 2)), np.zeros(3)
        xl = np.zeros((len(self.coil_loop), 3))
        xl[:, 0], xl[:, 2] = geom.offset(self.coil_loop[:, 0],
                                         self.coil_loop[:, 2], offset)
        for i in range(3):
            xlim[i] = [np.min(xl[:, i]), np.max(xl[:, i])]
            dx[i] = xlim[i][1] - xlim[i][0]
        xlim[0][0] = self.plasma_loop[np.argmax(self.plasma_loop[:, 2]),
                                      0]  # plasma top
        ar = dx[0] / dx[2]
        nz = int(np.sqrt(n / ar))
        nr = int(n / nz)
        r = np.linspace(xlim[0][0], xlim[0][1], nr)
        z = np.linspace(xlim[2][0], xlim[2][1], nz)
        R, Z = np.meshgrid(r, z, indexing='ij')
        rpl = np.zeros((nr, nz))
        for i, r_ in enumerate(r):
            for j, z_ in enumerate(z):
                if geom.inloop(xl[:, 0], xl[:, 2], r_, z_):
                    rpl[i, j] = self.point((r_, 0, z_), variable='ripple')
                else:
                    rpl[i, j] = np.NaN

        levels = 10**(np.linspace(np.log10(0.01), np.log10(5), 7))
        levels = np.round(levels, decimals=2)

        geom.polyfill(self.plasma_loop[:, 0], self.plasma_loop[:, 2],
                      alpha=0.3, color=sns.color_palette('Set2', 5)[3])
        self.get_ripple()  # get max ripple on plasma contour
        rpl_max = self.res['fun']
        iplasma = np.argmin(abs(np.log10(levels) - np.log10(rpl_max)))
        levels[iplasma] = rpl_max  # select edge contour
        CS = pl.contour(R, Z, rpl, levels=levels, colors=[0.6 * np.ones(3)])
        zc = CS.collections[iplasma]
        pl.setp(zc, color=[0.4 * np.ones(3)])
        pl.clabel(CS, inline=1, fontsize='xx-small', fmt='%1.2f')


if __name__ is '__main__':
    from nova.coils import TF

    config = 'DEMO_SN'

    tf = TF(config, coil_type='S', npoints=100)

    # tf.coil.set_input(inputs={'upper':0.8,'top':0.5})

    tf.coil.plot()

    x = tf.coil.draw()
    tf.get_loops(x)
    tf.fill()

    rp = ripple(nTF=6, plasma={'config': config}, coil={'cl': tf.x['cl']})

    setup = Setup(config)
    eqdsk = geqdsk.read(setup.filename)
    Bo = rp.point((eqdsk['rcentr'], 0, eqdsk['zmagx']), variable='feild')
    I = eqdsk['bcentr'] / Bo[1]

    import time
    tic = time.time()
    print('ripple {:1.3f}%'.format(rp.get_ripple()))
    print('time', time.time() - tic)

    # rp.plot_ripple_contours(n=3e3)

    # tf.coil.set_input()

    B = np.zeros((tf.npoints, 3))
    for i, (r, z) in enumerate(zip(tf.x['cl']['r'], tf.x['cl']['z'])):
        B[i, :] = I * rp.point((r, 0, z), variable='feild')

    npoints = 100
    rcl = np.linspace(np.min(tf.x['cl']['r']),
                      np.max(tf.x['cl']['r']), npoints)
    zcl = eqdsk['zmagx'] * np.ones(npoints)
    Bcl = np.zeros((npoints, 3))

    for i, (r, z) in enumerate(zip(rcl, zcl)):
        Bcl[i, :] = I * rp.point((r, 0, z), variable='feild')

    pl.figure()
    a = np.mean(tf.x['cl']['r'] * abs(B[:, 1]))
    pl.plot(tf.x['cl']['r'], abs(B[:, 1]))
    pl.plot(tf.x['cl']['r'], abs(B[:, 0]))
    pl.plot(tf.x['cl']['r'], abs(B[:, 2]))
    pl.plot(tf.x['cl']['r'], a / tf.x['cl']['r'], '--')
    pl.plot(rcl, abs(Bcl[:, 1]))

    pl.plot(eqdsk['rcentr'], abs(eqdsk['bcentr']), 'o')

    # rp.plot_loops()
