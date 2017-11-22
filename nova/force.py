import nova.cross_coil as cc
import pylab as pl
import numpy as np
import matplotlib
from amigo import geom


class force_feild(object):

    def __init__(self, index, pf_coil, eq_coil, eq_plasma_coil, Iscale=1e6,
                 multi_filament=True, plot=False, **kwargs):
        self.index = index
        self.pf_coil = pf_coil
        self.eq_coil = eq_coil
        self.eq_plasma_coil = eq_plasma_coil
        self.Iscale = Iscale  # current units (MA)
        self.nC = len(self.pf_coil)
        self.active_coils = kwargs.get(
            'active_coils', list(self.pf_coil.keys()))
        self.passive_coils = kwargs.get('passive_coils', ['Plasma'])
        self.set_force_feild(state='both', multi_filament=multi_filament)
        self.set_current()
        self.initalize_F()
        if plot:
            self.plot()

    def initalize_F(self):
        self.Fcoil = {}
        for name in self.active_coils:
            self.Fcoil[name] = {'Fx': 0, 'Fz': 0}
        self.Fcoil['CS'] = {'Fz': 0}

    def check(self):
        if not self.force_feild_active:
            errtxt = 'no force feild\n'
            errtxt += 'set_force_feild\n'
            raise ValueError(errtxt)
        if self.index['PF']['n'] == 0:
            errtxt = 'PF_coils empty\n'
            raise ValueError(errtxt)
        if self.index['CS']['n'] == 0:
            errtxt = 'CS_coils empty\n'
            raise ValueError(errtxt)

    def set_force_feild(self, state='both', multi_filament=False):
        # [I]T([Fa][I]+[Fp]) = F
        self.force_feild_active = True
        if state == 'both' or state == 'active':
            self.set_active_force_feild(multi_filament=multi_filament)
        if state == 'both' or state == 'passive':
            self.set_passive_force_feild()

    def set_active_force_feild(self, multi_filament=False):
        self.Fa = np.zeros((self.nC, self.nC, 2))  # active
        for i, sink in enumerate(self.active_coils):
            for j, source in enumerate(self.active_coils):
                rG = cc.Gtorque(self.eq_coil, self.pf_coil,
                                source, sink, multi_filament) * self.Iscale**2
                self.Fa[i, j, 0] = 2 * np.pi * cc.mu_o * rG[1]  # cross product
                self.Fa[i, j, 1] = -2 * np.pi * cc.mu_o * rG[0]

    def set_passive_force_feild(self):
        self.Fp = np.zeros((self.nC, 2))  # passive
        for i, sink in enumerate(self.active_coils):
            rB = cc.Btorque(self.eq_coil, self.eq_plasma_coil,
                            self.passive_coils, sink) * self.Iscale
            self.Fp[i, 0] = 2 * np.pi * cc.mu_o * rB[1]  # cross product
            self.Fp[i, 1] = -2 * np.pi * cc.mu_o * rB[0]

    def set_force(self, I):  # evaluate coil force and force jacobian
        self.F = np.zeros((self.nC, 2))
        dF = np.zeros((self.nC, self.nC, 2))
        Im = np.dot(I.reshape(-1, 1), np.ones((1, self.nC)))  # current matrix
        for i in range(2):  # coil force (bundle of eq elements)
            self.F[:, i] = 1e-6 * \
                (I * (np.dot(self.Fa[:, :, i], I) + self.Fp[:, i]))  # MN
            dF[:, :, i] = Im * self.Fa[:, :, i]
            diag = np.dot(self.Fa[:, :, i], I) +\
                I * np.diag(self.Fa[:, :, i]) + self.Fp[:, i]
            np.fill_diagonal(dF[:, :, i], diag)
        dF *= 1e-6  # force jacobian MN/MA
        return self.F, dF

    def get_force(self):
        self.check()
        F, dF = self.set_force(self.I)
        Fcoil = {'PF': {'x': 0, 'z': 0}, 'CS': {
            'sep': 0, 'zsum': 0}, 'F': F, 'dF': dF}
        Fcoil['PF']['x'] = np.max(abs(F[self.index['PF']['index'], 0]))
        Fcoil['PF']['z'] = np.max(abs(F[self.index['PF']['index'], 1]))
        FzCS = F[self.index['CS']['index'], 1]
        if self.index['CS']['n'] > 1:
            Fsep = np.zeros(self.index['CS']['n'] - 1)  # seperation force
            for j in range(self.index['CS']['n'] - 1):  # evaluate each gap
                Fsep[j] = np.sum(FzCS[j + 1:]) - np.sum(FzCS[:j + 1])
            Fcoil['CS']['sep'] = np.max(Fsep)
        Fcoil['CS']['zsum'] = np.sum(FzCS)
        for name, F in zip(self.active_coils, F):
            self.Fcoil[name] = {'fx': F[0], 'fz': F[1]}
        self.Fcoil['CS'] = {'fz': Fcoil['CS']['zsum']}
        return Fcoil

    def set_current(self):  # set eq current vector from pf
        self.I = np.zeros(self.nC)
        for i, name in enumerate(self.active_coils):
            Nfilament = self.eq_coil[name + '_0']['Nf']
            self.I[i] = self.pf_coil[name]['I'] / Nfilament  # store current
        self.I /= self.Iscale  # convert current unit (MA)

    def plot(self, scale=3):
        fs = matplotlib.rcParams['legend.fontsize']
        if not hasattr(self, 'F'):
            self.set_force(self.I)
        Fmax = np.max(np.linalg.norm(self.F, axis=1))
        index = np.append(self.index['PF']['index'], self.index['CS']['index'])
        names = np.append(self.index['PF']['name'], self.index['CS']['name'])
        for i, name in zip(index, names):
            x, z = self.pf_coil[name]['x'], self.pf_coil[name]['z']
            F = self.F[i]
            zarrow = scale * F[1] / Fmax
            # pl.arrow(x,z,scale*F[0]/Fmax,0,  # Fr
            #         linewidth=2,head_width=0.3,head_length=0.4,
            #         ec=0.4*np.ones(3),fc=0.7*np.ones(3))
            pl.arrow(x, z, 0, zarrow,  # Fz
                     linewidth=2*2, head_width=2*0.12, head_length=2*0.1,
                     ec=0.6 * np.ones(3), fc=0.6 * np.ones(3))
            if name in self.index['PF']['name']:
                zarrow += np.sign(zarrow) * 0.5
                if abs(zarrow) < self.pf_coil[name]['dz']:
                    zarrow = np.sign(zarrow) * self.pf_coil[name]['dz']
                if zarrow > 0:
                    va = 'bottom'
                else:
                    va = 'top'
                pl.text(x, z + zarrow, '{:1.0f}MN'.format(F[1]),
                        ha='center', va=va, fontsize=fs,
                        color=0.1 * np.ones(3),
                        backgroundcolor=0.9 * np.ones(3))

    def set_bm(self, cage):
        x = {'cl': {'x': cage.coil_loop[:, 0], 'z': cage.coil_loop[:, 2]}}
        i = np.argmax(x['cl']['z'])
        ro, zo = x['cl']['x'][i], x['cl']['z'][i]
        self.bm = ro * cage.point((ro, 0, zo),
                                  variable='feild')[1]  # TF moment

    def topple(self, point, J, cage, Bpoint, method='function', **kwargs):
        # eq.Bpoint == point calculated method (slow)
        # sf.Bpoint == spline interpolated method (fast)
        x = {'cl': {'x': cage.coil_loop[:, 0], 'z': cage.coil_loop[:, 2]}}
        if 'streamfunction' in Bpoint.__str__():
            topright = Bpoint((np.max(x['cl']['x']),
                               np.max(x['cl']['z'])),
                              check_bounds=True)
            bottomleft = Bpoint((np.max(x['cl']['x']),
                                 np.max(x['cl']['z'])),
                                check_bounds=True)
            if not(topright and bottomleft):
                errtxt = 'TF coil extends outside Bpoint interpolation grid\n'
                errtxt += 'Extend sf grid\n'
                raise ValueError(errtxt)
        if method == 'function':  # calculate tf feild as fitted 1/x function
            if not hasattr(self, 'bm'):
                self.set_bm(cage)
        elif method != 'BS':  # raise error if method not 'function' or 'BS'
            errtxt = 'invalid tf feild method {}\n'.format(method)
            errtxt += 'select method from \'function\' or \'BS\'\n'
            raise ValueError(errtxt)
        if method == 'function':
            b = np.zeros(3)
            # TF feild (fast version / only good for TF cl)
            b[1] = self.bm / (point[0])
        elif method == 'BS':  # calculate tf feild with full Biot-Savart
            b = cage.point(point, variable='feild')  # (slow / correct)
        theta = np.arctan2(point[1], point[0])

        pf_point = np.dot(geom.rotate(-theta, 'z'),
                          point)  # rotate to PF plane
        # PF feild (sf-fast, eq-slow)
        pf_b = Bpoint([pf_point[0], pf_point[2]])
        #b += np.dot(geom.rotate(theta, 'z'), [pf_b[0], 0, pf_b[1]])  # add PF
        Fbody = np.cross(J, b)  # body force
        return Fbody


if __name__ is '__main__':  # test functions
    from nova.config import Setup
    from nova.streamfunction import SF
    from nova.elliptic import EQ
    from nova.coils import PF
    from nova.radial_build import RB

    setup = Setup('SN_3PF_18TF')

    sf = SF(setup.filename)
    pf = PF(sf.eqdsk)
    rb = RB(sf, setup)
    eq = EQ(sf, pf, dCoil=0.75, sigma=0,
            boundary=sf.get_sep(expand=0.25), n=1e3)
    eq.get_plasma_coil()
    # eq.gen_opp()

    ff = force_feild(pf.index, pf.coil, pf.sub_coil, pf.plasma_coil,
                     multi_filament=True)
    Fcoil = ff.get_force()

    ff.plot()
    pf.plot(label=True, plasma=False, current=True)
    pf.plot(subcoil=True, label=False, plasma=True, current=False)
    pl.axis('equal')
    pl.axis('off')
