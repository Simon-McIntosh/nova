import nova.cross_coil as cc
from amigo.pyplot import plt
import numpy as np
import matplotlib
from amigo import geom


class force_field(object):

    def __init__(self, coilset, Iscale=1e6, multi_filament=True,
                 plot=False, **kwargs):
        self.coilset = coilset  # requires update
        self.Iscale = Iscale  # current units (MA)
        self.active_coils = kwargs.get('active_coils',
                                       list(self.coilset['coil'].keys()))
        self.passive_coils = kwargs.get('passive_coils', ['Plasma'])
        self.nC = len(self.active_coils)  # number of active coils
        self.nP = len(self.passive_coils)  # number of active coils
        self.initalize_force_field(multi_filament=multi_filament)
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
        if not self.force_field_active:
            errtxt = 'no force field\n'
            errtxt += 'set_force_field\n'
            raise ValueError(errtxt)
        if self.coilset['index']['PF']['n'] == 0:
            errtxt = 'PF_coils empty\n'
            raise ValueError(errtxt)
        if self.coilset['index']['CS']['n'] == 0:
            errtxt = 'CS_coils empty\n'
            raise ValueError(errtxt)

    def initalize_force_field(self, multi_filament=True, **kwargs):
        if 'Fa' in self.coilset and 'Fp' in self.coilset:
            self.force_field_active = True
            self.Fa = self.coilset['Fa']
            self.Fp = self.coilset['Fp']
            nC = np.shape(self.Fa)[0]
            if nC != self.nC:
                err_txt = 'active coil force matrix shape {}\n'.format(nC)
                err_txt += 'incompatible with number of'
                err_txt += 'active {} coils'.format(self.nC)
                raise IndexError(err_txt)
        else:
            self.set_force_field(state='both', multi_filament=multi_filament)

    def set_force_field(self, state='both', multi_filament=False):
        # [Ic]T([Fa][Ic]+[Fp]) = F
        self.force_field_active = True
        if state == 'both' or state == 'active':
            self.set_active_force_field(multi_filament=multi_filament)
        if state == 'both' or state == 'passive':
            self.set_passive_force_field()

    def set_active_force_field(self, multi_filament=False):
        self.Fa = np.zeros((self.nC, self.nC, 2))  # active
        for i, sink in enumerate(self.active_coils):
            for j, source in enumerate(self.active_coils):
                xG = cc.Gtorque(self.coilset['subcoil'], self.coilset['coil'],
                                source, sink, multi_filament) * self.Iscale**2
                self.Fa[i, j, 0] = 2 * np.pi * cc.mu_o * xG[1]  # cross product
                self.Fa[i, j, 1] = -2 * np.pi * cc.mu_o * xG[0]

    def set_passive_force_field(self):
        self.Fp = np.zeros((self.nC, 2))  # passive
        for i, sink in enumerate(self.active_coils):
            xB = cc.Btorque(self.coilset['subcoil'],
                            self.coilset['plasma_coil'],
                            self.passive_coils, sink) * self.Iscale
            self.Fp[i, 0] = 2 * np.pi * cc.mu_o * xB[1]  # cross product
            self.Fp[i, 1] = -2 * np.pi * cc.mu_o * xB[0]

    def set_force(self, Ic):  # evaluate coil force and force jacobian
        self.F = np.zeros((self.nC, 2))
        dF = np.zeros((self.nC, self.nC, 2))
        Im = np.dot(Ic.reshape(-1, 1), np.ones((1, self.nC)))  # current matrix
        for i in range(2):  # coil force (bundle of eq elements)
            self.F[:, i] = 1e-6 * \
                (Ic * (np.dot(self.Fa[:, :, i], Ic) + self.Fp[:, i]))  # MN
            dF[:, :, i] = Im * self.Fa[:, :, i]
            diag = np.dot(self.Fa[:, :, i], Ic) +\
                Ic * np.diag(self.Fa[:, :, i]) + self.Fp[:, i]
            np.fill_diagonal(dF[:, :, i], diag)
        dF *= 1e-6  # force jacobian MN/MA
        return self.F, dF

    def get_force(self):
        self.check()
        Ic = self.set_current()
        F, dF = self.set_force(Ic)
        Fcoil = {'PF': {'x': 0, 'z': 0,
                        'x_array': np.ndarray, 'z_array': np.ndarray},
                 'CS': {'sep_array': np.ndarray, 'sep': 0, 'zsum': 0,
                        'x_array': np.ndarray, 'z_array': np.ndarray},
                 'F': F, 'dF': dF}
        Fcoil['PF']['x_array'] = F[self.coilset['index']['PF']['index'], 0]
        Fcoil['PF']['z_array'] = F[self.coilset['index']['PF']['index'], 1]
        Fcoil['PF']['x'] = \
            np.max(abs(F[self.coilset['index']['PF']['index'], 0]))
        Fcoil['PF']['z'] = \
            np.max(abs(F[self.coilset['index']['PF']['index'], 1]))
        FzCS = F[self.coilset['index']['CS']['index'], 1]
        if self.coilset['index']['CS']['n'] > 1:
            # seperation force
            Fsep = np.zeros(self.coilset['index']['CS']['n'] - 1)
            for j in range(self.coilset['index']['CS']['n'] - 1):
                # evaluate each gap
                Fsep[j] = np.sum(FzCS[j + 1:]) - np.sum(FzCS[:j + 1])
            Fcoil['CS']['sep_array'] = Fsep
            Fcoil['CS']['sep'] = np.max(Fsep)
        Fcoil['CS']['zsum'] = np.sum(FzCS)
        self.Fcoil = Fcoil
        return self.Fcoil

    def set_current(self):  # set subcoil current vector from pf
        self.Ic = np.zeros(self.nC)
        for i, name in enumerate(self.active_coils):
            Nfilament = self.coilset['subcoil'][name + '_0']['Nf']
            # store current
            self.Ic[i] = self.coilset['coil'][name]['Ic'] / Nfilament
        self.Ic /= self.Iscale  # convert current unit (MA if Iscale=1e6)
        return self.Ic

    def plot(self, coils=['PF', 'CS'], scale=2, Fvector=['Fo'],
             **kwargs):
        # vector = 'Fo', 'Fx', 'Fz' or combination
        fs = matplotlib.rcParams['legend.fontsize']
        if not hasattr(self, 'F'):
            self.set_force(self.Ic)
        if 'Fmax' in kwargs:
            Fmax = kwargs['Fmax']
        else:
            Fmax = np.max(np.linalg.norm(self.F, axis=1))
        index, names = [], []
        for coil in coils:
            index.extend(self.coilset['index'][coil]['index'])
            names.extend(self.coilset['index'][coil]['name'])
        for i, name in zip(index, names):
            x = self.coilset['coil'][name]['x']
            z = self.coilset['coil'][name]['z']
            index = self.active_coils.index(name)
            F = self.F[index]
            Fvec = scale * F / Fmax
            for Fv, mag, fc in zip(['Fo', 'Fx', 'Fz'],
                                   [(1, 1), (1, 0), (0, 1)],
                                   [0.3*np.ones(3), 0.6*np.ones(3),
                                    0.6*np.ones(3)]):
                if Fv in Fvector:
                    plt.arrow(x, z, mag[0]*Fvec[0], mag[1]*Fvec[1],
                              linewidth=2, ec=0.4*fc, fc=fc, head_width=0.2,
                              lw=1.0, length_includes_head=False)
            if name in self.coilset['index']['PF']['name']:
                Fvec[1] += np.sign(Fvec[1]) * 0.5
                if abs(Fvec[1]) < self.coilset['coil'][name]['dz']:
                    Fvec[1] = np.sign(Fvec[1]) * \
                        self.coilset['coil'][name]['dz']
                if Fvec[1] > 0:
                    va = 'bottom'
                else:
                    va = 'top'
                plt.text(x, z + 0.75 * Fvec[1], '{:1.0f}MN'.format(F[1]),
                         ha='center', va=va, fontsize=fs,
                         color=0.1 * np.ones(3),
                         backgroundcolor=0.85 * np.ones(3))

    def plotCS(self, scale=1e-3):
        colors = [c['color'] for c in matplotlib.rcParams['axes.prop_cycle']]
        colors = [np.array(matplotlib.colors.hex2color(c)) for c in colors]
        Fmax = np.max(abs(self.Fcoil['CS']['sep_array']))
        names = self.coilset['index']['CS']['name']
        for i, Fsep in enumerate(self.Fcoil['CS']['sep_array']):
            txt = '{:1.0f}MN'.format(Fsep)
            color = colors[2] if Fsep < 0 else colors[3]
            lower = self.coilset['coil'][names[i]]  # lower
            upper = self.coilset['coil'][names[i+1]]  # upper
            z_lower = lower['z'] + lower['dz']/2
            z_upper = upper['z'] - upper['dz']/2
            z_gap = np.mean([z_lower, z_upper])
            x_gap = np.mean([lower['x'], upper['x']])
            dx = np.max([lower['dx'], upper['dx']])
            xvec = x_gap + np.linspace(-dx/2, dx/2, 5)
            for xo in xvec:
                for sign in [1, -1]:
                    zo = z_lower if sign * Fsep > 0 else z_upper
                    zarrow = -sign * np.sign(Fsep) * scale * Fsep / Fmax
                    zoffset = zarrow if Fsep < 0 else 0
                    include_head = Fsep < 0
                    plt.arrow(xo, zo-zoffset, 0, zarrow,
                              ec=0.4*color, fc=color,
                              head_width=0.15, lw=1.0,
                              length_includes_head=include_head)
            plt.text(x_gap - dx, z_gap, txt, color=0.2*np.ones(3),
                     va='center', ha='right',
                     backgroundcolor=0.85 * np.ones(3))

    def set_bm(self, cage):
        x = {'cl': {'x': cage.coil_loop[:, 0], 'z': cage.coil_loop[:, 2]}}
        i = np.argmax(x['cl']['z'])
        ro, zo = x['cl']['x'][i], x['cl']['z'][i]
        self.bm = ro * cage.point((ro, 0, zo),
                                  variable='field')[1]  # TF moment

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
        if method == 'function':  # calculate tf field as fitted 1/x function
            if not hasattr(self, 'bm'):
                self.set_bm(cage)
        elif method != 'BS':  # raise error if method not 'function' or 'BS'
            errtxt = 'invalid tf field method {}\n'.format(method)
            errtxt += 'select method from \'function\' or \'BS\'\n'
            raise ValueError(errtxt)
        if method == 'function':
            b = np.zeros(3)
            # TF field (fast version / only good for TF cl)
            b[1] = self.bm / (point[0])
        elif method == 'BS':  # calculate tf field with full Biot-Savart
            b = cage.point(point, variable='field')  # (slow / correct)
        theta = np.arctan2(point[1], point[0])
        # rotate to PF plane
        pf_point = np.dot(geom.rotate(-theta, 'z'), point)
        # PF field (sf-fast, eq-slow)
        pf_b = Bpoint([pf_point[0], pf_point[2]])
        b += np.dot(geom.rotate(theta, 'z'), [pf_b[0], 0, pf_b[1]])  # add PF
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

    ff = force_field(pf.index, pf.coil, pf.subcoil, pf.plasma_coil,
                     multi_filament=True)
    Fcoil = ff.get_force()

    ff.plot()
    pf.plot(label=True, plasma=False, current=True)
    pf.plot(subcoil=True, label=False, plasma=True, current=False)
    plt.axis('equal')
    plt.axis('off')
