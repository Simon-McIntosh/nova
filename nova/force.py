import nova.cross_coil as cc
from amigo.pyplot import plt, arrow_arc
import numpy as np
import matplotlib
from amigo import geom
import pandas as pd
from math import isclose
from amigo.time import clock


class force_field(object):

    def __init__(self, coilset, Iscale=1e6, multi_filament=True,
                 plot=False, **kwargs):
        self.coilset = coilset  # requires update
        self.Iscale = Iscale  # current units (MA)
        self.passive_coils = kwargs.get('passive_coils', ['Plasma'])
        active_coils = kwargs.get('active_coils',
                                  list(self.coilset['coil'].keys()))
        self.active_coils = [name for name in active_coils if name not in
                             self.passive_coils]
        self.nC = len(self.active_coils)  # number of active coils
        self.nC_filament = np.sum([self.coilset['coil'][name]['Nf']
                                   for name in self.active_coils])
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
        if 'force' not in self.coilset:
            self.coilset['force'] = {}
        force = ['Fa', 'Fa_filament', 'Fp', 'Fp_filament']
        if np.array([key in self.coilset['force'] for key in force]).all():
            self.force_field_active = True
            for key in force:
                setattr(self, key, self.coilset['force'][key])
            nC = np.shape(self.Fa)[0]
            if nC != self.nC:
                err_txt = 'active coil force matrix shape {}\n'.format(nC)
                err_txt += 'incompatible with number of'
                err_txt += 'active {} coils'.format(self.nC)
                raise IndexError(err_txt)
        else:
            self.set_force_field(state='both', multi_filament=multi_filament)

    def set_force_field(self, state='both', multi_filament=True):
        # [Ic]T([Fa][Ic]+[Fp]) = F
        self.force_field_active = True
        if state == 'both' or state == 'active':
            self.set_active_force_field(multi_filament=multi_filament)
        if state == 'both' or state == 'passive':
            self.set_passive_force_field()

    def set_active_force_field(self, multi_filament=True):
        io = 0
        self.Fa = np.zeros((self.nC, self.nC, 4))  # active coil
        self.Fa_filament = np.zeros((self.nC_filament,
                                     self.nC, 2))  # active filament
        tick = clock(self.nC**2, header='computing active force field')
        for i, sink in enumerate(self.active_coils):
            for j, source in enumerate(self.active_coils):
                self.set_active_coil(i, j, source, sink, multi_filament)
                self.set_active_filament(io, j, source, sink)
                tick.tock()
            io += self.coilset['coil'][sink]['Nf']

    def set_active_coil(self, i, j, source, sink, multi_filament):
        xG = cc.Gtorque(self.coilset['coil'], self.coilset['subcoil'],
                        source, sink, multi_filament)
        xG *= self.Iscale**2
        self.Fa[i, j, 0] = xG[1]  # Fx
        self.Fa[i, j, 1] = -xG[0]  # Fz
        self.Fa[i, j, 2] = xG[2]  # moment
        self.Fa[i, j, 3] = xG[3]  # z crush

    def set_active_filament(self, io, j, source, sink):
        for k in range(self.coilset['coil'][sink]['Nf']):
            xG = cc.Gtorque(self.coilset['coil'], self.coilset['subcoil'],
                            source, sink, True, Nsink=[k])
            xG *= self.Iscale**2
            self.Fa_filament[io+k, j, 0] = xG[1]  # Fx
            self.Fa_filament[io+k, j, 1] = -xG[0]  # Fz

    def set_passive_force_field(self):
        io = 0
        self.Fp = np.zeros((self.nC, 4))  # passive coil
        self.Fp_filament = np.zeros((self.nC_filament, 2))  # passive filament
        for i, sink in enumerate(self.active_coils):
            self.set_passive_coil(i, sink)
            self.set_passive_filament(io, sink)
            io += self.coilset['coil'][sink]['Nf']

    def set_passive_coil(self, i, sink):
        xB = cc.Btorque(self.coilset['coil'], self.coilset['subcoil'],
                        self.coilset['plasma'], self.passive_coils, sink)
        xB *= self.Iscale
        self.Fp[i, 0] = xB[1]  # Fx
        self.Fp[i, 1] = -xB[0]  # Fz
        self.Fp[i, 2] = xB[2]  # moment
        self.Fp[i, 3] = xB[3]  # z crush

    def set_passive_filament(self, io, sink):
        for k in range(self.coilset['coil'][sink]['Nf']):
            xB = cc.Btorque(self.coilset['coil'], self.coilset['subcoil'],
                            self.coilset['plasma'], self.passive_coils,
                            sink, Nsink=[k])
            xB *= self.Iscale
            self.Fp_filament[io+k, 0] = xB[1]  # Fx
            self.Fp_filament[io+k, 1] = -xB[0]  # Fz

    def set_force(self):  # evaluate coil force and force jacobian
        self.F = np.zeros((self.nC, 4))
        dF = np.zeros((self.nC, self.nC, 4))
        # current matrix
        Im = np.dot(self.If.reshape(-1, 1), np.ones((1, self.nC)))
        for i in range(4):  # coil force (bundle of eq elements)
            self.F[:, i] = self.If * (np.dot(self.Fa[:, :, i], self.If) +
                                      self.Fp[:, i])
            dF[:, :, i] = Im * self.Fa[:, :, i]
            diag = np.dot(self.Fa[:, :, i], self.If) +\
                self.If * np.diag(self.Fa[:, :, i]) + self.Fp[:, i]
            np.fill_diagonal(dF[:, :, i], diag)
        self.F /= self.Iscale  # force MN (iff Iscale set to 1e6)
        dF /= self.Iscale  # force jacobian MN/MA (iff Iscale set to 1e6)
        return self.F, dF

    def set_force_filament(self):  # evaluate coil force on filament level
        self.F_filament = np.zeros((self.nC_filament, 2))
        If_filament = np.array([])
        for name, If in zip(self.coilset['coil'], self.If):
            Nf = self.coilset['coil'][name]['Nf']
            If_filament = np.append(If_filament, If*np.ones(Nf))
        for i in range(2):
            self.F_filament[:, i] = \
                If_filament * (np.dot(self.Fa_filament[:, :, i], self.If) +
                               self.Fp_filament[:, i])
        self.F_filament /= self.Iscale  # force MN (iff Iscale set to 1e6)
        for i, name in enumerate(self.coilset['subcoil']):  # populate subcoil
            coil = self.coilset['coil']['_'.join(name.split('_')[:-1])]
            Nt, Nf = coil['Nt'], coil['Nf']
            subcoil = self.coilset['subcoil'][name]
            length = 2*np.pi*subcoil['x']  # coil length
            for j, var in enumerate(['x', 'z']):
                subcoil[f'Ff{var}'] = self.F_filament[i, j]  # per filament
                subcoil[f'Ff_length{var}'] = subcoil[f'Ff{var}'] / length
                subcoil[f'Ft{var}'] = subcoil[f'Ff{var}'] * Nf / Nt  # per turn
                subcoil[f'Ft_length{var}'] = subcoil[f'Ft{var}'] / length

    def get_force(self):
        self.check()
        self.set_current()
        F, dF = self.set_force()
        self.set_force_filament()
        Fcoil = {'PF': {'x': 0, 'z': 0,
                        'x_array': np.ndarray, 'z_array': np.ndarray,
                        'moment_array': np.ndarray},
                 'CS': {'sep_array': np.ndarray, 'sep': 0, 'zsum': 0,
                        'x_array': np.ndarray, 'z_array': np.ndarray,
                        'c_array': np.ndarray,
                        'moment_array': np.ndarray,
                        'axial': np.ndarray},
                 'F': F, 'dF': dF}
        Fcoil['PF']['x_array'] = F[self.coilset['index']['PF']['index'], 0]
        Fcoil['PF']['z_array'] = F[self.coilset['index']['PF']['index'], 1]
        Fcoil['PF']['moment_array'] = \
            F[self.coilset['index']['PF']['index'], 2]
        Fcoil['PF']['x'] = \
            np.max(abs(F[self.coilset['index']['PF']['index'], 0]))
        Fcoil['PF']['z'] = \
            np.max(abs(F[self.coilset['index']['PF']['index'], 1]))
        Fcoil['CS']['x_array'] = F[self.coilset['index']['CS']['index'], 0]
        Fcoil['CS']['z_array'] = F[self.coilset['index']['CS']['index'], 1]
        Fcoil['CS']['c_array'] = F[self.coilset['index']['CS']['index'], 3]
        Fcoil['CS']['moment_array'] = \
            F[self.coilset['index']['CS']['index'], 2]
        FzCS = F[self.coilset['index']['CS']['index'], 1]
        Fcoil['CS']['zsum'] = np.sum(FzCS)
        if self.coilset['index']['CS']['n'] > 1:
            # seperation force
            Fsep = np.zeros(self.coilset['index']['CS']['n'] + 1)
            for j in range(self.coilset['index']['CS']['n'] - 1):
                # evaluate each gap
                Fsep[j+1] = np.sum(FzCS[j + 1:]) - np.sum(FzCS[:j + 1])
            Fsep[0] = Fcoil['CS']['zsum']
            Fsep[-1] = -Fcoil['CS']['zsum']
            Fcoil['CS']['sep_array'] = Fsep
            Fcoil['CS']['sep'] = np.max(Fsep)
        self.Fcoil = Fcoil

    def set_axial_stack(self):
        mg = 1.182  # module weight MN
        Ftp = -201.33  # tie-plate pre-load
        alpha = 6.85e-3  # alpha Fx (Poisson)
        beta = np.array([-1.78e-1, -1.45e-1, -1.13e-1,
                         -8.12e-2, -4.89e-2, -1.65e-2])  # beta Fz
        gamma = -2.95e-2  # gamma Fc (crush)
        CScoils = self.coilset['index']['CS']['name']
        Fx = pd.DataFrame(self.Fcoil['CS']['x_array'].reshape(1, -1),
                          columns=CScoils)
        Fz = pd.DataFrame(self.Fcoil['CS']['z_array'].reshape(1, -1),
                          columns=CScoils)
        Fc = pd.DataFrame(self.Fcoil['CS']['c_array'].reshape(1, -1),
                          columns=CScoils)
        CScaps = np.append(np.append('LDP', CScoils), 'LDP')
        CSgaps = [f'{CScaps[i]}_{CScaps[i+1]}'
                  for i in range(len(CScoils)+1)]
        self.FCSz = pd.DataFrame(columns=CSgaps)
        self.FCSz['CS3U_LDP'] = alpha * Fx.sum(axis=1)
        self.FCSz['CS3U_LDP'] += (beta * Fz).sum(axis=1)
        self.FCSz['CS3U_LDP'] += gamma * Fc.sum(axis=1)
        self.FCSz['CS3U_LDP'] += Ftp
        uppergap = 'CS3U_LDP'
        for coil, gap in zip(CScoils[::-1], CSgaps[::-1][1:]):
            self.FCSz[gap] = self.FCSz[uppergap] + Fz.loc[:, coil] - mg
            uppergap = gap
        Fo = 16.8242
        dFdF = -0.5263
        Flimit = Fo + dFdF * self.FCSz['LDP_CS3L']
        self.FCSz['base'] = abs(self.Fcoil['CS']['zsum']) / Flimit

    def set_current(self):  # set subcoil current vector from pf
        self.If = np.zeros(self.nC)
        for i, name in enumerate(self.active_coils):
            Nf = self.coilset['coil'][name]['Nf']
            self.If[i] = self.coilset['coil'][name]['It'] / Nf
        self.If /= self.Iscale  # convert current unit (MA if Iscale=1e6)
        return self.If

    @staticmethod
    def plot_force_vector(x, z, Fvec, Fvector=['Fo'], alpha=1, width=0.2,
                          ax=None):
        #  Fvector = ['Fo', 'Fx', 'Fz']
        if ax is None:
            ax = plt.gca()
        for Fv, mag, fc in zip(
                ['Fo', 'Fx', 'Fz'], [(1, 1), (1, 0), (0, 1)],
                [0.3*np.ones(3), 0.6*np.ones(3), 0.6*np.ones(3)]):
            if Fv in Fvector:
                ax.arrow(x, z, mag[0]*Fvec[0], mag[1]*Fvec[1],
                         linewidth=1, ec=0.4*fc, fc=fc, head_width=width,
                         lw=1.0, length_includes_head=False, alpha=alpha)
                ax.plot(x+mag[0]*Fvec[0], z+mag[1]*Fvec[1], '.k', alpha=0)

    @staticmethod
    def force_label_PF(coil, F, Fvec):
        #  coil = coilset['coil'][name], Fz = Fvec[1]
        F_label = Fvec.copy()
        Fz_label = F_label[1] + np.sign(F_label[1]) * 0.5
        x, z = coil['x'], coil['z']
        if abs(Fz_label) < coil['dz']:
            Fz_label = np.sign(Fz_label) * coil['dz']
        if isclose(Fz_label, 0):  # protect zero force placment
            Fz_label = coil['dz']
        if Fz_label > 0:
            va = 'bottom'
        else:
            va = 'top'
        fs = matplotlib.rcParams['legend.fontsize']
        plt.text(x, z + 0.75 * Fz_label, '{:1.0f}MN'.format(F[1]),
                 ha='center', va=va, fontsize=fs, color=0.1 * np.ones(3),
                 backgroundcolor=0.85 * np.ones(3))

    def get_Fmax(self, **kwargs):
        if not hasattr(self, 'F'):
            self.set_force()
        if 'Fmax' in kwargs:
            Fmax = kwargs['Fmax']
        else:
            Fmax = np.max(np.linalg.norm(self.F[:, :2], axis=1))
        if isclose(Fmax, 0):
            Fmax = 1
        return Fmax

    def plot(self, coils=['PF', 'CS'], scale=2, Fvector=['Fo'],
             label=False, **kwargs):
        # vector = 'Fo', 'Fx', 'Fz' or combination
        Fmax = self.get_Fmax(**kwargs)
        index, names = [], []
        for coil in coils:
            mask = [name in self.active_coils for name in
                    self.coilset['index'][coil]['name']]
            index.extend(self.coilset['index'][coil]['index'][mask])
            names.extend(self.coilset['index'][coil]['name'][mask])

        for name in names:
            x = self.coilset['coil'][name]['x']
            z = self.coilset['coil'][name]['z']
            index = self.active_coils.index(name)
            F = self.F[index]
            Fvec = scale * F / Fmax
            self.plot_force_vector(x, z, Fvec, Fvector=Fvector)
            if name in self.coilset['index']['PF']['name'] and label:
                self.force_label_PF(self.coilset['coil'][name], F, Fvec)

    def plotCS(self, scale=1e-3):
        fs = matplotlib.rcParams['legend.fontsize']
        colors = [c['color'] for c in matplotlib.rcParams['axes.prop_cycle']]
        colors = [np.array(matplotlib.colors.hex2color(c)) for c in colors]
        Fmax = np.max(abs(self.Fcoil['CS']['sep_array']))
        if isclose(Fmax, 0):
            Fmax = 1
        names = self.coilset['index']['CS']['name']
        for i, Fsep in enumerate(self.Fcoil['CS']['sep_array'][1:-1]):
            txt = '{:1.1f}MN'.format(Fsep)
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
                     va='center', ha='right', fontsize=fs,
                     backgroundcolor=0.85 * np.ones(3))

    def plot_single(self, name, scale=1, ax=None, **kwargs):
        Nf = self.coilset['index'][name]['n']
        Nt = self.coilset['coil'][name]['Nt']
        coil_index = list(self.coilset['coil'].keys()).index(name)
        Fmax = np.linalg.norm(self.F[coil_index, :2])
        Fmax /= Nt  # per turn
        names = self.coilset['index'][name]['name']
        for name in names:
            x = self.coilset['subcoil'][name]['x']
            z = self.coilset['subcoil'][name]['z']
            F = np.array([self.coilset['subcoil'][name]['Ffx'],
                          self.coilset['subcoil'][name]['Ffz']])
            Fvec = scale * F / Fmax * Nf / Nt
            self.plot_force_vector(x, z, Fvec, Fvector=['Fo'], ax=ax)

    def plot_moment(self, scale=5e-1):
        Mmax = np.max(abs(self.Fcoil['CS']['moment_array']))
        for name, M in zip(self.coilset['index']['CS']['name'],
                           self.Fcoil['CS']['moment_array']):
            x = self.coilset['coil'][name]['x']
            z = self.coilset['coil'][name]['z']
            color = 'C3' if M > 0 else 'C2'

            arrow_arc(x, z, scale*M/Mmax, color=color)
            # arrow_arc(x, z, np.sign(M), color=color)

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

    setup = Setup('SN_3PF_18TF')

    sf = SF(filename=setup.filename)
    pf = PF(sf.eqdsk)
    pf.mesh_coils(dCoil=0.75)
    eq = EQ(pf.coilset, sf.eqdsk, sigma=0,
            boundary=sf.eq_boundary(expand=0.25), n=1e3)
    eq.get_plasma()
    # eq.gen_opp()

    ff = force_field(pf.coilset, multi_filament=True)
    Fcoil = ff.get_force()

    ff.plot()
    ff.plotCS()
    pf.plot(label=True, plasma=False, current=True)
    pf.plot(subcoil=True, label=False, plasma=True, current=False)
    plt.axis('equal')
    plt.axis('off')

