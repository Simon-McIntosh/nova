import numpy as np
from scipy.integrate import odeint
from amigo.pyplot import plt
from nova.coils import PF
from nova.inverse import INV
from collections import OrderedDict
from itertools import count
from nep.coil_geom import VSgeom, PFgeom
from nep.DINA.read_tor import read_tor
from nep.DINA.read_plasma import read_plasma


class inductance:

    def __init__(self):
        self.ncp = 0  # number of constraints
        self.mpc = OrderedDict()  # multi-point constraint
        self.pf = PF()  # initalize PF coil object
        self.Ro = []  # coil resistance
        self.nt = []  # coil turns
        self.nC = 0

    def add_coil(self, x, z, dx, dz, Ic, R=0, nt=1, **kwargs):
        self.pf.add_coil(x, z, dx, dz, Ic, categorize=False, **kwargs)
        self.Ro.append(R)
        self.nt.append(nt)
        self.nC += 1  # increment coil counter

    def get_coil(self, coil):
        x, z, dx, dz = coil['x'], coil['z'], coil['dx'], coil['dz']
        Ic = coil['Ic']
        if 'R' in coil:
            R = coil['R']
        else:
            R = 0
        return x, z, dx, dz, Ic, R

    def add_pf_coil(self, coil, turns=None):
        if turns is None:
            turns = np.ones(len(coil))
        for name, nt in zip(coil, turns):
            x, z, dx, dz, Ic, R = self.get_coil(coil[name])
            self.add_coil(x, z, dx, dz, Ic, R=R, nt=nt, name=name)

    def initalise_cp_set(self):
        name = 'cp{:d}'.format(self.ncp)  # cp set name
        if name not in self.mpc:  # create dof set
            self.mpc[name] = {}
            self.mpc[name]['nodes'] = np.array([], dtype=int)
            self.mpc[name]['dr'] = np.array([], dtype=int)
            self.mpc[name]['dc'] = np.array([], dtype=int)
            self.mpc[name]['nc'] = 0
        return name

    def add_cp(self, nodes, nset='next', antiphase=False):
        nodes = np.copy(nodes)  # make local copy
        if nset == 'next':  # (default)
            self.ncp += 1
        elif isinstance(nset, int):
            self.ncp = nset
        else:
            errtxt = 'nset must be interger or string, \'high\' or \'next\''
            raise ValueError(errtxt)
        if self.ncp == 0:
            self.ncp = 1
        name = self.initalise_cp_set()
        self.mpc[name]['nodes'] = np.append(self.mpc[name]['nodes'], nodes)
        self.mpc[name]['nc'] += len(nodes) - 1
        self.mpc[name]['neq'] = self.mpc[name]['nc']
        self.mpc[name]['Cc'] = -np.identity(self.mpc[name]['neq'])
        self.mpc[name]['Cr'] = np.identity(self.mpc[name]['neq'])
        if antiphase:  # coil pair run in antiphase
            self.mpc[name]['Cr'] *= -1
        self.extract_cp_nodes(self.mpc[name])

    def extract_cp_nodes(self, mpc):
        row = mpc['nodes']
        for j in range(mpc['nc']):
            mpc['dr'] = np.append(mpc['dr'], row[0])
            mpc['dc'] = np.append(mpc['dc'], row[1 + j])

    def assemble(self, plot=False):
        self.pf.mesh_coils()
        if plot:
            self.pf.plot(subcoil=True)
        self.inv = INV({'index': self.pf.index, 'coil': self.pf.coil,
                        'subcoil': self.pf.subcoil,
                        'plasma_coil': self.pf.plasma_coil}, Iscale=1)
        self.inv.update_coils()
        Nf = np.ones(self.inv.nC)
        turns = np.array(self.nt)
        self.Io = np.ones(self.nC)
        for i, coil in enumerate(self.inv.adjust_coils):
            x, z = self.pf.coil[coil]['x'], self.pf.coil[coil]['z']
            self.inv.add_psi(1, point=(x, z))
            self.Io[i] = self.pf.coil[coil]['Ic']
        self.inv.set_foreground()
        # ensure symetric (jacket-conductor coupling)
        tril_index = np.tril_indices(len(self.inv.G), k=-1)  # lower triangle
        self.inv.G[tril_index] = self.inv.G.T[tril_index]  # duplicate upper tr
        t2 = np.dot(turns.reshape((-1, 1)), turns.reshape((1, -1)))
        fillaments = np.dot(np.ones((len(turns), 1)), Nf.reshape(1, -1))
        self.Mo = 2 * np.pi * self.inv.G * t2 * fillaments
        self.nM = len(self.Mo)

    def assemble_cp_nodes(self):
        self.cp_nd = {'dr': np.array([], dtype=int),
                      'dc': np.array([], dtype=int), 'n': 0}
        self.couple = OrderedDict()
        ieq = count(0)
        for name in self.mpc:
            for node in ['dr', 'dc']:
                self.cp_nd[node] = np.append(self.cp_nd[node],
                                             self.mpc[name][node])
            self.cp_nd['n'] += self.mpc[name]['neq']
            for i, (Cr, Cc) in enumerate(zip(self.mpc[name]['Cr'],
                                             self.mpc[name]['Cc'])):
                eqname = 'eq{:d}'.format(next(ieq))
                self.couple[eqname] = {'dr': self.mpc[name]['dr'], 'Cr': Cr,
                                       'dc': self.mpc[name]['dc'], 'Cc': Cc,
                                       'dro': self.mpc[name]['dr'][i]}

    def constrain(self):  # apply and colapse constraints
        self.assemble_cp_nodes()
        self.M = np.copy(self.Mo)
        self.Rc = np.copy(self.Ro)
        self.nd = {}  # node index
        self.nd['do'] = np.arange(0, self.nM, dtype=int)  # all nodes
        self.nd['mask'] = np.zeros(self.nM, dtype=bool)  # all nodes
        self.nd['mask'][self.cp_nd['dc']] = True  # condense
        self.nd['dc'] = self.nd['do'][self.nd['mask']]  # condensed
        self.nd['dr'] = self.nd['do'][~self.nd['mask']]  # retained
        self.nd['nc'] = np.sum(self.nd['mask'])
        self.nd['nr'] = np.sum(~self.nd['mask'])
        self.Cc = np.zeros((self.cp_nd['n'], self.cp_nd['n']))
        self.Cr = np.zeros((self.cp_nd['n'], self.nd['nr']))
        for i, name in enumerate(self.couple):
            couple = self.couple[name]
            self.Cr[i, np.in1d(self.nd['dr'], couple['dr'])] = couple['Cr']
            self.Cc[i, np.in1d(self.cp_nd['dc'], couple['dc'])] = couple['Cc']

        # build transformation matrix
        self.Tc = np.zeros((self.nd['nc'], self.nd['nr']))  # initalise
        index = np.in1d(self.nd['dc'], self.cp_nd['dc'])
        self.Tc[index, :] = np.dot(-np.linalg.inv(self.Cc), self.Cr)
        self.T = np.append(np.identity(self.nd['nr']), self.Tc, axis=0)

        # sort and couple M, Rc and Ic
        self.M = np.append(self.M[self.nd['dr'], :],
                           self.M[self.nd['dc'], :], axis=0)
        self.M = np.append(self.M[:, self.nd['dr']],
                           self.M[:, self.nd['dc']], axis=1)
        self.M = np.dot(np.dot(self.T.T, self.M), self.T)
        # turn resistance
        self.Rc = np.append(self.Rc[self.nd['dr']],
                            self.Rc[self.nd['dc']], axis=0)
        self.Rc = np.dot(self.T.T, self.Rc)  # sum
        # turn current
        self.Ic = np.append(self.Io[self.nd['dr']],
                            self.Io[self.nd['dc']], axis=0)
        self.Ic = np.dot(self.T.T, self.Ic)  # sum

    def dIdt(self, Ic, t, *args):  # current rate (function for odeint)
        vfun = args[0]
        if vfun is None:
            vbg = np.zeros(len(Ic))  # background field
        else:
            vbg = np.array([vf(t) for vf in vfun])

        Idot = np.dot(self.Minv, vbg - Ic*self.Rc)
        return Idot

    def reduce(self):
        self.assemble()
        self.constrain()

    def solve(self, t, **kwargs):
        self.Minv = np.linalg.inv(self.M)  # inverse for odeint
        vfun = kwargs.get('vfun', None)
        Iode = odeint(self.dIdt, self.Ic, t, (vfun,)).T
        return Iode

    def plot(self, **kwargs):
        self.pf.plot(**kwargs)


if __name__ is '__main__':

    Io = -60e3
    Ip = -15e6

    Lbb = 0.2e-3  # busbar inductance

    ind = inductance()
    vs_geom = VSgeom()
    pf_geom = PFgeom()

    # 0: 'MD_DW_exp22'
    # 1: 'MD_DW_lin36'
    # 2: 'MD_DW_lin50'
    # 3: 'MD_UP_exp16'
    # 4: 'MD_UP_exp22'
    # 5: 'MD_UP_lin50'
    # 6: 'VDE_DW_fast'
    # 7: 'VDE_DW_slow',
    # 8: 'VDE_DW_slow_fast'
    # 9: 'VDE_UP_fast'
    # 10: 'VDE_UP_slow'
    # 11: 'VDE_UP_slow_fast'

    #file_index = 11
    for file_index in [3]:#range(12):
        pl = read_plasma('disruptions')
        # load Ivs3
        trip, Ivs3_data, Ivs3, Ivs3_fun = pl.Ivs3_single(file_index)
        t_cq = trip['t_cq']  # trip
        Ivs_o = Ivs3_data['Icontrol'][trip['i_cq']]  # inital vs3 current

        tor = read_tor('disruptions')
        tor.load_file(file_index)
        tor_index = np.argmin(abs(tor.t-t_cq))
        tor.set_current(tor_index)
        '''
        nvs_o = ind.nC
        npl = len(tor.plasma_coil[tor_index])
        for i, coil in enumerate(tor.plasma_coil[tor_index]):  # plasma
            x, z, dx, dz, Ic, R = \
                ind.get_coil(tor.plasma_coil[tor_index][coil])
            ind.add_coil(x, z, dx, dz, Ic, R=R, nt=1/npl)
            if i > 0:
                ind.add_cp([nvs_o, nvs_o+i])  # link coils
        '''

        nvs_o = ind.nC
        turns = np.append(np.ones(4), -np.ones(4))
        ind.add_pf_coil(vs_geom.pf.subcoil, turns=turns)
        vs3 = 'single'

        if vs3 == 'single':
            for i, index in enumerate(nvs_o+np.arange(1, 8)):  # vs3 loops
                ind.add_cp([nvs_o, index])  # link VS coils
        elif vs3 == 'dual':
            for i, index in enumerate(nvs_o+np.arange(1, 4)):  # lower loops
                ind.add_cp([nvs_o, index])
            for i, index in enumerate(nvs_o+np.arange(5, 8)):  # upper loops
                ind.add_cp([nvs_o+4, index])
        '''
        for coil in tor.coil:  # add pf coils
            x, z, dx, dz, Ic, R = ind.get_coil(tor.coil[coil])
            nt = pf_geom.coil[coil]['N']
            ind.add_coil(x, z, dx, dz, Ic, R=R, nt=nt)

        for i, coil in enumerate(tor.blanket_coil):  # blanket
            x, z, dx, dz, Ic, R = ind.get_coil(tor.blanket_coil[coil])
            nt = 1 if np.mod(i, 2) == 0 else -1
            ind.add_coil(x, z, dx, dz, Ic, R=R, nt=nt)
            if np.mod(i, 2) == 1:
                ind.add_cp([ind.nC-2, ind.nC-1])  # couple

        Rvessel = 1e-3*np.ones(len(tor.vessel_coil))  # vessel
        ind.add_pf_coil(tor.vessel_coil)
        ind.add_cp([ind.nC-3, ind.nC-2])  # blanket support
        '''
        ind.assemble()
        ind.assemble_cp_nodes()
        ind.constrain()

        '''
        ind.Rc[0] = ind.M[0, 0]/16e-3  # update plasma resistance
        ind.Ic[1] = Ivs_o + Io  # add vs3 current
        ind.Rc[1] = 17.66e-3  # total vs3 resistance
        ind.M[1, 1] += Lbb  # add busbar inductance
        tau = ind.M[1, 1] / ind.Rc[1]
        '''

        '''
        t = np.linspace(0, 0.3, 300)

        Ic_o = np.copy(ind.Ic)
        Ic_o[1] -= Io  # subtract inital current delta

        ind.Minv = np.linalg.inv(ind.M)  # inverse for odeint

        Ivs_o = odeint(ind.dIdt, Ic_o, t).T[1]
        Iode = odeint(ind.dIdt, ind.Ic, t)
        Ic = Iode.T[1, :]

        dIdt = np.gradient(Ivs_o, t)
        vs_o = Ivs_o*ind.Rc[1] + ind.M[1, 1]*dIdt  # voltage source
        vs_fun = interp1d(t, vs_o, fill_value='extrapolate')

        tc = timeconstant(t, Ic-Ivs_o, trim_fraction=0)  # diffrence tau
        Io_d, tau_d, tfit, Ifit = tc.nfit(3)
        txt_d = timeconstant.ntxt(Io_d/tc.Id[0], tau_d)
        ax = plt.subplots(1, 1, sharex=True)[1]
        text = linelabel(value='', postfix='ms')

        ax.plot(1e3*t, 1e-3*Ivs_o, '-', color='gray', label='$I_{DINA}$')
        ax.plot(1e3*t, 1e-3*Ic, 'C3',
                label='$I_o+$'+'{:1.0f}kA'.format(1e-3*Io))

        ax.plot(1e3*t, 1e-3*(Ic-Ivs_o), '-', color='C0',
                label=r'$\Delta I$')
        ax.plot(1e3*tfit, 1e-3*Ifit, 'C1-.', label='exp fit '+txt_d)
        ax.plot(1e3*t, 1e-3*(Ivs_o + Ifit), 'C4-.',
                label='overall fit')
        plt.legend()
        plt.despine()
        plt.xlabel('$t$ ms')
        plt.ylabel('$I$ kA')
        plt.ylim([-90, 0])

        # print('tau_o', 1e3*tau, 'ms')
        # print('tau', 1e3*tdis, 'ms')

        max_index = np.argmax(abs(Ic))
        print('Ivs3 max {:1.1f}kA'.format(1e-3*Ic[max_index]))
        '''

        plt.figure(figsize=(8, 10))
        ind.pf.plot()
        ax = plt.gca()
        txt = tor.name
        if ind.nd['nr'] > 1:
            if vs3 == 'single':
                txt += '\nplasma-vs3: {:1.1f}'.format(1e6*ind.M[0, 1])
                txt += r'$\mu$H'
            elif vs3 == 'dual':
                txt += '\nplasma-upper: {:1.1f}'.format(1e6*ind.M[0, 2])
                txt += r'$\mu$H'
                txt += '\nplasma-lower: {:1.1f}'.format(1e6*ind.M[0, 1])
                txt += r'$\mu$H'
            else:
                txt += '\nplasma-vs3: {:1.1f}'.format(1e6*ind.M[0, 1])
                txt += r'$\mu$H'
            ax.text(0.5, 1.0, txt, transform=ax.transAxes,
                    ha='center', va='bottom',
                    bbox=dict(facecolor='lightgray'))
