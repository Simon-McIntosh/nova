import numpy as np
from scipy.integrate import odeint
from amigo.pyplot import plt
from nova.coils import PF
from nova.inverse import INV
from collections import OrderedDict
from itertools import count
from nep.coil_geom import VSgeom, PFgeom
from nep.DINA.read_tor import read_tor
from scipy.interpolate import interp1d
from read_dina import timeconstant


class inductance:

    def __init__(self):
        self.ncp = 0  # number of constraints
        self.mpc = OrderedDict()  # multi-point constraint
        self.pf = PF()  # initalize PF coil object
        self.Ro = []  # coil resistance
        self.nt = []  # coil turns
        self.nC = 0

    def add_coil(self, x, z, dx, dz, Ic, R=0, nt=1):
        self.pf.add_coil(x, z, dx, dz, Ic, categorize=False)  # retain order
        self.Ro.append(R)
        self.nt.append(nt)
        self.nC += 1  # increment coil counter

    def get_coil(self, coil):
        x, z, dx, dz = coil['x'], coil['z'], coil['dx'], coil['dz']
        Ic = coil['Ic']
        return x, z, dx, dz, Ic

    def add_pf_coil(self, coil, R=[0], turns=None):
        if turns is None:
            turns = np.ones(len(coil))
        Rc, Ic = np.zeros(len(coil)), np.zeros(len(coil))
        Rc[:len(R)] = R  # resistance array
        for name, rc, nt in zip(coil, Rc, turns):
            x, z, dx, dz, Ic = self.get_coil(coil[name])
            self.add_coil(x, z, dx, dz, Ic, R=rc, nt=nt)

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
            self.pf.plot()
        self.inv = INV(self.pf, Iscale=1)
        self.inv.update_coils(categorize=False)
        Nf = np.ones(self.inv.nC)
        turns = np.array(self.nt)
        self.Io = np.ones(self.nC)
        for i, coil in enumerate(self.inv.adjust_coils):
            x, z = self.pf.coil[coil]['x'], self.pf.coil[coil]['z']
            self.inv.add_psi(1, point=(x, z))
            self.Io[i] = self.pf.coil[coil]['Ic']

        self.inv.set_foreground()
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

        # sort and couple M and Ic
        self.M = np.append(self.M[self.nd['dr'], :],
                           self.M[self.nd['dc'], :], axis=0)
        self.M = np.append(self.M[:, self.nd['dr']],
                           self.M[:, self.nd['dc']], axis=1)
        self.M = np.dot(np.dot(self.T.T, self.M), self.T)
        self.Minv = np.linalg.inv(self.M)

        self.Rc = np.append(self.Rc[self.nd['dr']],
                            self.Rc[self.nd['dc']], axis=0)
        self.Rc = np.dot(self.T.T, self.Rc)  # sum
        self.Ic = np.dot(self.T.T, self.Io)
        # self.Ic = self.Io[self.nd['dr']]  # retained inital currents


if __name__ is '__main__':

    Io = 60e3
    Ip = -15e6

    Lbb = 0.2e-3  # busbar inductance

    ind = inductance()
    vs_geom = VSgeom()
    pf_geom = PFgeom()

    tor_index = 0
    tor = read_tor('disruptions')
    tor.read_file(3)
    tor.set_current(tor_index)

    # ind.add_coil(5.3, 0, 0.1, 0.1, Ip, 0)  # plasma  3.2
    nvs_o = ind.nC
    Ip = 0
    for i, coil in enumerate(tor.plasma_coil[tor_index]):  # plasma
        x, z, dx, dz, Ic = ind.get_coil(tor.plasma_coil[tor_index][coil])
        Ip += Ic
        ind.add_coil(x, z, dx, dz, Ic, nt=1, R=0)
        if i > 0:
            ind.add_cp([nvs_o, nvs_o+i], antiphase=False)  # link coils
    print(Ip)
    nvs_o = ind.nC
    turns = np.append(np.ones(4), -np.ones(4))
    ind.add_pf_coil(vs_geom.pf.coil, R=[17.66e-3], turns=turns)
    vs3 = 'single'

    if vs3 == 'single':
        for i, index in enumerate(nvs_o+np.arange(1, 8)):  # vs3 loops
            ind.add_cp([nvs_o, index])  # link VS coils
    elif vs3 == 'dual':
        for i, index in enumerate(nvs_o+np.arange(1, 4)):  # lower loops
            ind.add_cp([nvs_o, index])
        for i, index in enumerate(nvs_o+np.arange(5, 8)):  # upper loops
            ind.add_cp([nvs_o+4, index])


    for coil in tor.coil:  # add pf coils
        x, z, dx, dz, Ic = ind.get_coil(tor.coil[coil])
        nt = pf_geom.coil[coil]['N']
        ind.add_coil(x, z, dx, dz, Ic, nt=nt, R=0)
    '''
    for i, coil in enumerate(tor.blanket_coil):
        x, z, dx, dz, Ic = ind.get_coil(tor.blanket_coil[coil])

        ind.add_coil(x, z, dx, dz, Ic, nt=nt, R=1e-3)
        if np.mod(i, 2) == 1:
            ind.add_cp([ind.nC-2, ind.nC-1], antiphase=True)  # couple

    Rvessel = 3e-3*np.ones(len(tor.vessel_coil))
    ind.add_pf_coil(tor.vessel_coil, R=Rvessel)
    ind.add_cp([ind.nC-3, ind.nC-2], antiphase=False)  # blanket support
    '''
    ind.assemble()
    ind.assemble_cp_nodes()
    ind.constrain()

    ind.Rc[0] = ind.M[0, 0]/16e-3  # update plasma resistance
    ind.Ic[1] = Io  # vs3 inital current
    ind.M[1, 1] += Lbb  # add busbar resistance

    tau = ind.M[1, 1] / 17.66e-3

    t = np.linspace(0, 0.3, 1000)

    def dIdt(Ic, t):
        # Idot = np.linalg.solve(-ind.M, Ic*ind.Rc)
        Idot = np.dot(-ind.Minv, Ic*ind.Rc)
        return Idot

    Ic_o = np.copy(ind.Ic)
    Ic_o[1] = 0  # zero inital current referance
    Ivs_o = odeint(dIdt, Ic_o, t).T[1]

    Iode = odeint(dIdt, ind.Ic, t)

    ax = plt.subplots(2, 1, sharex=True)[1]
    for i, Ic in enumerate(Iode.T[:2]):
        if i == 0:
            ax[0].plot(1e3*t, 1e-6*Ic)
        else:
            ax[1].plot(1e3*t, 1e-3*Ic)

    dIdt = np.gradient(Ivs_o, t)
    vs_o = Ivs_o*ind.Rc[1] + ind.M[1, 1]*dIdt  # voltage source
    vs_fun = interp1d(t, vs_o, fill_value='extrapolate')

    def dIdt_fun(Ic, t):
        g = (vs_fun(t) - Ic*ind.Rc[1])/ind.M[1, 1]
        return g

    # Iode = Ivs_o[0]+odeint(dIdt_fun, Io, t)
    # ax[1].plot(1e3*t, 1e-3*(Ivs_o + Io*np.exp(-t/tau)), '--')
    # ax[1].plot(1e3*t, 1e-3*Iode, '-.')
    ax[1].plot(1e3*t, 1e-3*Ivs_o, '-', color='gray')
    ax[1].plot(1e3*t, 1e-3*(Ic-Ivs_o), '-.', color='gray')

    tc = timeconstant(t, Ic-Ivs_o, trim_fraction=0.25)
    tdis, ttype, tfit, Ifit = tc.fit(plot=True, ax=ax[1])
    print('tau_o', 1e3*tau, 'ms')
    print('tau', 1e3*tdis, 'ms')

    ax[1].plot(1e3*t, 1e-3*(Ivs_o + Io*np.exp(-t/tdis)), '--')

    plt.figure()
    dIdt_ = np.gradient(Ic, t)

    plt.plot(1e3*t, 1e-6*Ivs_o**2*ind.M[1, 1], color='gray')
    plt.plot(1e3*t, 1e-6*(Ic-Ic[0])**2*ind.M[1, 1])