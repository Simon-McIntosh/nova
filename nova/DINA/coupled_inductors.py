import numpy as np
from scipy.integrate import odeint
from nova.coils import PF
from nova.inverse import INV
from collections import OrderedDict
from itertools import count


class inductance:

    def __init__(self):
        self.ncp = 0  # number of constraints
        self.mpc = OrderedDict()  # multi-point constraint
        self.pf = PF()  # initalize PF coil object
        self.Ro = []  # coil resistance
        self.nt = []  # coil turns
        self.nC = 0

    def add_coil(self, x, z, dx, dz, It, R=0, nt=1, **kwargs):
        self.pf.add_coil(x, z, dx, dz, It, categorize=False, **kwargs)
        self.Ro.append(R)
        self.nt.append(nt)
        self.nC += 1  # increment coil counter

    def get_coil(self, coil):
        x, z, dx, dz = coil['x'], coil['z'], coil['dx'], coil['dz']
        if 'It' in coil:
            It = coil['It']
        else:
            It = 0
        if 'R' in coil:
            R = coil['R']
        else:
            R = 0
        if 'Nt' in coil:
            Nt = coil['Nt']
        else:
            Nt = 1
        return x, z, dx, dz, It, R, Nt

    def add_pf_coil(self, coil, turns=None):
        if turns is None:
            turns = np.ones(len(coil))
        for name, nt in zip(coil, turns):
            x, z, dx, dz, It, R, Nt = self.get_coil(coil[name])
            if nt == 1:
                nt = Nt
            self.add_coil(x, z, dx, dz, It, R=R, nt=nt, name=name)

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
        self.pf.mesh_coils(dCoil=0.25)
        if plot:
            self.pf.plot(subcoil=True)
        self.inv = INV(self.pf.coilset, Iscale=1)
        self.inv.update_coils()
        Nf = np.ones(self.inv.nC)
        turns = np.array(self.nt)
        self.Io = np.ones(self.nC)
        for i, coil in enumerate(self.inv.adjust_coils):
            x = self.pf.coilset['coil'][coil]['x']
            z = self.pf.coilset['coil'][coil]['z']
            self.inv.add_psi(1, point=(x, z))
            self.Io[i] = self.pf.coilset['coil'][coil]['It']
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

        # sort and couple M, Rc and It
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
        self.It = np.append(self.Io[self.nd['dr']],
                            self.Io[self.nd['dc']], axis=0)
        self.It = np.dot(self.T.T, self.It)  # sum

    def dIdt(self, It, t, *args):  # current rate (function for odeint)
        vfun = args[0]
        if vfun is None:
            vbg = np.zeros(len(It))  # background field
        else:
            vbg = np.array([vf(t) for vf in vfun])

        Idot = np.dot(self.Minv, vbg - It*self.Rc)
        return Idot

    def reduce(self):
        self.assemble()
        self.constrain()

    def solve(self, t, **kwargs):
        self.Minv = np.linalg.inv(self.M)  # inverse for odeint
        vfun = kwargs.get('vfun', None)
        Iode = odeint(self.dIdt, self.It, t, (vfun,)).T
        return Iode

    def plot(self, **kwargs):
        self.pf.plot(**kwargs)
