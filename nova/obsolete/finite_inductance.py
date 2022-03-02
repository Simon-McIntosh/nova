import numpy as np
from scipy.integrate import odeint
from nova.coils import PF
from nova.inverse import INV
from collections import OrderedDict
from itertools import count
import pandas as pd
from amigo.pyplot import plt


class inductance:

    def __init__(self):
        self.ncp = 0  # number of constraints
        self.mpc = OrderedDict()  # initalize multi-point constraint
        self.pf = PF()  # initalize PF coil object
        self.nC = 0

    def add_coilset(self, coilset):
        self.pf.add_coilset(coilset)
        self.nC += len(coilset['coil'])

    def initalise_cp_set(self):
        name = 'cp{:d}'.format(self.ncp)  # cp set name
        if name not in self.mpc:  # create dof set
            self.mpc[name] = {}
            self.mpc[name]['nodes'] = np.array([], dtype=int)
            self.mpc[name]['dr'] = np.array([], dtype=int)
            self.mpc[name]['dc'] = np.array([], dtype=int)
            self.mpc[name]['nc'] = 0
        return name

    def add_cp(self, nodes, label=None, nset='next', antiphase=False):
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
        self.mpc[name]['label'] = label
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
        inv = INV(self.pf.coilset, Iscale=1)
        self.active_coils = np.copy(inv.adjust_coils)  # active coilset
        inv.inductance()  # calculate inductance matrix
        self.Mo = pd.DataFrame(inv.Mt, index=self.active_coils,
                               columns=self.active_coils)
        self.nMo = len(self.Mo)
        self.extract_discarge_data()

    def extract_discarge_data(self):
        # inital resistance, resistive mass, resistive material,
        # inital temperature,
        # current, turn-current, turn number, conductor cross section type
        variables = ['Ic', 'R', 'm', 'T', 'material']
        self.discarge_data_o = pd.DataFrame(index=self.active_coils,
                                            columns=variables)
        for name in self.active_coils:
            for var in variables:
                self.discarge_data_o.loc[name, var] = \
                    self.pf.coilset['coil'][name][var]

    def assemble_cp_nodes(self):
        self.cp_nd = {'dr': np.array([], dtype=int),
                      'dc': np.array([], dtype=int), 'n': 0}
        self.couple = OrderedDict()
        ieq = count(0)
        for name in self.mpc:
            if self.mpc[name]['label'] is not None:
                ndo = self.mpc[name]['nodes'][0]  # retained node
                self.active_coils[ndo] = self.mpc[name]['label']  # rename
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
        self.nd = {}  # node index
        self.nd['do'] = np.arange(0, self.nMo, dtype=int)  # all nodes
        self.nd['mask'] = np.zeros(self.nMo, dtype=bool)  # all nodes
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

        # active coilset
        self.active_coils = self.active_coils[~self.nd['mask']]  # retained

        # sort and couple M and Rc
        M = np.copy(self.Mo)
        M = np.append(M[self.nd['dr'], :], M[self.nd['dc'], :], axis=0)
        M = np.append(M[:, self.nd['dr']], M[:, self.nd['dc']], axis=1)
        self.M = pd.DataFrame(np.dot(np.dot(self.T.T, M), self.T),
                              index=self.active_coils,
                              columns=self.active_coils)
        # turn resistance
        '''
        R = np.copy(self.Ro)
        R = np.append(R[self.nd['dr']], R[self.nd['dc']], axis=0)
        self.R = pd.Series(np.dot(self.T.T, R), index=self.active_coils)
        '''

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

    def plot(self, ax=None, **kwargs):
        if ax is None:
            ax = plt.subplots(1, 1, figsize=(8, 10))[1]
        self.pf.plot(ax=ax, **kwargs)
