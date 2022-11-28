import pytest
import numpy as np

from nova.structural.finiteframe import finiteframe
#from nova.structural.properties import secondmoment
from nova.utilities import geom
import matplotlib.pyplot as plt


class testbeam(finiteframe):

    def __init__(self, name, N=27, theta=0):
        finiteframe.__init__(self, frame='3D')
        self.name = name
        self.N = N  # node number
        self.theta = theta  # frame rotation
        self.set_section()
        self.bar()  # add nodes / elements
        self.analytic = {}

    def set_section(self):
        self.add_shape('circ', r=0.02, ro=0.01)
        self.add_mat('tube', ['steel_cast'], [self.section])

        # extract sectional properties
        self.w = -9.81 * self.mat[0]['mat_o']['rho'] *\
            self.mat[0]['mat_o']['A']  # weight / l
        E = self.mat[0]['mat_o']['E']
        A = self.mat[0]['mat_o']['A']
        Iy = self.mat[0]['mat_o']['I'][1]
        self.EI = E * Iy
        self.EA = E * A

    def bar(self, L=3, **kwargs):
        self.L = L
        #self.x = np.linspace(0, self.L, 50)
        X = np.zeros((self.N, 3))
        X[:, 0] = np.linspace(0, self.L, self.N)
        X = geom.qrotate(X, theta=kwargs.get('theta', self.theta), dx='y')
        self.add_nodes(X)
        self.add_elements(part_name='beam', nmat='tube')
        self.update_rotation()  # check / update rotation matrix

    @property
    def g(self):
        return geom.qrotate([0, 0, -1], theta=self.theta, dx='y')[0]

    def add_weight(self):
        finiteframe.add_weight(self, g=self.g)

    def extract_model(self, i=2):
        if self.name == 'axial tip point load':
            self.model = {'u': self.part['beam']['u'][:, 0]}
        elif self.name == 'vertical hanging beam':
            self.model = {'U': self.part['beam']['U'][:, 2]}
        else:
            v = geom.qrotate(self.part['beam']['D'],
                             theta=-self.theta, dx='y')[:, i]  # deflection
            m = self.EI*self.part['beam']['d2u'][:, i]  # moment
            s = self.EI*self.part['beam']['d3u'][:, i]  # shear
            self.model = {'v': v, 'm': m, 's': s}

    def solve(self):
        finiteframe.solve(self)
        self.x = self.part['beam']['Lshp']
        self.extract_model()

    def check(self):
        for attribute in self.analytic:
            atol = 0.05 * np.max(abs(self.analytic[attribute]))
            assert np.allclose(self.analytic[attribute],
                               self.model[attribute], atol=atol)

    def plot(self):
        if self.name in ['axial tip point load', 'vertical hanging beam']:
            plt.figure()
            plt.title(self.name)
            plt.xlabel('beam length')
            plt.ylabel('extension')
            if self.name == 'axial tip point load':
                plt.plot(self.part['beam']['Lshp'], self.model['u'], 'C0')
                plt.plot(self.x, self.analytic['u'], '--C1')
            elif self.name == 'vertical hanging beam':
                plt.plot(self.part['beam']['Lshp'], self.model['U'])
                plt.plot(self.x, self.analytic['U'], '--')
        else:
            fig, ax = plt.subplots(3, 1, sharex=True, squeeze=True)
            ax[0].set_title(self.name)
            ax[0].plot(self.part['beam']['Lshp'], self.model['v'])
            ax[0].plot(self.x, self.analytic['v'], '--')
            ax[0].set_ylabel(r'deflection')
            ax[1].plot(self.part['beam']['Lshp'], self.model['m'])
            if 'm' in self.analytic:
                ax[1].plot(self.x, self.analytic['m'], '--')
            ax[1].set_ylabel(r'moment')
            ax[2].plot(self.part['beam']['Lshp'], self.model['s'])
            if 's' in self.analytic:
                ax[2].plot(self.x, self.analytic['s'], '--')
            ax[2].set_ylabel(r'shear')
            ax[2].set_xlabel('beam length')
        plt.despine()

def test_simple(plot=False):
    tb = testbeam('simple beam')
    # define model
    tb.add_bc('ny', 0, part='beam', ends=0)
    tb.add_bc('ny', -1, part='beam', ends=1)
    tb.add_weight()
    tb.solve()
    # analytic solution
    tb.analytic['v'] = tb.w*tb.x / (24*tb.EI) * (tb.L**3 - 2*tb.L*tb.x**2 +
                                                   tb.x**3)
    tb.analytic['m'] = -tb.w*tb.x / 2 * (tb.L-tb.x)
    tb.analytic['s'] = -tb.w * (tb.L/2 - tb.x)
    # assert
    tb.check()
    if plot:
        tb.plot()

def test_simple_xy(plot=False):
    tb = testbeam('simple beam')
    # define model
    tb.add_bc('ny', 0, part='beam', ends=0)
    tb.add_bc('ny', -1, part='beam', ends=1)
    finiteframe.add_weight(tb, g=[0, -1, 0])
    tb.solve()
    tb.extract_model(i=0)
    finiteframe.plot(tb, projection='xy')
    # analytic solution
    tb.analytic['v'] = tb.w*tb.x / (24*tb.EI) * (tb.L**3 - 2*tb.L*tb.x**2 +
                                                   tb.x**3)
    tb.analytic['m'] = -tb.w*tb.x / 2 * (tb.L-tb.x)
    tb.analytic['s'] = -tb.w * (tb.L/2 - tb.x)
    # assert
    #tb.check()
    if plot:
        tb.plot()

def test_cantilever(plot=False):
    tb = testbeam('cantilever beam')
    # define model
    tb.add_bc('fix', 0, part='beam', ends=0)
    tb.add_weight()
    tb.solve()
    # analytic solution
    tb.analytic['v'] = tb.w * tb.x**2 / (24 * tb.EI) \
        * (6*tb.L**2 - 4*tb.L*tb.x + tb.x**2)
    tb.analytic['m'] = tb.w * (tb.L - tb.x)**2 / 2
    tb.analytic['s'] = -tb.w * (tb.L - tb.x)
    # assert
    tb.check()
    if plot:
        tb.plot()

def test_pin_fix(plot=False):
    tb = testbeam('pin fix')
    # define model
    tb.add_bc('pin', 0, part='beam', ends=0)
    tb.add_bc('fix', -1, part='beam', ends=1)
    tb.add_weight()
    tb.solve()
    # analytic solution
    tb.analytic['v'] = tb.w * tb.x / (48 * tb.EI) * \
        (tb.L**3 - 3*tb.L*tb.x**2 + 2*tb.x**3)
    tb.analytic['m'] = -3*tb.w*tb.L*tb.x/8 + tb.w*tb.x**2/2
    tb.analytic['s'] = -tb.w*(3*tb.L/8 - tb.x)
    # assert
    tb.check()
    if plot:
        tb.plot()

def test_cantilever_point_load(plot=False):
    tb = testbeam('cantilever - point load')
    # define model
    P = -1e3
    tb.add_bc('fix', -1, part='beam', ends=1)
    tb.add_bc('v', 0, part='beam', ends=0)  # fix v displacement
    tb.add_nodal_load(0, 'fz', P)
    tb.solve()
    # analytic solution
    tb.analytic['v'] = P/(6*tb.EI) * (2*tb.L**3 - 3*tb.L**2*tb.x + tb.x**3)
    tb.analytic['m'] = P*tb.x
    tb.analytic['s'] = P*np.ones(len(tb.x))
    # assert
    tb.check()
    if plot:
        tb.plot()

def test_end_moment(plot=False):
    tb = testbeam('end moment')
    # define model
    Mo = 1e6
    tb.add_bc('fix', 0, part='beam', ends=0)
    tb.add_nodal_load(tb.nnd-1, 'my', Mo)
    tb.solve()
    # analytic solution
    tb.analytic['v'] = -Mo*tb.x**2/(2*tb.EI)
    tb.analytic['m'] = -Mo * np.ones(len(tb.x))
    # assert
    tb.check()
    if plot:
        tb.plot()

def test_cantilever_tapered_distributed_load(plot=False):
    tb = testbeam('cantilever tapered distributed load')
    # define model
    wo = -10  # N/m
    tb.add_bc('fix', 0, part='beam', ends=0)
    L = np.append(0, np.cumsum(tb.el['dl']))
    for i, el in enumerate(tb.part['beam']['el']):
        Lel = (L[i] + L[i+1]) / (2 * L[-1])
        W = wo*(Lel-1) * tb.g
        tb.add_load(el=el, W=W)  # self weight
    tb.solve()
    # analytic solution
    tb.analytic['v'] = wo*tb.x**2 / (120*L[-1]*tb.EI) * \
        (10*L[-1]**3 - 10*L[-1]**2 * tb.x + 5*L[-1] * tb.x**2 - tb.x**3)
    # assert
    tb.check()
    if plot:
        tb.plot()

def test_pinned_tapered_distributed_load(plot=False):
    tb = testbeam('pinned tapered distributed load')
    # define model
    wo = -10  # N/m
    tb.add_bc('fix', 0, part='beam', ends=0)
    L = np.append(0, np.cumsum(tb.el['dl']))
    for i, el in enumerate(tb.part['beam']['el']):
        Lel = (L[i] + L[i+1]) / (2 * L[-1])
        W = wo*(Lel-1) * tb.g
        tb.add_load(el=el, W=W)  # self weight
    tb.solve()
    # analytic solution
    tb.analytic['v'] = wo*tb.x**2 / (120*L[-1]*tb.EI) * \
        (10*L[-1]**3 - 10*L[-1]**2*tb.x + 5*L[-1]*tb.x**2 - tb.x**3)
    # assert
    tb.check()
    if plot:
        tb.plot()

def test_axial_tip_point_load(plot=False):
    tb = testbeam('axial tip point load')
    # define model
    Fx = 1e6
    tb.add_bc('fix', 0, part='beam', ends=0)
    tb.add_nodal_load(tb.nnd-1, 'fx', Fx)
    tb.solve()
    # analytic solution
    tb.analytic['u'] = Fx * tb.x / tb.EA
    # assert
    tb.check()
    if plot:
        tb.plot()


def test_vertical_hanging_beam(plot=False):
    tb = testbeam('vertical hanging beam', theta=0)
    # define model
    #g = geom.qrotate([0, 0, -1], theta=theta, dx='y')[0]
    tb.clfe()  # clear all (mesh, BCs, constraints and loads)
    tb.bar(theta=np.pi/2)  # rotate beam
    tb.add_bc('fix', 0, part='beam', ends=0)
    tb.add_weight()
    tb.solve()
    # analytic solution
    tb.analytic['U'] = tb.w / tb.EA * (tb.L * tb.x - tb.x**2 / 2)
    # assert
    tb.check()
    if plot:
        tb.plot()
    return tb


if __name__ == '__main__':
    #pytest.main([__file__])

    test_simple_xy(plot=True)

    #tb.plot()
    #tb.plot_stress()
    #tb.plot_moment()
    #tb.plot_matrix(tb.stiffness(0))
    #tb.plot_matrix(tb.Ko)
    #tb.plot_matrix(tb.K)

