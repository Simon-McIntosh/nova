import numpy as np
from amigo.pyplot import plt
from nova.finite_element import FE
from nova.properties import second_moment
from amigo import geom


class TB:  # test beam

    def __init__(self, N):
        self.N = N  # node number
        self.fe = FE(frame='3D')
        self.section()

    def section(self):
        sm = second_moment()
        sm.add_shape('circ', r=0.02, ro=0.01)
        C, I, A = sm.report()
        section = {'C': C, 'I': I, 'A': A, 'J': I['xx'], 'pnt': sm.get_pnt()}
        self.fe.add_mat('tube', ['steel_cast'], [section])

        # extract sectional properties
        self.w = -9.81 * self.fe.mat[0]['mat_o']['rho'] *\
            self.fe.mat[0]['mat_o']['A']  # weight / l
        E = self.fe.mat[0]['mat_o']['E']
        A = self.fe.mat[0]['mat_o']['A']
        Iy = self.fe.mat[0]['mat_o']['I'][1]
        self.EI = E * Iy
        self.EA = E * A

    def bar(self, theta=0, L=3):
        self.L = L
        self.x = np.linspace(0, self.L, 50)
        X = np.zeros((self.N, 3))
        X[:, 0] = np.linspace(0, self.L, self.N)
        X = geom.qrotate(X, theta, dx='y')
        self.fe.add_nodes(X)
        self.fe.add_elements(part_name='beam', nmat='tube')
        self.fe.update_rotation()  # check / update rotation matrix

    def plot(self, case, v, m, s, title, theta=0):
        if case == 7 or case == 8:
            plt.figure()
            plt.title(title)
            plt.xlabel('beam length')
            plt.ylabel('extension')

            if case == 7:
                plt.plot(self.fe.part['beam']['Lshp'],
                         self.fe.part['beam']['u'][:, 0], 'C0')
                plt.plot(self.x, 1e6 * self.x / self.EA, '--C1')
            elif case == 8:
                u = self.w / self.EA * (self.L * self.x - self.x**2 / 2)
                plt.plot(self.fe.part['beam']['Lshp'],
                         self.fe.part['beam']['U'][:, 2])
                plt.plot(self.x, u, '--')
        else:
            fig, ax = plt.subplots(3, 1, sharex=True, squeeze=True)
            ax[0].set_title(title)
            D = geom.qrotate(self.fe.part['beam']['D'], -theta, dx='y')
            ax[0].plot(self.fe.part['beam']['Lshp'], D[:, 2])
            ax[0].plot(self.x, v, '--')
            ax[0].set_ylabel(r'deflection')
            ax[1].plot(self.fe.part['beam']['Lshp'],
                       self.EI*self.fe.part['beam']['d2u'][:, 2])
            ax[1].plot(self.x, m, '--')
            ax[1].set_ylabel(r'moment')
            ax[2].plot(self.fe.part['beam']['Lshp'],
                       self.EI*self.fe.part['beam']['d3u'][:, 2])
            ax[2].plot(self.x, s, '--')
            ax[2].set_ylabel(r'shear')
            ax[2].set_xlabel('beam length')
        plt.despine()

    def test(self, case, theta=0):
        self.fe.clfe()  # clear all (mesh, BCs, constraints and loads)
        g = geom.qrotate([0, 0, -1], theta, dx='y')[0]

        self.bar(theta=theta)  # add nodes / elements
        v = np.zeros(np.shape(self.x))
        m = np.zeros(np.shape(self.x))
        s = np.zeros(np.shape(self.x))

        if case == 0:  # simple beam
            name = 'simple beam'
            v = self.w * self.x / (24 * self.EI) *\
                (self.L**3 - 2 * self.L * self.x**2 + self.x**3)
            m = -self.w * self.x / 2 * (self.L - self.x)
            s = -self.w * (self.L/2 - self.x)
            self.fe.add_bc('ny', 0, part='beam', ends=0)
            self.fe.add_bc('ny', -1, part='beam', ends=1)
            self.fe.add_weight(g=g)

        elif case == 1:  # cantilever beam
            name = 'cantilever beam'
            v = self.w * self.x**2 / (24 * self.EI) \
                * (6*self.L**2 - 4*self.L*self.x + self.x**2)
            m = self.w * (self.L - self.x)**2 / 2
            s = -self.w * (self.L - self.x)
            self.fe.add_bc('fix', 0, part='beam', ends=0)
            self.fe.add_weight(g=g)

        elif case == 2:  # pin, fix
            name = 'pin fix'
            v = self.w * self.x / (48 * self.EI) * \
                (self.L**3 - 3*self.L*self.x**2 + 2*self.x**3)
            m = -3*self.w*self.L*self.x/8 + self.w*self.x**2/2
            s = -self.w*(3*self.L/8 - self.x)
            self.fe.add_bc('pin', 0, part='beam', ends=0)
            self.fe.add_bc('fix', -1, part='beam', ends=1)
            self.fe.add_weight(g=g)

        elif case == 3:  # cantilever point load
            name = 'cantilever - point load'
            P = -1e3
            v = P/(6*self.EI) * (2*self.L**3 - 3*self.L**2*self.x + self.x**3)
            m = P*self.x
            s = P*np.ones(len(self.x))
            self.fe.add_bc('fix', -1, part='beam', ends=1)
            self.fe.add_nodal_load(0, 'fz', P)

        elif case == 4:
            name = 'end moment'
            Mo = 1e6
            self.fe.add_bc('fix', 0, part='beam', ends=0)
            self.fe.add_nodal_load(tb.fe.nnd-1, 'my', Mo)
            v = -Mo*self.x**2/(2*self.EI)
            m = -Mo * np.ones(len(self.x))

        elif case == 5:
            name = 'cantilever tapered distributed load'
            wo = -10  # N/m
            self.fe.add_bc('fix', 0, part='beam', ends=0)
            L = np.append(0, np.cumsum(tb.fe.el['dl']))
            for i, el in enumerate(self.fe.part['beam']['el']):
                Lel = (L[i] + L[i+1]) / (2 * L[-1])
                W = wo*(Lel-1) * g
                self.fe.add_load(el=el, W=W)  # self weight
            v = wo*self.x**2/(120*L[-1]*self.EI)*(10 * L[-1]**3 -
                                                  10 * L[-1]**2 * self.x +
                                                  5 * L[-1] * self.x**2 -
                                                  self.x**3)

        elif case == 6:
            name = 'pinned tapered distributed load'
            wo = -10  # N/m
            self.fe.add_bc('fix', 0, part='beam', ends=0)
            L = np.append(0, np.cumsum(tb.fe.el['dl']))
            for i, el in enumerate(self.fe.part['beam']['el']):
                Lel = (L[i] + L[i+1]) / (2 * L[-1])
                W = wo*(Lel-1) * g
                self.fe.add_load(el=el, W=W)  # self weight
            v = wo*self.x**2/(120*L[-1]*self.EI)*(10 * L[-1]**3 -
                                                  10 * L[-1]**2 * self.x +
                                                  5 * L[-1] * self.x**2 -
                                                  self.x**3)
        elif case == 7:
            name = 'axial tip point load'
            self.fe.add_bc('fix', 0, part='beam', ends=0)
            self.fe.add_nodal_load(self.fe.nnd-1, 'fx', 1e6)

        elif case == 8:
            name = 'hanging beam'
            self.fe.clfe()  # clear all (mesh, BCs, constraints and loads)
            theta = 0
            g = geom.qrotate([0, 0, -1], theta, dx='y')[0]
            self.bar(theta=np.pi/2)  # rotate beam
            self.fe.add_bc('fix', 0, part='beam', ends=0)
            self.fe.add_weight(g=g)

        self.fe.solve()  # solve
        self.plot(case, v, m, s, name, theta=theta)


if __name__ == '__main__':

    tb = TB(13)
    tb.test(0, theta=1*np.pi/180)
    tb.fe.plot()

    tb.fe.plot_stress()
    
    tb.fe.plot_moment()
    #tb.fe.plot_matrix(tb.fe.stiffness(0))
    #tb.fe.plot_matrix(tb.fe.Ko)
    #tb.fe.plot_matrix(tb.fe.K)

