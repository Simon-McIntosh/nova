import numpy as np
from nova.finite_element import FE
from nova.properties import second_moment
from amigo import geom
from amigo.pyplot import plt, plstyle

#from matplotlib import pyplot as plt

pls = plstyle('talk')

pls.avalible()

#pls.reset()
#plt.style.use('default')



#

# import matplotlib as mpl
# mpl.rcParams['figure.figsize'] = [12, 4.8]


class TB:  # test beam

    def __init__(self, N, L=3):
        self.fe = FE(frame='3D')
        self.section()
        self.bar(N)

    def section(self):
        sm = second_moment()
        sm.add_shape('circ', r=0.2, ro=0.1)
        C, I, A = sm.report()
        section = {'C': C, 'I': I, 'A': A, 'J': I['xx'], 'pnt': sm.get_pnt()}
        self.fe.add_mat('tube', ['steel_cast'], [section])

        # extract sectional properties
        self.w = -9.81 * self.fe.mat[0]['mat_o']['rho'] *\
            self.fe.mat[0]['mat_o']['A']  # weight / l
        E = self.fe.mat[0]['mat_o']['E']
        Iy = self.fe.mat[0]['mat_o']['I'][1]
        self.EI = E * Iy

    def bar(self, N, L=3):
        self.L = L
        self.x = np.linspace(0, self.L, 50)
        X = np.zeros((N, 3))
        X[:, 0] = np.linspace(0, self.L, N)
        self.fe.add_nodes(X)
        self.fe.add_elements(part_name='beam', nmat='tube')

    def plot(self, v, m, title):
        fig, ax = plt.subplots(2, 1, sharex=True, squeeze=True)
        ax[0].set_title(title)
        ax[0].plot(self.fe.part['beam']['Lnd'], self.fe.D['z'])
        ax[0].plot(self.x, v, '--')
        ax[0].set_ylabel(r'$\delta$')
        ax[1].plot(self.fe.part['beam']['Lshp'],
                   self.EI*self.fe.part['beam']['d2u'][:, 2])
        ax[1].plot(self.x, m, '--')
        ax[1].set_ylabel(r'$M$')
        plt.despine()

    def test(self, case):
        self.fe.initalize_BC()  # clear boundary conditions

        if case == 0:  # simple beam
            name = 'simple beam'
            v = self.w * self.x / (24 * self.EI) *\
                (self.L**3 - 2 * self.L * self.x**2 + self.x**3)
            m = -self.w * self.x / 2 * (self.L - self.x)  # check sign
            self.fe.add_bc('ny', 0, part='beam', ends=0)
            self.fe.add_bc('ny', -1, part='beam', ends=1)

        elif case == 1:  # cantilever beam
            name = 'cantilever beam'
            v = self.w * self.x**2 / (24 * self.EI) * (6*self.L**2 -
                                      4*self.L*self.x + self.x**2)
            m = self.w * self.x**2 / 2
            self.fe.add_bc('fix', 0, part='beam', ends=0)
            self.fe.add_weight()  # add weight to all elements

        elif case == 2:  # pin, fix
            v = self.w * self.x / (48 * self.EI) * \
                (self.L**3 - 3*self.L*self.x**2 + 2*self.x**3)
            m = -3*self.w*self.L*self.x/8 + self.w*self.x**2/2  # check sign
            self.fe.initalize_BC()  # clear boundary conditions
            self.fe.add_bc('pin', 0, part='beam', ends=0)
            self.fe.add_bc('fix', -1, part='beam', ends=1)

        self.fe.add_weight()
        self.fe.solve()  # solve
        self.plot(v, m, name)


if __name__ == '__main__':

    tb = TB(30)
    tb.test(1)


'''

fe.deform(1e6)
fe.plot_F(scale=1e-4)
fe.plot_displacment()
fe.plot_nodes()



x = fe.part[part]['U'][:,0]
# v = F*x**2/6*(3*L-x)
v = -9.81*fe.mat['rho'][0]*fe.mat['A'][0]*x**2/24*(6*L**2-4*x*L+x**2)
pl.plot(x,v,'--')
text.add('theory')
#pl.axis('equal')

pl.axis('off')
# text.plot()


print(fe.part[part]['U'][:,1].min(),
      L*fe.part['beam']['L'][-1]**3/(3*fe.mat['E'][0]*fe.mat['Iz'][0]))


pl.figure()
for part in fe.part:
    pl.plot(fe.part[part]['l'],fe.part[part]['d2u'][:,1])

print(fe.part[part]['U'][:, 1].min(),
      9.81 * fe.mat['rho'][0] * fe.mat['A'][0] *
      fe.part['beam']['L'][-1]**4 / (8 * fe.mat['E'][0] * fe.mat['Iz'][0]))
'''