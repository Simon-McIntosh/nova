import numpy as np
import pylab as pl
from nova.finite_element import FE
from nova.properties import second_moment
from amigo import geom

fe = FE(frame='3D')
sm = second_moment()
sm.add_shape('circ', r=0.2, ro=0.1)
C, I, A = sm.report()
section = {'C': C, 'I': I, 'A': A, 'J': I['xx'], 'pnt': sm.get_pnt()}
fe.add_mat('bar', ['steel_cast'], [section])

# extract sectional properties
w = -9.81 * fe.mat[0]['mat_o']['rho'] * fe.mat[0]['mat_o']['A']  # weight / l
E = fe.mat[0]['mat_o']['E']
Iy = fe.mat[0]['mat_o']['I'][1]

L = 3
N = 21
x = np.linspace(0, L, N)
X = np.zeros((N, 3))
X[:, 0] = x
fe.add_nodes(X)
fe.add_elements(part_name='beam', nmat='bar')


def plot(fe, x, v, m, title):
    fig, ax = pl.subplots(2, 1, sharex=True, squeeze=True, figsize=(12, 8))
    ax[0].set_title(title)
    ax[0].plot(x, fe.D['z'])
    ax[0].plot(x, v, '--')
    ax[0].set_ylabel(r'$\delta$')
    ax[1].plot(fe.part['beam']['l'], E*Iy*fe.part['beam']['d2u'][:, 2])
    ax[1].plot(x, m, '--')
    ax[1].set_ylabel(r'$M$')


# simple beam
v = w * x / (24 * E * Iy) * (L**3 - 2 * L * x**2 + x**3)  # deflection
m = -w * x / 2 * (L - x)  # check sign
fe.add_bc('ny', 0, part='beam', ends=0)
fe.add_bc('ny', -1, part='beam', ends=1)
fe.add_weight()  # add weight to all elements
fe.solve()
plot(fe, x, v, m, 'simple beam')

# cantilever beam
v = w / (24 * E * Iy) * (x**4 - 4*L**3*x + 3*L**4)  # deflection
m = w * x**2 / 2
fe.initalize_BC()  # clear boundary conditions
fe.add_bc('fix', -1, part='beam', ends=1)
fe.solve()
plot(fe, x, v, m, 'cantilever beam')

# pin, fix
v = w * x / (48 * E * Iy) * (L**3 - 3*L*x**2 + 2*x**3)  # deflection
m = -3*w*L*x/8 + w*x**2/2  # check sign
fe.initalize_BC()  # clear boundary conditions
fe.add_bc('pin', 0, part='beam', ends=0)
fe.add_bc('fix', -1, part='beam', ends=1)
fe.solve()
plot(fe, x, v, m, 'pin fix')


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