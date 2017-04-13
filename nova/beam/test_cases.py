import numpy as np
import pylab as pl
from finite_element import FE
import seaborn as sns
rc = {'figure.figsize': [3, 3 * 12 / 16], 'savefig.dpi': 240,
      'savefig.jpeg_quality': 100, 'savefig.pad_inches': 0.1,
      'lines.linewidth': 2}
sns.set(context='talk', style='white', font='sans-serif', palette='Set2',
        font_scale=7 / 8, rc=rc)
from amigo.addtext import linelabel


L, F, alpha = 1, 20, 0 * np.pi / 180
Nel = 20
X = np.zeros((Nel + 1, 3))
X[:, 0] = np.linspace(0, L, Nel + 1)
R = np.array([[np.cos(alpha), -np.sin(alpha), 0],
              [np.sin(alpha), np.cos(alpha), 0],
              [0,            0,             1]])
X = np.dot(X, R)
fe = FE(frame='3D')

fe.add_mat(0, E=1, I=1, A=1, G=1, J=1, rho=1)


fe.add_nodes(X)
fe.add_elements(part_name='beam')


fe.add_nodes([1, -0.1, 0])
fe.add_elements(n=[fe.part['beam']['el'][-12], fe.nndo], part_name='strut')


fe.freeze()


fe.addBC('fix', [0, -1], part='beam')
fe.addBC('fix', [-1], part='strut')


fe.add_load(part='beam', L=0.4, F=[-0.1 * F, -F, 0])  # F,f==global,local


# fe.add_load(el=12,W=[0,F,0])  # F,f==global,local

fe.add_weight()  # add weight to all elements

fe.solve()
fe.plot_nodes()
# fe.plot_F(scale=1e-2)

fe.interpolate()


#text = linelabel(value='',postfix='',Ndiv=8)
for part in fe.part:
    pl.plot(fe.part[part]['U'][:, 0], fe.part[part]['U'][:, 1])
    # text.add('FEA')

'''    
x = fe.part[part]['U'][:,0]
#v = F*x**2/6*(3*L-x)
v = -9.81*fe.mat['rho'][0]*fe.mat['A'][0]*x**2/24*(6*L**2-4*x*L+x**2)
pl.plot(x,v,'--')
text.add('theory')
#pl.axis('equal')
'''
pl.axis('off')
# text.plot()

'''
print(fe.part[part]['U'][:,1].min(),
      L*fe.part['beam']['L'][-1]**3/(3*fe.mat['E'][0]*fe.mat['Iz'][0]))


pl.figure()
for part in fe.part:
    pl.plot(fe.part[part]['l'],fe.part[part]['d2u'][:,1])
'''
print(fe.part[part]['U'][:, 1].min(),
      9.81 * fe.mat['rho'][0] * fe.mat['A'][0] *
      fe.part['beam']['L'][-1]**4 / (8 * fe.mat['E'][0] * fe.mat['Iz'][0]))
