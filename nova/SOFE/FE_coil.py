import numpy as np
import pylab as pl
from nova.finite_element import FE
from nova.config import select

from nova.radial_build import RB
from nova.elliptic import EQ
from nova.coils import PF
from nova.inverse import INV
from nova.coils import TF
import nova.cross_coil as cc
from amigo import geom
from nova.loops import Profile
from nova.force import force_feild
from nova.structure import architect
import seaborn as sns
rc = {'figure.figsize': [8 * 12 / 16, 8], 'savefig.dpi': 120,
      'savefig.jpeg_quality': 100, 'savefig.pad_inches': 0.1,
      'lines.linewidth': 2}
sns.set(context='talk', style='white', font='sans-serif', palette='Set2',
        font_scale=7 / 8, rc=rc)

'''
Un structural solver incroyable
Le passé est l'avenir!
'''

nTF, ripple, ny, nr = 18, True, 6, 6
base = {'TF': 'demo_nTF', 'eq': 'DEMO_SN_EOF'}
config, setup = select(base, nTF=nTF, update=False)
atec = architect(config, setup, nTF=nTF)

fe = FE(frame='3D')  # initalise FE solver

'''
#matID['TFin'] =
matID['TFout'] =
matID['gs'] =
matID['OIS'] =
'''

atec.add_mat('TFin', ['wp', 'steel_forged'],
             [atec.winding_pack(), atec.case(0)])
atec.add_mat('TFout', ['wp', 'steel_cast'],
             [atec.winding_pack(), atec.case(1)])
atec.add_mat('gs', ['steel_cast'], [atec.gravity_support()])
atec.add_mat('OIS', ['wp'], [atec.intercoil_support()])






#rb = RB(setup,sf)
#pf = PF(sf.eqdsk)
'''
eq = EQ(sf, pf, dCoil=2, boundary=tf.get_loop(expand=0.5), n=1e3, sigma=0)
eq.get_plasma_coil()
ff = force_feild(pf.index, pf.coil, pf.sub_coil, pf.plasma_coil)

pf.plot(subcoil=True, label=False, plasma=True, current=False, alpha=0.5)
sf.contour()

to = time()
tf.split_loop()


nodes = {}
for part in ['loop', 'nose']:  # ,'nose'
    x, z = tf.x[part]['r'], tf.x[part]['z']
    if part == 'nose':
        x = np.min(x) * np.ones(len(x))
    X = np.zeros((len(x), 3))
    X[:, 0], X[:, 2] = x, z
    fe.add_nodes(X)
    nodes[part] = np.arange(fe.nndo, fe.nnd)
n = np.append(np.append(nodes['nose'][-1], nodes['loop']), nodes['nose'][0])
fe.add_elements(n=n, part_name='loop', nmat=matID['TFout'])
fe.add_elements(n=nodes['nose'], part_name='nose', nmat=matID['TFin'])
fe.add_bc('nw', 'all', part='nose')


nd_GS = fe.el['n'][fe.part['loop']['el'][10]][0]  # gravity support connect
fe.add_nodes([13, -2, -12])
fe.add_nodes([13, 2, -12])
fe.add_nodes(fe.X[nd_GS])
fe.add_elements(n=[fe.nndo - 2, fe.nndo, fe.nndo - 1],
                part_name='support', nmat=matID['gs'])
fe.add_bc('nry', [0], part='support', ends=0)
fe.add_bc('nry', [-1], part='support', ends=1)
fe.add_cp([fe.nndo, nd_GS], dof='nrx')  # 'nrx'


nd_OISo = fe.el['n'][fe.part['loop']['el'][15]][0]  # OIS
nd_OIS1 = fe.el['n'][fe.part['loop']['el'][27]][0]

fe.add_nodes(
    np.dot(fe.X[nd_OISo], geom.rotate(-np.pi / config['nTF'], axis='z')))
fe.add_nodes(np.dot(fe.X[nd_OISo], geom.rotate(
    np.pi / config['nTF'], axis='z')))
fe.add_elements(n=[fe.nndo - 1, nd_OISo, fe.nndo], part_name='OISo',
                el_dy=np.cross(fe.el['dx'][nd_OISo], [0, 1, 0]),
                nmat=matID['OIS'])
fe.add_cp([fe.nndo - 1, fe.nndo], dof='fix', rotate=True, axis='z')

fe.add_nodes(np.dot(fe.X[nd_OIS1],
                    geom.rotate(-np.pi/config['nTF'], axis='z')))
fe.add_nodes(np.dot(fe.X[nd_OIS1],
                    geom.rotate(np.pi/config['nTF'], axis='z')))
fe.add_elements(n=[fe.nndo-1, nd_OIS1, fe.nndo], part_name='OISo',
                el_dy=np.cross(fe.el['dx'][nd_OIS1], [0, 1, 0]),
                nmat=matID['OIS'])
fe.add_cp([fe.nndo - 1, fe.nndo], dof='fix', rotate=True, axis='z')

print(np.cross(fe.el['dx'][nd_OIS1], [0, 1, 0]))

fe.add_weight()  # add weight to all elements
# burst and topple
fe.add_tf_load(sf, ff, tf, sf.Bpoint, method='function')


fe.solve()
fe.deform(scale=50)

print('time {:1.3f}'.format(time() - to))

fe.plot_nodes()
fe.plot_F(scale=1)

fe.plot_displacment()
pl.axis('off')
pl.tight_layout(rect=[-0.3, 0.1, 0.9, 0.9])


fe.plot_3D(pattern=config['nTF'])
# fe.plot_curvature()

fe.plot_twin()

fe.plot_curvature()
'''
