from nova.TF.DEMOxlsx import DEMO
from nova.coils import PF, TF
from nova.loops import Profile
from nova.coil_cage import coil_cage
from nova.streamfunction import SF
from nova.config import Setup
from scipy.optimize import minimize_scalar, minimize
import numpy as np
import json
from nova.config import trim_dir
import seaborn as sns
import pylab as pl
from DEMOxlsx import DEMO
from nova.radial_build import RB
from amigo import geom

rc = {'figure.figsize': [8 * 10 / 16, 8], 'savefig.dpi': 150,  # *12/16
      'savefig.jpeg_quality': 100, 'savefig.pad_inches': 0.1,
      'lines.linewidth': 0.75}
sns.set(context='paper', style='white', font='sans-serif', palette='Set2',
        font_scale=7 / 8, rc=rc)

nTF = 16
config = {'TF': 'SN', 'eq': 'DEMO_SNb'}


profile = Profile(config['TF'], family='S', load=True, part='TF',
                  nTF=nTF, obj='L', npoints=250)
setup = Setup(config['eq'])
sf = SF(setup.filename)
tf = TF(profile, sf=sf)
pf = PF(sf.eqdsk)
rb = RB(setup, sf)

cage = coil_cage(nTF=nTF, rc=tf.rc, plasma={
                 'config': config['eq']}, coil=tf.x['cl'])

demo = DEMO()
demo.fill_part('Vessel')
demo.fill_part('Blanket')
demo.fill_part('TF_Coil')
demo.plot_ports()
demo.plot_limiter()

sf.contour(Nlevel=51, plot_vac=False, lw=0.5)
pl.plot(sf.rbdry, sf.zbdry, color=0.75 * np.ones(3), lw=1)

r, z = demo.parts['Plasma']['out']['r'], demo.parts['Plasma']['out']['z']
rb.Rb, rb.Zb = geom.rzInterp(r, z)

rb.trim_sol()
tf.fill(alpha=0.8)
# cage.plot_contours()


pl.axis('equal')
pl.axis('off')

# pl.savefig('../../Figs/TF_ripple_{:d}.pdf'.format(nTF))

ro = np.max(demo.parts['TF_Coil']['in']['r'])
r1 = np.max(demo.parts['TF_Coil']['out']['r'])
offset = (r1 - ro) / 2
rcl, zcl = geom.offset(demo.parts['TF_Coil']['in']['r'],
                       demo.parts['TF_Coil']['in']['z'], offset)

print('shape', cage.energy() * 1e-9)


cage = coil_cage(nTF=nTF, rc=tf.rc, plasma={
                 'config': config['eq']}, coil={'r': rcl, 'z': zcl})
print('baseline', cage.energy() * 1e-9)
