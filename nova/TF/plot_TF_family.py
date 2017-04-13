import numpy as np
from nova.loops import Profile
from nova.shape import Shape
from amigo.addtext import linelabel
import pylab as pl
from nova.TF.DEMOxlsx import DEMO
from nova.streamfunction import SF
from nova.config import Setup

import seaborn as sns
rc = {'figure.figsize': [10 * 1.1, 10], 'savefig.dpi': 100,  # *12/16
      'savefig.jpeg_quality': 100, 'savefig.pad_inches': 0,
      'lines.linewidth': 0.75}
sns.set(context='talk', style='white', font='sans-serif', palette='Set2',
        font_scale=1.5, rc=rc)

text = linelabel(Ndiv=30, value='', loc='end')

demo = DEMO()
demo.fill_part('Blanket')
demo.fill_part('Vessel')

config = {'TF': 'dtt', 'eq': 'SN'}
setup = Setup(config['eq'])
sf = SF(setup.filename)
# sf.contour(plot_vac=False)
pl.plot(np.append(sf.rbdry, sf.rbdry[0]),
        np.append(sf.zbdry, sf.zbdry[0]),
        color=0.75 * np.ones(3), lw=2)

NTF = np.arange(13, 19)[::-1]  # 13,
pl.plot([3, 23], [-10, 9], 'ko', alpha=1)


'''
config['TF'] = 'dtt'
config['TF'] = '{}{}{:d}'.format(config['eq'],config['TF'],18)
profile = Profile(config['TF'],family='S',part='TF',load=True)
shp = Shape(profile,nTF=18,obj='L',eqconf=config['eq'],ny=1)
shp.update()
shp.tf.fill()
pl.axis('off')
pl.axis('equal')
pl.tight_layout()
pl.savefig('../../Figs/TFfam_0.png')


j=6
for i,nTF in enumerate(NTF[:j]):
    config['TF'] = 'dtt'
    config['TF'] = '{}{}{:d}'.format(config['eq'],config['TF'],nTF)
    profile = Profile(config['TF'],family='S',part='TF',load=True)
    shp = Shape(profile,nTF=nTF,obj='L',eqconf=config['eq'],ny=3)
    shp.update()
    E = shp.cage.energy()*1e-9
    x = shp.loop.draw()
    n = int(len(x['r'])/2)-i
    for var in ['r','z']:
        x[var] = np.append(x[var][n-1:],x[var][:n])
    pl.plot(x['r'],x['z'],lw=3)
    text.add('{:d}TF {:1.0f}GJ'.format(nTF,E))
shp.cage.plot_loops()
text.plot(Ralign=True)      
pl.axis('off')
pl.axis('equal')
pl.tight_layout()
pl.savefig('../../Figs/TFfam_{:d}.png'.format(j))
'''
