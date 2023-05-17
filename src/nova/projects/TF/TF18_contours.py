import numpy as np
from nova.loops import Profile
from nova.shape import Shape
from amigo.addtext import linelabel
import pylab as pl
from nova.DEMOxlsx import DEMO
from nova.streamfunction import SF
from nova.config import Setup

import seaborn as sns
rc = {'figure.figsize': [10 * 1.1, 10], 'savefig.dpi': 100,  # *12/16
      'savefig.jpeg_quality': 100, 'savefig.pad_inches': 0,
      'lines.linewidth': 3}
sns.set(context='talk', style='white', font='sans-serif', palette='Set2',
        font_scale=1.5, rc=rc)

text = linelabel(Ndiv=30, value='', loc='end')

config = {'TF': 'dtt', 'eq': 'SN'}
setup = Setup(config['eq'])
sf = SF(setup.filename)
# sf.contour(plot_vac=False)
pl.plot(np.append(sf.rbdry, sf.rbdry[0]),
        np.append(sf.zbdry, sf.zbdry[0]),
        color=0.75 * np.ones(3), lw=2)

NTF = np.arange(13, 19)[::-1]  # 13,
pl.plot([3, 20], [-10, 13], 'ko', alpha=1)

nTF = 18
family = 'D'

config['TF'] = 'dtt'
config['TF'] = '{}{}{:d}'.format(config['eq'], config['TF'], nTF)
profile = Profile(config['TF'], family=family, part='TF', load=True)
shp = Shape(profile, nTF=nTF, obj='L', eqconf=config['eq'], ny=3)
# shp.update()
shp.tf.fill()

shp.cage.plot_loops(sticks=True)
# shp.cage.plot_contours(variable='ripple',n=3e3,loop=shp.loop.draw())

pl.tight_layout()
pl.savefig('../../Figs/TFripple.png')
# pl.savefig('../../Figs/TF{:d}{}_contour.png'.format(nTF,family))
