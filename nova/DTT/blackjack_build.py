import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as manimation
from nova.streamfunction import SF
from nova.elliptic import EQ
from nova.inverse import INV
from itertools import cycle
import numpy as np
from nova.radial_build import RB
from nova.shelf import PKL
from eqConfig import Config
import pylab as pl
from time import time
import sys
import datetime


import seaborn as sns
ar = 1.25  # 14/10.0 # figure aspect
rc = {'figure.figsize': [8 / ar, 8], 'savefig.dpi': 150,
      'savefig.jpeg_quality': 100, 'savefig.pad_inches': 0,
      'lines.linewidth': 0.75}
sns.set(context='paper', style='white', font='sans-serif', palette='Set2',
        font_scale=1.5, rc=rc)

config = 'SF'
config = 'SNdtt18_6PF_5CS'
#config = 'SNdtt18_5PF_5CS'
#config = 'SNdtt18_4PF_5CS'
pkl = PKL(config, directory='../../Movies/')
sf, eq, inv = pkl.fetch(['sf', 'eq', 'inv'])

inv.update_position(inv.log['Lo'][-1]['norm'], update_area=True)
inv.swing_flux(inv.swing['flux'][0])

sequence = [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            [0, 1, 3, 4, 5, 6, 7, 8, 9, 10],
            [0, 1, 3, 4, 5, 6, 7, 9, 10],
            [1, 3, 4, 5, 6, 7, 9, 10],
            [1, 3, 5, 6, 7, 9, 10],
            [1, 5, 6, 7, 9, 10],
            [1, 5, 6, 7, 10],
            [1, 5, 6, 10],
            [5, 6, 10],
            [6, 10],
            [6],
            []
            ]

for index in np.arange(11, -1, -1):

    if index < 11:
        inv.remove_active(sequence[index])
    inv.solve_slsqp()

    ax = pl.gca()
    ax.cla()
    pl.plot(2, -13, 'o', alpha=0)
    pl.plot(19, 11, 'o', alpha=0)

    inv.plot_fix()
    eq.run(update=False, verbose=False)

    coils = {}
    for name in eq.pf.coil:
        i = int(name.replace('Coil', ''))
        if i not in sequence[index]:
            coils[name] = eq.pf.coil[name]

    sf.contour(lw=1.25, boundary=False)
    eq.pf.plot(coils=coils, label=False, plasma=False, current=True)
    pl.text(6, -12, r'$< \Delta \psi^2 >^{-0.5} = $'
            + '{:1.0f}mm'.format(1e3 * inv.rms), fontsize=16)
    pl.savefig('../../Figs/BJ_sequence_{:d}.png'.format(index))
