import pylab as pl
import numpy as np
from matplotlib import animation, gridspec
from shelf import PKL
import seaborn as sns
rc = {'figure.figsize': [6, 6 * 7.5 / 24], 'savefig.dpi': 150,  # *12/16
      'savefig.jpeg_quality': 100, 'savefig.pad_inches': 0.1,
      'lines.linewidth': 1.75}
sns.set(context='poster', style='white', font='sans-serif', palette='Set2',
        font_scale=0.5, rc=rc)
color = sns.color_palette('Set2')
from amigo.addtext import linelabel

from itertools import cycle
Color = cycle(sns.color_palette('Set2'))
from radial_build import RB
#from eqConfig import Config
#from nova.config import Setup


#conf = Config('SXex')
#setup = Setup('SXex')

pkl = PKL('moveSX')  # 'moveSX_dev2'
sf, inv = pkl.fetch(['sf', 'inv'])
gs = gridspec.GridSpec(1, 2)
ax = 2 * [None]
ax[0] = pl.subplot(gs[0, 0])
ax[1] = pl.subplot(gs[0, 1])  #
#pl.setp(ax[0].get_xticklabels(), visible=False)

Swing = np.linspace(np.max(inv.Swing), np.min(inv.Swing), 20)  # -10
nSwing = len(Swing)
nPF = len(inv.PF_coils)
F = np.zeros((nPF, nSwing, 2))
I = np.zeros((nPF, nSwing))
Fcs = np.zeros((nSwing, 2))

for j, swing in enumerate(Swing):
    inv.swing_fix(swing)
    inv.solve_slsqp()
    print(inv.Fsep * 1e-6)
    for i, name in enumerate(inv.PF_coils):
        F[i, j, 0] = inv.coil['active'][name]['Fr_sum']
        F[i, j, 1] = inv.coil['active'][name]['Fz_sum']
        I[i, j] = inv.coil['active'][name]['I_sum']
    Fcs[j, 0] = inv.Fsep
    Fcs[j, 1] = inv.FzCS


pl.sca(ax[0])
text = linelabel(Ndiv=12, value='', postfix='')
for i, name in enumerate(inv.PF_coils):
    pl.plot(-2 * np.pi * (Swing - Swing[0]), abs(F[i, :, 1]) * 1e-6)
    text.add(name)
    # pl.plot(-2*np.pi*(swing*np.ones(2)-Swing[0]),[0,450],'k--',alpha=0.25)
Fcs[:, 0][Fcs[:, 0] < 0] = 0  # no limit on compression load
pl.plot(-2 * np.pi * (Swing - Swing[0]), Fcs[:, 0] * 1e-6)
text.add('Fsep')
pl.plot(-2 * np.pi * (Swing - Swing[0]), abs(Fcs[:, 1]) * 1e-6)
text.add('FzCS')
pl.ylabel(r'$|Fz|$ MN')
pl.xlabel(r'Swing Wb')
pl.ylim([0, 450])
pl.tight_layout()
#pl.xlabel(r'Swing Wb')
sns.despine()
text.plot()
pl.tight_layout()

pl.sca(ax[1])
text = linelabel(Ndiv=12, value='', postfix='')
for i, name in enumerate(inv.PF_coils):
    pl.plot(-2 * np.pi * (Swing - Swing[0]), abs(I[i, :]) * 1e-6)
    text.add(name)
    # pl.plot(-2*np.pi*(swing*np.ones(2)-Swing[0]),[0,22],'k--',alpha=0.25)
pl.ylabel(r'$|I|$ MA')
pl.xlabel(r'Swing Wb')
pl.tight_layout()
sns.despine()
text.plot()

# pl.savefig('../Figs/SX_force_sweep.png')
