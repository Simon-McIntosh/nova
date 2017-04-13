import pylab as pl
import numpy as np
from addtext import linelabel
import seaborn as sns
rc = {'figure.figsize': [3 * 3.14, 3 * 3.14 * 7 / 12], 'savefig.dpi': 100,
      'savefig.jpeg_quality': 100, 'savefig.pad_inches': 0.1,
      "lines.linewidth": 4}
sns.set(context='poster', style='white', font='sans-serif', palette='Set2',
        font_scale=1.4, rc=rc,)
color = sns.color_palette('Set2', 5)
fs = 24

pfrac = 0

# with plt.xkcd(scale=1):

sns.set_context(rc={"lines.linewidth": 6})

fig = pl.figure()
text = linelabel(Ndiv=10, value='', loc='end')
ax = fig.add_axes((0.1, 0.2, 0.6, 0.7))
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')
pl.xticks([])
pl.yticks([])
ax.set_ylim([0.6, 1])

R = np.linspace(1, 1.5, 100)
pcore = (1 + pfrac) / R
prad = 0.76 * np.ones(len(R))

if pfrac == 0:
    pl.annotate('ne low for compatability', xy=(1.183, prad[0] * 0.8),
                arrowprops={'arrowstyle': '->', 'lw': 4}, xytext=(1.02, prad[0] * 0.95),
                fontsize=fs)
    dlabel, clabel = '', ''
elif pfrac < 0:
    dlabel = ' moves inwards'
    clabel = ' {:+1.0f}%'.format(pfrac * 100)
else:
    dlabel = ' moves outwards'
    clabel = ' {:+1.0f}%'.format(pfrac * 100)


i = np.argmin(abs(pcore - prad))
xd, yd = R[i], pcore[i]
pl.annotate('detachment' + dlabel, xytext=(1.1, yd + 0.2), xy=(xd, yd),
            arrowprops={'arrowstyle': '->', 'lw': 4},
            fontsize=fs)

pl.plot(R, pcore, color=color[0])
text.add('from core' + clabel, loc='end')


pl.plot(R, prad, color=color[2])
text.add('radiation', loc='end')
if pfrac == 0:
    text.add('power ~ ne ni Q(T)', loc='end')

text.plot()
pl.xlabel('Raduis')
pl.ylabel('Power flux')


pl.savefig('../Figs/' + 'power_{:+1.0f}%'.format(pfrac * 100))
