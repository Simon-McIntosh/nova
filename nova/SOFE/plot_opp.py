import pylab as pl
from nova.shelf import PKL
from nova import loops
import matplotlib.animation as manimation

import seaborn as sns
sns.set(context='talk', style='white')

name = 'DEMO_SF'
name = 'DEMO_SX'
pkl = PKL(name, directory='../../Movies/')
sf, inv, rb, tf = pkl.fetch(['sf', 'inv', 'rb', 'tf'])

fig, ax1 = pl.subplots(figsize=(12, 8))


def animate(index):
    print(index, inv.log['plasma_iter'][index])
    pl.sca(ax1)
    ax1.cla()
    ax1.set_axis_off()
    ax1.set_xlim([2, 18])
    ax1.set_ylim((-15.0, 10.0))

    pl.plot([0, 20], [-16.0, 12.0], 'o', alpha=0)
    pl.axis('equal')

    Lo = inv.log['Lo'][index]
    Lnorm = loops.normalize_variables(Lo)

    inv.update_position(Lnorm, update_area=True)
    inv.eq.run(update=False)
    inv.eq.sf.contour()
    inv.pf.plot(subcoil=True, plasma=True)
    tf.fill()
    inv.plot_fix()
    rb.plot()

FFMpegWriter = manimation.writers['ffmpeg']
writer = FFMpegWriter(fps=2, bitrate=-1)

with writer.saving(fig, '../../Movies/{}_opp.mp4'.format(name), 100):
    for i in range(inv.log['position_iter'][-1]):
        animate(i)
        writer.grab_frame()
