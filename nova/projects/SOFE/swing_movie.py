import pylab as pl
import numpy as np
from nova.shelf import PKL
import matplotlib.animation as manimation

FFMpegWriter = manimation.writers["ffmpeg"]
writer = FFMpegWriter(fps=5, bitrate=1000)

import seaborn as sns

rc = {
    "figure.figsize": [12, 5],
    "savefig.dpi": 175,  # *12/16
    "savefig.jpeg_quality": 100,
    "savefig.pad_inches": 0.1,
    "lines.linewidth": 1.75,
}
sns.set(
    context="poster",
    style="white",
    font="sans-serif",
    palette="Set2",
    font_scale=1,
    rc=rc,
)
color = sns.color_palette("Set2")
from amigo.addtext import linelabel

from itertools import cycle

Color = cycle(sns.color_palette("Set2"))

# tf = TF('DEMO_SN',coil_type='S',nTF=16,objective='L')


config = "DEMO_SF"
pkl = PKL(config, directory="../../Movies/")
sf, inv = pkl.fetch(["sf", "inv"])

pl.figure()


def animate(flux):  # ,data,ax
    cycle(sns.color_palette("Set2"))

    pl.cla()
    # ax1.set_axis_off()
    ax.set_ylim([3, 14])
    # pl.axis('equal')

    inv.solve_slsqp(flux)

    print(inv.I[-3])
    pl.plot(sf.xbdry, sf.zbdry)
    # tf.fill()

    # print(swing,sf.Xpsi,sf.Mpoint[1],inv.rms)
    # B = eq.Bfeild([inv.fix['r'][-1],inv.fix['z'][-1]])
    # arg = 180*np.arctan2(B[1],B[0])/np.pi
    # print('swing {:1.0f} arg {:1.2f} rms {:1.3f}'.format(swing,arg,inv.rms))

    inv.eq.run(update=False)
    # sf.sol(plot=True)
    # inv.eq.pf.plot(coils=inv.eq.pf.coil,label=False,plasma=False,current=False)

    # inv.plot_force()
    # eq.plasma(sep=False)
    # eq.plotb()
    inv.sf.contour(levels=np.linspace(-150, 150, 60), Xnorm=False)

    # rb.divertor_outline(False,plot=True,debug=False)

    """
    with open('./plot_data/'+rb.conf.config+'_FW.pkl', 'rb') as input:
                rb.targets = pickle.load(input)
                rb.Rb = pickle.load(input)
                rb.Zb = pickle.load(input)
    rb.Rb,rb.Zb = rb.midplane_loop(rb.Rb,rb.Zb)  # clockwise LFS
    pl.plot(rb.Rb,rb.Zb,'-',color='k',alpha=0.75,linewidth=1.25)

    rb.FWfill(dt=conf.tfw,loop=True,alpha=0.7,color=next(Color),s=2e-3)
    rb.fill(dt=conf.BB[::-1],alpha=0.7,ref_o=0.3,dref=0.2,
            referance='length',color=next(Color))
    rb.fill(dt=conf.tBBsupport,alpha=0.7,color=next(Color))
    rb.BBsheild_fill(dt=conf.sheild,ref_o=0.35*np.pi,dref=0.2*np.pi,offset=1/10*np.pi,
                     alpha=0.7,color=next(Color))
    rb.VVfill(dt=conf.VV,ref_o=0.25*np.pi,dref=0.25*np.pi,offset=0.5/10*np.pi,
              alpha=0.7,loop=True,color=next(Color))  # ref_o=0.385

    rb.TFopp(False,objF=conf.TFopp)  # L==length, V==volume
    rb.TFfill()
    """
    # pl.tight_layout()

    pl.sca(ax2)
    ax2.cla()
    text = linelabel(Ndiv=10, value="", postfix="")
    for i, name in enumerate(inv.PF_coils):
        pl.plot(-2 * np.pi * (Swing - Swing[0]), abs(F[i, :, 1]))
        text.add(name)
        pl.plot(
            -2 * np.pi * (swing * np.ones(2) - Swing[0]), [0, 450], "k--", alpha=0.25
        )
    pl.plot(-2 * np.pi * (Swing - Swing[0]), Fcs[:, 0])
    text.add("Fsep")
    pl.plot(-2 * np.pi * (Swing - Swing[0]), abs(Fcs[:, 1]))
    text.add("FzCS")
    pl.ylabel(r"$|Fz|$ MN")
    pl.ylim([0, 450])
    pl.tight_layout()
    # pl.xlabel(r'Swing Wb')
    sns.despine()
    text.plot()
    # pl.tight_layout()

    pl.sca(ax3)
    ax3.cla()
    text = linelabel(Ndiv=10, value="", postfix="")
    for i, name in enumerate(inv.PF_coils):
        pl.plot(-2 * np.pi * (Swing - Swing[0]), abs(I[i, :]) * 1e-6)
        text.add(name)
        pl.plot(
            -2 * np.pi * (swing * np.ones(2) - Swing[0]), [0, 22], "k--", alpha=0.25
        )
    pl.ylabel(r"$|I|$ MA")
    pl.xlabel(r"Swing Wb")
    pl.tight_layout()
    sns.despine()
    text.plot()
    # pl.tight_layout()


with writer.saving(fig, "../../Movies/{}_swing.wmv".format(config), 100):
    for s in Swing:  # inv.log['position_iter'][-1]
        animate(s)
        writer.grab_frame()


# anim = animation.FuncAnimation(fig,animate,frames=Swing,fargs=([],ax1))
"""
FFMpegWriter = animation.writers['ffmpeg']
metadata = dict(title='SXex sweep')
writer = FFMpegWriter(fps=3, bitrate=5000, metadata=metadata)
anim.save('SX_animation_tmp.wmv',dpi=75,
          savefig_kwargs={'bboxinches':'tight'},
          writer=writer)
"""
