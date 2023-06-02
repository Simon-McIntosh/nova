import pylab as pl
from nova.streamfunction import SF
from nova.elliptic import EQ
from nova.config import select
from itertools import cycle
from nova.radial_build import RB
from nova.coils import PF, TF
from nova.DEMOxlsx import DEMO
from nova.loops import Profile
from nova.force import force_feild

import seaborn as sns

rc = {
    "figure.figsize": [7 * 10 / 16, 7],
    "savefig.dpi": 250,  # *12/16
    "savefig.jpeg_quality": 100,
    "savefig.pad_inches": 0.1,
    "lines.linewidth": 0.75,
}
sns.set(
    context="paper",
    style="white",
    font="sans-serif",
    palette="Set2",
    font_scale=7 / 8,
    rc=rc,
)
Color = cycle(sns.color_palette("Set2"))


nTF = 18
config = {"TF": "demo", "eq": "SN2017_SOF"}
config, setup = select(config, nTF=nTF, update=False)

sf = SF(setup.filename)

rb = RB(sf, setup)
pf = PF(sf.eqdsk)
profile = Profile(config["TF"], family="S", part="TF", nTF=nTF, obj="L")
tf = TF(profile=profile)

# pf.plot(coils=pf.coil,label=False,plasma=False,current=True)

demo = DEMO()
demo.fill_part("Vessel")
demo.fill_part("Blanket")
# demo.fill_part('TF_Coil')
demo.plot_ports()
demo.plot_limiter()

pl.axis("off")
tf.fill()

eq = EQ(sf, pf, dCoil=1.5, sigma=0, boundary=sf.get_sep(expand=1.5), n=5e3)
eq.gen_opp()


"""
inv = INV(sf,eq,tf)
L = inv.grid_coils(offset=0.3)
inv.fix_boundary_psi(N=25,alpha=1-1e-4,factor=1)  # add boundary points
inv.fix_boundary_feild(N=25,alpha=1-1e-4,factor=1)  # add boundary points
inv.add_null(factor=1,point=sf.Xpoint)

inv.set_swing(width=250)
inv.initialize_log()
inv.set_background()
inv.get_weight()
inv.set_Lo(L)  # set position bounds
Lnorm = loops.normalize_variables(inv.Lo)
inv.update_position(Lnorm,update_area=True)
inv.plot_fix(tails=True)
eq.gen_opp()
"""

pf.plot(coils=pf.coil, label=True, current=True)
pf.plot(coils=eq.coil, label=False, plasma=True, current=False)
sf.contour(boundary=True)

ff = force_feild(eq.pf.index, eq.pf.coil, eq.coil, eq.plasma_coil, multi_filament=True)

ff.plot(scale=1.5)

# sf.eqwrite(pf,config=config['eq'])
