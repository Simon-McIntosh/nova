from nova.streamfunction import SF
from nova.inverse import INV
from nova.config import Setup, select
from itertools import cycle
import numpy as np
from nova.radial_build import RB
from nova.shelf import PKL
from nova.coils import PF, TF
from nova.loops import Profile
from nova.inverse import SWING
from nova.firstwall import main_chamber
import pylab as pl

import seaborn as sns

sns.set(context="talk", style="white", font="sans-serif", font_scale=7 / 8)
Color = cycle(sns.color_palette("Set2"))

pkl = PKL("DEMO_SN", directory="../../Movies/")

nPF, nCS, nTF = 5, 5, 16
base = {"TF": "demo", "eq": "DEMO_SN"}
config, setup = select(base, nTF=nTF, update=True)

setup_ref = Setup("DEMO_SN_EOF")  # referance plasma
sf = SF(setup_ref.filename)
pf = PF(sf.eqdsk)

mc = main_chamber("demo_SN", date="2017_06_01")
mc.generate(
    ["DEMO_SN_SOF", "DEMO_SN_EOF"],
    dx=0.225,
    psi_n=1.07,
    flux_fit=True,
    plot=False,
    symetric=False,
    plot_bounds=False,
    verbose=False,
)

rb = RB(sf, setup)
rb.generate(mc, plot=True, DN=False, debug=False)

profile = Profile(config["TF"], family="S", part="TF", nTF=nTF, obj="L")
tf = TF(profile=profile, sf=sf)
# tf.minimise(rb.segment['vessel_outer'], ripple=True,
#             ny=1, nr=1, verbose=True)
tf.fill()


R.TF = tf  # BP hookup (can remove when run from BP)
excl = R.define_port_exclusions(plot=True)  # very nice


inv = INV(pf, tf, dCoil=0.5)  # 0.5 for production
inv.colocate(sf, n=5e3, expand=2, centre=0, width=363 / (2 * np.pi), setup=setup)

inv.grid_PF(nPF)
inv.grid_CS(nCS)

# inv.set_limit(FPFz=200)
inv.Lex = R.TF.xzL(excl)  # translate to normalized coil length
inv.set_sail()

inv.wrap_PF(solve=True)
# inv.optimize()

swing = SWING(inv, sf, output=True)
swing.output()


# plot
inv.plot_fix()
sf.contour()
pf.plot(current=True, label=True)

pf.plot(subcoil=True, plasma=True)
inv.ff.plot(scale=10)
pl.axis("equal")
pl.axis("off")


# sf.eqwrite(pf, config=setup.configuration)
# pkl.write(data={'sf': sf, 'inv': inv, 'rb': rb, 'tf': tf})  # pickle data
