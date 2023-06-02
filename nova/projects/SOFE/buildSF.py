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

import seaborn as sns

sns.set(context="talk", style="white")
Color = cycle(sns.color_palette("Set2"))

pkl = PKL("DEMO_SF", directory="../../Movies/")

nTF = 16
base = {"TF": "demo", "eq": "DEMO_SF"}
config, setup = select(base, nTF=nTF, update=False)

setup_ref = Setup("SFm")  # referance plasma
sf = SF(setup_ref.filename)
pf = PF(sf.eqdsk)


profile = Profile(config["TF"], family="S", part="TF", nTF=nTF, obj="L")
tf = TF(profile=profile, sf=sf)

# inv = INV(pf, tf, dCoil=0.5)
# inv.colocate(sf, n=1e4, expand=2.75, centre=0, width=363/(2*np.pi))
# inv.wrap_PF(solve=True)
# inv.optimize()


mc = main_chamber("demo_SFm", date="2017_06_03")
mc.generate(
    ["SFm", "SFp"],
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

# tf.minimise(rb.segment['vessel_outer'], ripple=True,
#             ny=1, nr=3, verbose=True)
tf.fill()


inv = INV(pf, tf, dCoil=0.5)
inv.colocate(sf, n=5e3, expand=2.5, centre=0, width=100 / (2 * np.pi), setup=setup)
swing = SWING(inv, sf)

inv.grid_CS(5)
# inv.grid_PF(5)
# inv.wrap_PF(solve=False)
inv.optimize()


inv.plot_fix()

inv.eq.run(update=False)
sf.contour()
pf.plot(subcoil=True, plasma=True)

swing.output()


# sf.eqwrite(pf, config=setup.configuration)
# pkl.write(data={'sf': sf, 'inv': inv, 'rb': rb, 'tf': tf})  # pickle data
