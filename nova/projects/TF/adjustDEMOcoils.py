from nova.streamfunction import SF
from nova.inverse import INV
from nova.config import select
from itertools import cycle
import numpy as np
from nova.radial_build import RB
from nova.coils import PF
from nova.DEMOxlsx import DEMO
from nova.loops import Profile
from nova.shape import Shape
from nova.firstwall import main_chamber

import seaborn as sns

rc = {
    "figure.figsize": [7 * 10 / 16, 7],
    "savefig.dpi": 150,  # *12/16
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


nTF, nPF, nCS = 18, 6, 5
config = {"TF": "dtt", "eq": "SN"}
config, setup = select(config, nTF=nTF, nPF=nPF, nCS=nCS, update=False)

sf = SF(setup.filename)
rb = RB(sf, setup)
pf = PF(sf.eqdsk)

mc = main_chamber("dtt")
# mc.load_data(plot=True)  # load from file
mc.generate([config["eq_base"]], psi_n=1.07, flux_fit=True, plot=False, symetric=False)
mc.shp.plot_bounds()

rb.generate(mc, debug=False)
profile = Profile(config["TF"], family="S", part="TF", nTF=nTF, obj="L")
shp = Shape(profile, eqconf=config["eq_base"], ny=3)
shp.add_vessel(rb.segment["vessel_outer"])
shp.minimise(ripple=True, verbose=True)
tf = shp.tf
tf.fill()

demo = DEMO()
demo.plot_ports()
# demo.fill_part('Vessel')
# demo.fill_part('Blanket')
# demo.fill_part('TF_Coil')
# demo.plot_limiter()

inv = INV(pf, tf=tf)
inv.load_equlibrium(sf)
inv.fix_boundary()
# inv.fix_target()  # only for SX at the moment
inv.plot_fix(tails=True)

inv.add_plasma()
Lnorm = inv.snap_coils()

inv.set_swing(centre=0, width=20, array=np.linspace(-0.5, 0.5, 3))
inv.set_force_feild()
inv.update_position(Lnorm, update_area=True)
# inv.optimize(Lnorm)  # to optimise the position of the PF coils

pf.plot(label=True, current=True)
pf.plot(subcoil=True, label=False, plasma=True, current=False)
sf.contour(boundary=True)

inv.ff.plot(scale=1.5)

# sf.eqwrite(pf,config=config['eq'])
