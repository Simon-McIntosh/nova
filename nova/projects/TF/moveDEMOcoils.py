import pylab as pl
from nova.streamfunction import SF
from nova.elliptic import EQ
from nova.inverse import INV
from nova.config import Setup
from itertools import cycle
import numpy as np
from nova.radial_build import RB
from nova.shelf import PKL
from nova.coils import PF, TF
from nova import loops
from DEMOxlsx import DEMO
from nova.loops import Profile

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

pkl = PKL("DEMO_SN54")
nTF = 16

config = {"TF": "SN", "eq": "DEMO_SN"}
setup = Setup(config["eq"])
sf = SF(setup.filename)

rb = RB(setup, sf)
pf = PF(sf.eqdsk)
tf = TF(Profile(config["TF"], family="S", part="TF", nTF=nTF, obj="L", load=True))

eq = EQ(sf, pf, dCoil=2.0, sigma=0, boundary=sf.get_sep(expand=1.5), n=1e4)
eq.gen_opp()


inv = INV(sf, eq, tf)
Lpf = inv.grid_PF(nPF=4)
Lcs = inv.grid_CS(nCS=3, Zbound=[-8.2, 11], gap=0.1)
Lo = np.append(Lpf, Lcs)
inv.update_coils()

inv.fit_PF(offset=0.3)  # fit PF coils to TF
inv.fix_boundary_psi(N=25, alpha=1 - 1e-4, factor=1)  # add boundary points
inv.fix_boundary_feild(N=25, alpha=1 - 1e-4, factor=1)  # add boundary points
inv.add_null(factor=1, point=sf.Xpoint)

inv.set_swing()
inv.update_limits(LCS=[-9.5, 9.5])

Lo = inv.optimize(Lo)

inv.fix_flux(inv.swing["flux"][1])
inv.solve_slsqp()

# eq = EQ(sf,pf,dCoil=2,sigma=0,boundary=tf.get_loop(expand=0),n=1e4)

eq.get_Vcoil()
eq.gen_opp()
rb.firstwall(mode="calc", plot=True, debug=False)
sf.contour()

pf.plot(coils=pf.coil, label=True, plasma=True, current=True)
sf.contour(boundary=False)

inv.plot_fix(tails=True)
tf.fill()
demo = DEMO()
demo.fill_part("Vessel")
demo.fill_part("Blanket")
pl.plot(demo.limiter["L3"]["r"], demo.limiter["L3"]["z"])
pl.axis("equal")
pl.tight_layout()

loops.plot_variables(inv.Io, scale=1, postfix="MA")
loops.plot_variables(inv.Lo, scale=1)

pkl.write(data={"sf": sf, "eq": eq, "inv": inv})  # pickle data

sf.eqwrite(pf, config=config["TF"] + "_{:d}PF_{:d}TF".format(inv.nPF, nTF))

"""

#inv.write_swing()
#sf.eqwrite(config='SXex')

for swing in np.linspace(-20,80,5):
    pl.figure()
    pl.axis('equal')
    pl.axis('off')

    inv.swing_fix(swing)
    inv.solve() 
    
    inv.update_coils(plot=True)
    sf.plot_coils(Color,coils=sf.coil,label=False,plasma=False,current=True) 
    sf.plot_coils(Color,coils=eq.coil,label=False,plasma=False) 
 
    eq.run()
    
    sf.contour()
    eq.plasma()
    #eq.plotb()
    #sf.eqwrite(config='SXex')
    pl.plot(sf.rbdry,sf.zbdry,'--')
    inv.plot_fix()

print('L3D',inv.rb.sol.connection('outer',0)[-1][-1])
print('R',Rsol[-1])
print('R/X',Rsol[-1]/sf.Xpoint[0])
print('Itotal',inv.Itotal*1e-6,'MA')
print('R',rb.targets['outer']['Rsol'][-1],'Z',
      rb.targets['outer']['Zsol'][-1])
"""
