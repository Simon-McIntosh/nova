from spline_coil import Scoil
import pylab as pl
from nova.beam import Dcoil

import seaborn as sns

rc = {
    "figure.figsize": [10, 10 * 12 / 16],
    "savefig.dpi": 100,
    "savefig.jpeg_quality": 100,
    "savefig.pad_inches": 0.1,
    "lines.linewidth": 2,
}
sns.set(
    context="talk",
    style="white",
    font="sans-serif",
    palette="Set2",
    font_scale=7 / 8,
    rc=rc,
)

sc = Scoil(npoints=200)

x = sc.generate([0.66, 0.15, 0.9], ["upper", "top", "l"])

pl.plot(x["r"], x["z"], "-")

x, z = Dcoil.pD(4.486, 15.708)
# pl.plot(x,z,'--')

pl.axis("equal")
sns.despine()
