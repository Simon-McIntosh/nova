from amigo.addtext import linelabel
import pylab as pl
from nova.streamfunction import SF
from nova.config import select
from itertools import cycle

import seaborn as sns

rc = {
    "figure.figsize": [7 * 10 / 16, 7],
    "savefig.dpi": 250,  # *12/16
    "savefig.jpeg_quality": 100,
    "savefig.pad_inches": 0.1,
    "lines.linewidth": 1,
}
sns.set(
    context="poster",
    style="white",
    font="sans-serif",
    palette="Set2",
    font_scale=0.75,
    rc=rc,
)
Color = cycle(sns.color_palette("Set2"))

text = linelabel(Ndiv=20, value="")

nTF = 18

for eq in ["DEMO_SN_SOF", "DEMO_SN_EOF"]:
    config = {"TF": "demo", "eq": eq}
    config, setup = select(config, nTF=nTF, update=False)
    sf = SF(setup.filename)
    sf.get_boundary(plot=True, alpha=1 - 1e-5)
    text.add(eq)
    sf.sol(plot=True)


text.plot()
pl.axis("equal")
pl.axis("off")
