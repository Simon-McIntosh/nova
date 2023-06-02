import numpy as np
from amigo.pyplot import plt
import SchemDraw as schem
import SchemDraw.elements as e


def impulse_capacitor(ax=None):
    if ax is None:
        ax = plt.subplots(1, 1)[1]
    d = schem.Drawing(unit=6, inches_per_unit=1, fontsize=16)
    ind = d.add(
        e.transformer(t1=8, t2=4, loop=False),
        d="right",
        flip=True,
        lftlabel="VS3\nturns",
        rgtlabel="passive\nfilaments",
    )
    link = d.add(e.LINE, xy=ind.s1, l=d.unit / 8)
    d.add(e.LINE, xy=ind.s2, l=d.unit / 8)
    d.add(e.LINE, xy=ind.p2, l=d.unit / 2, d="left")
    SPDT = d.add(e.SWITCH_SPDT2_OPEN, d="left", flip=True, label="$t=t_{trip}$")
    Vbg = d.add(
        e.SOURCE_V,
        l=d.unit / 2,
        xy=ind.p1,
        tox=SPDT.c[0],
        d="left",
        botlabel="$V_{bg}$",
    )
    d.add(e.DOT)
    d.add(e.CAP, to=SPDT.c, label="{:1.2f}H".format(2.42))
    d.add(e.LINE, xy=[SPDT.c[0], Vbg.end[1]], l=d.unit / 2, d="left")
    d.add(e.SOURCE_V, toy=SPDT.b[1], d="up", label="$V_{ps}$")
    d.add(e.LINE, to=SPDT.b)
    d.draw(ax=ax)


if __name__ == "__main__":
    circuit = impulse_capacitor()
