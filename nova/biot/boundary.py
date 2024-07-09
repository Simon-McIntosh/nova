from nova.frame.coilset import CoilSet
from nova.geometry.polygon import Polygon


coilset = CoilSet(dcoil=-1, dplasma=-500, tplasma="hex", nlevelset=1e3, nwall=50)

coilset.firstwall.insert({"e": [6.2, 0.0, 2.2, 3.0]})
coilset.coil.insert(6.2, [2, -2], 0.25, 0.25)

coilset.plasma.solve()

coilset.sloc["coil", "Ic"] = 5e3
coilset.sloc["plasma", "Ic"] = 5e3


seperatrix = {"e": [5.9, 0.0, 1, 1.6]}
coilset.plasma.separatrix = seperatrix

coilset.plot()
coilset.plasma.plot()
coilset.plasma.lcfs.plot()


coilset.plasma.axes.plot(*Polygon(seperatrix).boundary.T)

# coilset.plasma.axes.plot(*coilset.plasma.separatrix.T)
# coilset.plasma.axes.plot(*coilset.loc["plasma", ["x", "z"]].values.T, ".")
