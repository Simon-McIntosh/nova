import numpy as np
import pylab as pl
from nova.finite_element import FE
from nova.properties import second_moment
from amigo import geom
import seaborn as sns

sns.set_context("talk")
sns.set_style("white")

fe = FE(frame="3D")
sm = second_moment()
sm.add_shape("circ", r=0.2, ro=0.199)
C, I, A = sm.report()
section = {"C": C, "I": I, "A": A, "J": I["xx"], "pnt": sm.get_pnt()}
fe.add_mat("bar", ["steel_cast"], [section])

# extract sectional properties
w = -9.81 * fe.mat[0]["mat_o"]["rho"] * fe.mat[0]["mat_o"]["A"]  # weight / l
E = fe.mat[0]["mat_o"]["E"]
Iy = fe.mat[0]["mat_o"]["I"][1]

print("w", w)


N = 4
X = np.zeros((N, 3))

X[:, 0] = [0, 2, 3, -1]
X[:, 2] = [0, 1, 1, -5]


fe.add_nodes(X)
fe.add_elements(part_name="bar", nmat="bar")

fe.add_bc("fix", 0, part="bar", ends=0)
# fe.add_bc('ny', -1, part='bar', ends=1)

fe.plot_nodes()

"""
for el in range(fe.nel):
    W = w * fe.el['dl'][el]
    for nd in fe.el['n'][el]:
        fe.add_nodal_load(nd, 'fz', W/2)
"""

fe.add_nodal_load(3, "fx", 500e3)
fe.add_nodal_load(3, "fz", -1000e3)

# fe.add_weight()  # add weight to all elements
fe.solve()

fe.deform(1e-1)
fe.plot_F(scale=1e-6)
fe.plot_displacment()
fe.plot_nodes()
"""
pl.figure()
pl.plot(fe.part['bar']['l'], E*Iy*fe.part['bar']['d2u'][:, 2])
"""
print(
    np.min(E * Iy * fe.part["bar"]["d2u"][:, 2]),
    np.max(E * Iy * fe.part["bar"]["d2u"][:, 2]),
)
