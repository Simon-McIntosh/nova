import numpy as np

from nova.structural.finite_element import FE
from nova.structural.properties import second_moment

fe = FE(frame="3D")
sm = second_moment()
sm.add_shape("circ", r=0.02, ro=0.01)
C, I, A = sm.report()

section = {"C": C, "I": I, "A": A, "J": I["xx"], "pnt": sm.get_pnt()}
fe.add_mat("bar", ["steel_cast"], [section])

# extract sectional properties
w = -9.81 * fe.mat[0]["mat_o"]["rho"] * fe.mat[0]["mat_o"]["A"]  # weight / l
E = fe.mat[0]["mat_o"]["E"]
Iy = fe.mat[0]["mat_o"]["I"][1]

N = 4
L = 2
theta = np.pi / 6

X = np.zeros((N, 3))
X[:, 0] = np.array(
    [0, L * np.cos(theta), L * (1 + np.cos(theta)), L * (2 + np.cos(theta))]
)
X[:, 2] = np.array([0, -L * np.sin(theta), -L * np.sin(theta), -L * np.sin(theta)])

fe.add_nodes(X)
fe.add_elements(part_name="comp", nmat="bar")

fe.add_bc("fix", [0], part="comp", ends=0)

fe.add_weight()  # add weight to all elements
fe.solve()

M = (2 + np.cos(theta)) ** 2 * L**2 * w / 2
u = M * L**2 / (2 * E * Iy)

print(fe.D["x"][1] * np.sin(theta) + fe.D["z"][1], u)

fe.plot(scale_factor=-0.75)
