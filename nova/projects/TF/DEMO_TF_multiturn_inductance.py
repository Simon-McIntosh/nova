import pylab as pl
from nova.config import Setup
from nova.streamfunction import SF
from nova.radial_build import RB
from nova.coils import PF, TF
from etna.coil_geom import configure, coil_cage
from etna.coil_apdl import coil_map
from inductance import neumann
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from time import time
import sys
import datetime
import pickle
from amigo import geom
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


def rotate(theta):
    Rz = np.array(
        [
            [np.cos(theta), -np.sin(theta), 0],
            [np.sin(theta), -np.cos(theta), 0],
            [0, 0, 1],
        ]
    )
    return Rz


r1, r2 = 4.486, 15.708  # DEMO baseline
r, z = Dcoil.pD(r1, r2, npoints=200)
Xo = np.zeros((len(r), 3))
Xo[:, 0], Xo[:, 2] = r, z

nTF = 18
theta = np.linspace(0, 2 * np.pi, nTF, endpoint=False)


r = np.sqrt(0.625 * 1.24)
nturn = 180

neu = neumann()

fig = pl.figure()
ax = fig.gca(projection="3d")

N = 18
M = np.zeros((N, N))
to = time()
for i in np.arange(0, N):
    X = np.dot(Xo, rotate(theta[i]))
    neu.setX(X)
    neu.setr(r)
    ax.plot(
        np.append(X[:, 0], X[0, 0]),
        np.append(X[:, 1], X[0, 1]),
        np.append(X[:, 2], X[0, 2]),
    )

    for j in np.arange(i, N):
        X_ = np.dot(Xo, rotate(theta[j]))
        neu.setX_(X_)
        M[i, j] = nturn**2 * neu.calculate()
        if i != j:
            M[j, i] = M[i, j]  # copy to lower

    t1 = time()
    t_fraction = 2 * ((i + 1) * (N - (i + 1) / 2)) / N**2
    t_wall = t1 - to
    t_total = t_wall / t_fraction
    t_remain = t_total - t_wall
    t_str = "\rtotal: " + str(datetime.timedelta(seconds=np.round(t_total))) + "\t"
    t_str += "complete:{:3.0f}%\t".format(100 * t_fraction)
    t_str += "remaining: " + str(datetime.timedelta(seconds=np.round(t_remain)))
    sys.stdout.write(t_str)
    sys.stdout.flush()

filename = "TFtmp_{:1.0f}_{:1.0f}".format(N, cgeom.Nseg)
# with open('D:/Code/Etna/Data/Inductance/'+filename+'.pkl', 'wb') as output:
#    pickle.dump(M,output,-1)

pl.axis("off")
ax.auto_scale_xyz([-10, 10], [-10, 10], [-15, 5])
np.set_printoptions(threshold=100, edgeitems=5, precision=3, linewidth=150)
print("")
print(M)
print(np.sum(M))
