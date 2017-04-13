import nova.cross_coil as cc
import pylab as pl
import numpy as np
import seaborn as sns
rc = {'figure.figsize': [8, 4], 'savefig.dpi': 100,  # *12/16
      'savefig.jpeg_quality': 100, 'savefig.pad_inches': 0.1,
      'lines.linewidth': 1.5}
sns.set(context='poster', style='white', font='sans-serif', palette='Set2',
        font_scale=7 / 8, rc=rc)
color = sns.color_palette('Set2')
from amigo import geom
from scipy.interpolate import interp1d

coil = {'r': 8, 'z': -10, 'I': 30e6, 'rc': 0.01}

mu_o = 4 * np.pi * 1e-7  # magnetic constant [Vs/Am]

r = np.linspace(-12, 12, 100)
y, z = 0, 0

Br = np.zeros(np.shape(r))
Bz = np.zeros(np.shape(r))

Br_loop = np.zeros(np.shape(r))
Bz_loop = np.zeros(np.shape(r))

Br_circle = np.zeros(np.shape(r))
Bz_circle = np.zeros(np.shape(r))

# prepare coil loop
nArc = 8

#loop = np.zeros((nArc,3))


theta, dtheta = np.linspace(
    0, 2 * np.pi, nArc, endpoint=False, retstep=True)  # angle
loop = np.transpose(np.array([coil['r'] * np.cos(theta), coil['r'] * np.sin(theta),
                              np.array([coil['z']] * nArc)]))  # position

gfl = cc.GreenFeildLoop(loop)
gfl.plot()

for i, ri in enumerate(r):

    B = mu_o * coil['I'] * cc.green_feild(ri, z, coil['r'], coil['z'])
    Br[i] = B[0]
    Bz[i] = B[1]

    B = mu_o * coil['I'] * gfl.B((ri, y, z,))
    Br_loop[i] = B[0]
    Bz_loop[i] = B[2]

    B = mu_o * coil['I'] * cc.green_feild_circle(coil, (ri, y, z,), N=nArc)
    Br_circle[i] = B[0]
    Bz_circle[i] = B[2]


pl.figure()
pl.plot(r, Br)
pl.plot(r, Br_loop, '--')
pl.plot(r, Br_circle, ':')

pl.figure()
pl.plot(r, Bz)
pl.plot(r, Bz_loop, '--')
pl.plot(r, Bz_circle, ':')

print(2 * np.pi * coil['r'])
