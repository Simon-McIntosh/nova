import numpy as np
import pylab as pl
from sklearn.gaussian_process import GaussianProcessRegressor as GPR
import matplotlib.cm as cm
import pickle
from nova import loops

nPF = 5

with open('../../Data/rms_cube_5L.pkl', 'rb') as input:
    cube = pickle.load(input)
    rms = pickle.load(input)
cube[:, :nPF] = np.sort(cube[:, :nPF])
cube[:, nPF:] = np.sort(cube[:, nPF:])

pca = PCA(n_components=inv.nL)
pca.fit(cube[np.argsort(rms)[:200]])
Leig = pca.components_


index = np.argsort(rms)
rms = rms[index]
cube = cube[index, :]

rms = rms[::100]
cube = cube[::100, :]


# Instanciate a Gaussian Process model
#gp = GaussianProcess(corr='cubic', theta0=1e-2,thetaL=1e-4,thetaU=1e-1)
gp = GPR()  # corr='squared_exponential',theta0=1e-2,thetaL=1e-4,thetaU=1e-1
gp.fit(cube, rms)

npoint = 2e3
lo = np.linspace(0, 1, int(np.sqrt(npoint)))
l1 = np.linspace(0, 1, int(np.sqrt(npoint)))
Lo, L1 = np.meshgrid(lo, l1, indexing='ij')

x = np.array([Lo.reshape(-1), L1.reshape(-1)])
y = gp.predict(x.T)

pl.pcolor(lo, l1, y.reshape(len(lo), len(l1)).T, cmap=cm.Spectral)
CB = pl.colorbar()
CS = pl.contour(lo, l1, y.reshape(len(lo), len(l1)).T,
                15, colors='k', alpha=0.5)
# CB.ax.set_ylabel('RMTFI')
pl.clabel(CS)

pl.xlabel('lo')
pl.ylabel('l1')


# print(gp.predict(np.array([4,18]).reshape(1,-1)))
