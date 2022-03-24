
import numpy as np
import scipy.spatial
import sklearn

from nova.linalg.basis import Bernstein, Svd
from nova.linalg.regression import OdinaryLeastSquares
from nova.imas.ensemble import Ensemble
from nova.imas.equilibrium import Equilibrium

from nova.utilities.pyplot import plt

ens = Ensemble('DINA-IMAS')
data = ens.data

metric = 'cityblock'

dist = scipy.spatial.distance.pdist(data.f_df_dpsi.data, metric=metric)
dist = scipy.spatial.distance.squareform(dist)

eps = 5

np.argmax(dist[0] > eps)

_data = [data.f_df_dpsi.data[0]]


def threshold(index: int, eps: float):
    while True:
        if (step := np.argmax(dist[index, index:] > eps)) == 0:
            break
        index += step

        _dist = scipy.spatial.distance.cdist(
            _data, data.f_df_dpsi.data[index].reshape(1, -1), metric=metric)
        if _dist.min() < eps/2:  # duplicate observation
            continue
        _data.append(data.f_df_dpsi[index])
        yield index


for index in threshold(1, eps):
    plt.plot(data.psi_norm, data.f_df_dpsi[index])

'''
#data = Equilibrium(135011, 7).data

basis = Bernstein(data.dims['psi_norm'], 3)

basis = Svd(data.dims['psi_norm'], 3)(data.f_df_dpsi)

ols = OdinaryLeastSquares(basis.matrix)

model = np.zeros((data.dims['time'], basis.order+1))
for itime in range(data.dims['time']):
    model[itime] = ols / data.f_df_dpsi[itime].data


tsne = sklearn.manifold.TSNE(perplexity=30).fit_transform(model)

labels = sklearn.cluster.OPTICS(min_samples=2).fit_predict(model)

for label in np.unique(labels):
    if label == -1:
        color = 'gray'
    else:
        color = f'C{label%10}'
    index = labels == label
    plt.plot(tsne[index, 0], tsne[index, 1], '.', color=color)
plt.axis('equal')
#plt.axis('off')
plt.title(f'Unique labels {len(np.unique(labels))-1}')
'''


'''
index = labels == 3

ols /= data.f_df_dpsi.data[index][-1]

_model = np.mean(model[index], axis=0)
ols.update_model(_model)
ols.plot()
'''
