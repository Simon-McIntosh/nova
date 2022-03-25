
import numpy as np
import scipy.spatial
import sklearn

from nova.linalg.basis import Bernstein, Svd
from nova.linalg.regression import OdinaryLeastSquares
from nova.imas.ensemble import Ensemble
from nova.imas.equilibrium import Equilibrium

from nova.utilities.pyplot import plt

ens = Ensemble('DINA_IMAS')
ens = Ensemble('CORSICA')
data = ens.data

#@dataclass

attr = 'f_df_dpsi'
#attr = 'dpressure_dpsi'
mean = data[attr].data.mean(axis=1).reshape(-1, 1)
ramp = mean * np.linspace(1, 0, data.dims['psi_norm'])
signal = (data[attr] - ramp) / data[attr].std(axis=1)

#signal = (data[attr] - data[attr].mean(axis=1)) / data[attr].std(axis=1)
#signal /= data.dims['psi_norm']

dist = scipy.spatial.distance.pdist(signal, metric='correlation')
dist = scipy.spatial.distance.squareform(dist)

eps = 0.05 #* data.dims['psi_norm']

select = [0]


def threshold(index: int, eps: float):
    while index < data.dims['time']:
        if dist[:, index][select].min() < eps:  # duplicate observation
            index += 1
            continue
        select.append(index)
        yield index


for index in threshold(1, eps):
    plt.plot(data.psi_norm, signal[index])

plt.despine()
plt.title(f'signal reduction {100 * len(select)/data.dims["time"]:1.1f}% '
          f'{len(select)} of {data.dims["time"]}')

#plt.ylim([-0.001, 0.001])

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
