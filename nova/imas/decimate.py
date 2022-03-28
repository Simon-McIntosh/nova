from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt
import scipy.spatial
import sklearn
import xarray

from nova.database.filepath import FilePath
from nova.imas.ensemble import Ensemble
from nova.imas.equilibrium import Equilibrium
from nova.linalg.basis import Bernstein, Svd
from nova.linalg.regression import OdinaryLeastSquares


from nova.utilities.pyplot import plt


@dataclass
class Reduce(FilePath):
    """Extract representative subset from signal timeseries."""

    eps: float = 0.5
    metric: str = 'cosine'
    data: xarray.Dataset = field(init=False, repr=False)
    #norm: npt.ArrayLike = field(init=False, repr=False)
    #index: li

    '''
    def __post_init__(self):
        norm = self.signal / self.signal.std(axis=1)
        dist = scipy.spatial.distance.pdist(norm, metric='correlation')
        dist = scipy.spatial.distance.squareform(dist)

        self.index = [0]
        for index in range(1, self.shape[0]):
            if dist[:, index][self.index].min() < self.eps:
                continue
            print(index, self.index)
            self.index.append(index)
        print(self.index)
    '''

    @property
    def attrs(self):
        """Return select attributes."""
        return dict(eps=self.eps, metric=self.metric)

    def build(self, signal: xarray.DataArray):
        """Build select signal dataset."""
        self.data = xarray.Dataset(attrs=self.attrs)
        self.data['signal'] = signal
        self.data['norm'] = signal.std(axis=-1)
        return self

    '''
    @property
    def shape(self):
        """Return signal shape."""
        return self.signal.shape

    def plot(self):
        """Plot signal reduction."""
        plt.plot(np.linspace(0, 1, self.shape[1]), self.signal[self.index].T)

        plt.despine()
        #plt.title(f'signal reduction {len(select)} of {data.dims["time"]}'
        #          f' ({100 * len(select)/data.dims["time"]:1.1f}%)')
    '''


if __name__ == '__main__':

    #ens = Ensemble('DINA-IMAS')
    ens = Ensemble('CORSICA')
    #ens = Ensemble('ASTRA')

    attr = 'f_df_dpsi'
    attr = 'dpressure_dpsi'

    reduce = Reduce().build(ens.data[attr])
    #select.plot()

'''
mean = data[attr].data.mean(axis=1).reshape(-1, 1)
ramp = mean * np.linspace(1, 0, data.dims['psi_norm'])
#signal = data[attr] / np.linalg.norm(data[attr], 2, axis=1).reshape(-1, 1)
signal = data[attr] / data[attr].std(axis=1)

#signal = (data[attr] - data[attr].mean(axis=1)) / data[attr].std(axis=1)
#signal /= data.dims['psi_norm']
'''


#* data.dims['psi_norm']

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
