from dataclasses import dataclass, field
from functools import cached_property

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
class Scale:
    """Store signal data and manage signal normalization methods."""

    signal: xarray.DataArray
    signal_axis: int = 0
    scale_axis: tuple[int] = field(init=False)
    mean_offset: bool = False
    norm_order: int = 2

    def __post_init__(self):
        """Define scale axis."""
        if self.signal_axis >= self.ndim:
            raise IndexError(f'signal_axis {self.signal_axis} inconsistent '
                             f'with signal shape {self.shape}')
        self.scale_axis = tuple(i for i in range(self.ndim)
                                if i != self.signal_axis)

    def __getitem__(self, method: str):
        """Return result of scaling method."""
        return getattr(self, method)()

    @cached_property
    def shape(self):
        """Return signal shape."""
        return self.signal.shape

    @cached_property
    def ndim(self) -> int:
        """Return signal dimension number."""
        return len(self.shape)

    @cached_property
    def scale_max(self):
        """Return maximum along scale axis."""
        return self.signal.max(axis=self.scale_axis)

    @cached_property
    def scale_min(self):
        """Return minimum along scale axis."""
        return self.signal.main(axis=self.scale_axis)

    @cached_property
    def scale_maxmin(self):
        """Retrun signal maxmin diffrence."""
        return self.signal_max - self.signal_min

    @cached_property
    def scale_mean(self):
        """Return signal mean along scale axis."""
        return self.signal.mean(axis=self.scale_axis)

    @cached_property
    def scale_std(self):
        """Return signal standard deviation along scale axis."""
        return self.signal.std(axis=self.scale_axis)

    @cached_property
    def scale_norm(self):
        """Return signal vector / matrix norm across scale_axis."""
        return np.linalg.norm(self.signal, self.norm_order, self.scale_axis)

    def maxmin(self):
        """Return signal with maxmin scaling."""
        return (self.signal - self.signal_min) / self.signal_maxmin

    def mean(self):
        """Return signal with mean scaling."""
        return (self.signal - self.signal_mean) / self.signal_maxmin

    def standard(self):
        """Return signal with standard (z-score) scaling."""
        if self.mean_offset:
            return (self.signal - self.signal_mean) / self.scale_std
        return self.signal / self.scale_std

    def ramp(self):
        """Return signal with ramped mean offset standard scaling."""
        ramp = self.scale_mean * \
            np.linspace(1, 0, self.shape(self.signal_axis))
        return (self.signal - ramp) / self.scale_std

    def unit(self):
        """Return signal with vector norm scaling."""
        return self.signal / self.scale_norm


@dataclass
class Feature:
    """Extract representative features from signal timeseries."""

    eps: float = 0.25
    metric: str = 'correlation'
    scaler: str = 'standard'
    signal_dim: str = 'time'
    mean_offset: bool = False
    norm_order: int = 2
    data: xarray.Dataset = field(init=False, repr=False)

    @property
    def attrs(self):
        """Return select attributes."""
        return {attr: getattr(self, attr)
                for attr in ['eps', 'metric', 'scaler', 'signal_dim',
                             'mean_offset', 'norm_order']}

    def pairwise(self, signal):
        """Return signal autocorrelation distance matrix."""
        return scipy.spatial.distance.squareform(
            scipy.spatial.distance.pdist(signal, metric=self.metric))

    def extract_features(self) -> list[int]:
        """Return feature index extracted from pairwise distance matrix."""
        feature_index = [0]
        for i in range(1, len(self.data.pairwise)):
            if self.data.pairwise.data[:, i][feature_index].min() < self.eps:
                continue
            feature_index.append(i)
        return feature_index

    def build(self, signal: xarray.DataArray):
        """Build signal dataset."""
        self.data = xarray.Dataset(attrs=self.attrs)
        self.data['signal'] = signal
        self.data['scale'] = Scale(signal)[self.scaler]
        self.data['pairwise'] = (self.signal_dim, self.signal_dim), \
            self.pairwise(self.data['scale'])
        self.data.coords['feature_index'] = self.extract_features()
        self.data['feature'] = self.data.scale[self.data.feature_index]
        return self

    def plot(self):
        """Plot signal reduction."""
        feature_dim = next(dim for dim in self.data.signal.dims
                           if dim != self.signal_dim)
        plt.plot(self.data[feature_dim], self.data.scale.T,
                 color='lightgray', lw=0.2)
        plt.plot(self.data[feature_dim],
                 self.data.feature.T)

        plt.despine()
        nsignal = self.data.dims[self.signal_dim]
        nfeature = self.data.dims["feature_index"]
        plt.title(f'signal reduction {nfeature} of {nsignal}'
                  f' ({100 * nfeature/nsignal:1.1f}%)')


if __name__ == '__main__':

    #ens = Ensemble('DINA-IMAS')
    ens = Ensemble('DINA')
    #ens = Ensemble('ASTRA')
    ens.build()

    attr = 'f_df_dpsi'
    attr = 'dpressure_dpsi'

    feature = Feature().build(ens.data[attr])
    feature.plot()


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
