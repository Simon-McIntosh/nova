from dataclasses import dataclass, field
from typing import Union

import numpy as np
import scipy
import sklearn.manifold
import pandas

from nova.utilities.localdata import LocalData
from nova.frame.IO.read_scenario import scenario_data
import matplotlib.pyplot as plt


@dataclass
class SampleWaveForm:
    """Extract waveform from DINA simulation."""

    scenario: int

    def __post_init__(self):
        self.data = self._extract(self.scenario)

    def _extract(self, scenario):
        scenario = scenario_data(scenario)
        source_columns = ["t", "Rcur", "Zcur"]
        current_iloc = np.unique(scenario.Ic_iloc)
        source_columns.extend([label for label in scenario.index[current_iloc]])
        return scenario.frame.loc[:, source_columns]


if __name__ == "__main__":
    from sklearn.decomposition import KernelPCA

    data = []

    for i in range(-20, -6):
        sample = SampleWaveForm(i)
        current = sample.data.iloc[:, 3:15]
        current = current[current.loc[:, "Ip"] < -0.01]
        data.append(current.iloc[::100, :])
    data = pandas.concat(data)

    transformer = KernelPCA(n_components=2, kernel="rbf", fit_inverse_transform=True)
    X = transformer.fit_transform(data)

    plt.plot(*X.T, ".")

    """
    for i in range(-3, 0):
        print(i)
        sample = SampleWaveForm(i)
        manifold = sklearn.manifold.MDS(n_components=2)
        print('embedding')
        
        embedded = manifold.fit_transform(sample.data.iloc[::50, 1:])
        
        step = int(len(embedded) / 9)
        for i in range(len(embedded) // step):
            plt.plot(*embedded.T[:, i*step:(i+1)*step], '.')
    """
