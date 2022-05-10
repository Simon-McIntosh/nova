"""Manage overlap errorfield calculations."""
from dataclasses import dataclass
from typing import ClassVar

import numpy as np
import xarray

from nova.assembly.model import Dataset
from nova.database.filepath import FilePath


@dataclass
class Model(Dataset):
    """Predict overlap error field."""

    filename: str = 'errorfield'

    points: ClassVar[dict] = dict(A=np.array([2612.7, -3688.8]),
                                  C=np.array([0, 6277.3]),
                                  E=np.array([-4240.3, -3683.8]),
                                  F=np.array([0, -6303.5]))
    coordinates: ClassVar[list[str]] = ['radial', 'tangential', 'vertical',
                                        'pitch', 'yaw', 'roll']
    overlap_limit: ClassVar[dict[int, float]] = \
        {1: 4.2e-4, 2: 2.5e-4, 3: 0.8e-4}
    ncoil: ClassVar[int] = 18

    def __post_init__(self):
        """Load finite impulse response filter."""
        super().__post_init__()
        self.load_filter()

    def load_filter(self):
        """Extract filter from dataset."""
        self.filter = {plasma: self._filter(plasma).data
                       for plasma in range(1, 4)}

    def build(self):
        """Load data extracted from Y26X3K_v1_0."""
        data = FilePath(self.path, filename='displace').load().data
        self.data = xarray.Dataset(data.coords)
        self.data['coefficient'] = ['real', 'imag']
        self.data['signal_fft'] = xarray.DataArray(0., self.data.coords)
        self.data['response_fft'] = xarray.DataArray(0., self.data.coords)
        self.data['filter'] = xarray.DataArray(0., self.data.coords)
        self.data.response_fft[..., 0] = data['real'].T
        self.data.response_fft[..., 1] = data['imag'].T
        self.data['signal'] = data.displacment.T
        self.data.signal_fft[..., 0] = self.data['signal']  # impulse
        _filter = (self.data.response_fft[..., 0] +
                   1j * self.data.response_fft[..., 1]) / \
            (self.data.signal_fft[..., 0] + 1j * self.data.signal_fft[..., 1])
        self.data.filter[..., 0] = _filter.real
        self.data.filter[..., 1] = _filter.imag

        length_AE = np.linalg.norm(self.points['E'] - self.points['A'])
        length_CF = np.linalg.norm(self.points['F'] - self.points['C'])
        self.data.filter[3] *= length_AE  # pitch
        self.data.filter[4] *= length_CF  # yaw
        self.data.filter[5] *= length_CF  # roll
        return self.store()

    def _filter(self, plasma: int):
        """Return complex filter."""
        return self.data.filter[..., 0].sel(plasma=plasma) + \
            1j * self.data.filter[..., 1].sel(plasma=plasma)

    def predict(self, plasma: int, *signals):
        """Return prediction."""
        response = 0
        for signal, coordinate in zip(signals,
                                      ['radial', 'tangential', 'vertical',
                                       'pitch', 'roll', 'yaw']):
            if signal is None:
                continue
            index = self.coordinates.index(coordinate)
            signal_fft = np.fft.rfft(signal, n=self.ncoil)[..., 1]
            response += signal_fft * self.filter[plasma][index]
        return abs(response) / self.overlap_limit[plasma]


if __name__ == '__main__':

    model = Model()
