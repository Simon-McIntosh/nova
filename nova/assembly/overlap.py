"""Manage overlap errorfield calculations."""
from dataclasses import dataclass

import numpy as np
import xarray

from nova.assembly.model import Dataset
from nova.database.filepath import FilePath


@dataclass
class Overlap(Dataset):
    """Predict overlap error field."""

    filename: str = 'errorfield'

    def build(self):
        """Load data extracted from Y26X3K_v1_0."""
        data = FilePath(self.path, filename='displace').load().data
        self.data = xarray.Dataset(data.coords)
        self.data['coefficient'] = ['real', 'imag']
        self.data['signal_fft'] = xarray.DataArray(0., self.data.coords)
        self.data['response_fft'] = xarray.DataArray(0., self.data.coords)
        self.data.response_fft[..., 0] = data['real'].T
        self.data.response_fft[..., 1] = data['imag'].T
        self.data['signal'] = data.displacment.T

        signal_fft = np.fft.rfft(
            self.data.signal.values[..., np.newaxis], n=18)[..., 0]
        self.data.signal_fft[..., 0] = signal_fft.real
        self.data.signal_fft[..., 1] = signal_fft.imag
    '''
    @property
    def response_coef(self):
        """Return complex response coefficents."""
        return self.data.response_fft[..., 0] + \
            1j * self.data.response_fft[..., 1]
    '''

        #x = arcsin(y/CF)
        #y = arcsin(z/AE)
        #z = arcsin(y/AE)


if __name__ == '__main__':

    overlap = Overlap()
