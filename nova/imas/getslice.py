"""Extract time slices from equilibrium IDS."""
from dataclasses import dataclass, field

import numpy as np
import xarray


@dataclass
class GetSlice:
    """Convinence method to provide access to sliced ids data."""

    time_index: int | None = field(init=False, default=None)
    data: xarray.Dataset | xarray.DataArray = \
        field(default_factory=xarray.Dataset, repr=False)

    def __post_init__(self):
        """Set time index."""
        super().__post_init__()
        self.itime = self.time_index

    def get(self, key: str):
        """Regulate access to imas dataset."""
        data = self.data[self.match(key)][self.itime].data
        try:
            return data.item()
        except ValueError:
            return data

    def __getitem__(self, key: str):
        """Return dataset value with dict-like access."""
        return self.get(key)

    def match(self, key: str) -> str:
        """Return key matched to internal naming convention."""
        match key:
            case 'p_prime':
                return 'dpressure_dpsi'
            case 'ff_prime':
                return 'f_df_dpsi'
            case str():
                return key
            case _:
                raise ValueError(f'invalid key {key}')

    @property
    def itime(self):
        """Manage solution time index."""
        if self.time_index is None:
            raise IndexError('itime is None')
        return self.time_index

    @itime.setter
    def itime(self, time_index: int | None):
        if time_index is None:
            return
        self.time_index = time_index
        self.update()

    @property
    def time(self):
        """Manage solution time."""
        return self['time']

    @time.setter
    def time(self, time):
        self.itime = np.searchsorted(self.data.time, time)

    def update(self):
        """Clear cache following update to itime. Extend as required."""
