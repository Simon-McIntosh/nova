"""Extract time slices from equilibrium IDS."""

from dataclasses import dataclass, field
from functools import cached_property
from typing import ClassVar

import numpy as np
import xarray


@dataclass
class GetSlice:
    """Convinence method to provide access to sliced ids data."""

    time_index: int | None = field(init=False, default=None)
    data: xarray.Dataset = field(default_factory=xarray.Dataset, repr=False)
    _cache: dict = field(init=False, repr=False, default_factory=dict)

    persist: ClassVar[list[str]] = ["data"]

    def __post_init__(self):
        """Set time index."""
        if hasattr(super(), "__post_init__"):
            super().__post_init__()
        self.itime = self.time_index

    def get_data(self, key: str):
        """Regulate access to imas dataset."""
        var = self.data[self.match(key)]
        assert var.dims[0] == "time"
        data = var[self.itime].data
        try:
            return data.item()
        except ValueError:
            return data

    def __getitem__(self, key: str):
        """Return dataset value with dict-like access."""
        try:
            return self._cache[key]
        except KeyError:
            self._cache[key] = self.get_data(key)
            return self._cache[key]

    def __setitem__(self, key: str, value):
        """Update reference to item in data and cache."""
        self.data[self.match(key)][self.itime] = value
        self._cache[key] = value

    def __iter__(self):
        """Return data iterator."""
        return iter(self.data)

    def match(self, key: str) -> str:
        """Return key matched to internal naming convention."""
        match key:
            case "p_prime":
                return "dpressure_dpsi"
            case "ff_prime":
                return "f_df_dpsi"
            case str():
                return key
            case _:
                raise TypeError(f"invalid key {key}")

    @property
    def itime(self):
        """Manage solution time index."""
        if self.time_index is None:
            self.time_index = 0
        return self.time_index

    @itime.setter
    def itime(self, time_index: int | None):
        if time_index is None:
            return
        self.time_index = time_index
        self.update()

    @staticmethod
    def copy(data):
        """Return copy of instance attribute."""
        try:
            return data.copy(deep=True)  # Dataset
        except TypeError:
            return data.copy()  # dict | np.ndarray
        except AttributeError:  # int | float | tuple
            return data

    def cache(self):
        """Update data cache."""
        for attr in self.persist:
            self._cache[attr] = self.copy(getattr(self, attr))

    def reset(self):
        """Reset data with cached copy."""
        try:
            for attr in self.persist:
                setattr(self, attr, self.copy(self._cache[attr]))
        except KeyError as error:
            raise KeyError(
                "data cache not found - call cache_data to store copy of data in cache."
            ) from error
        self.update()

    @cached_property
    def _partition(self):
        """Return time vector midpoints."""
        time_partition = np.copy(self.data.time)
        time_partition[:-1] += np.diff(self.data.time) / 2
        return time_partition

    def get_itime(self, time: float) -> np.integer:
        """Return searchsorted itime for time."""
        return np.searchsorted(self._partition, time)

    @property
    def time(self):
        """Manage solution time."""
        return self["time"]

    @time.setter
    def time(self, time):
        self.itime = self.get_itime(time)

    def clear_cache(self):
        """Clear data cache."""
        self._cache = {attr: self._cache.get(attr, None) for attr in self.persist}

    def update(self):
        """Clear cache following update to itime. Extend as required."""
        self.clear_cache()
