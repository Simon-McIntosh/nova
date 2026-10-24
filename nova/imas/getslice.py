"""Extract time slices from equilibrium IDS."""

from dataclasses import dataclass, field
from functools import cached_property
from typing import ClassVar

import numpy as np
import xarray


def vacuum_field(b0: np.ndarray, index: int) -> float:
    """Return the toroidal field an equilibrium slice is read against [T].

    A pulse may store one value for the whole IDS or one per slice, and the
    convention check needs the same reading as the Data Dictionary translation,
    so the rule lives here once rather than in each of its callers.
    """
    if b0.size == 0:
        return float("nan")
    return float(b0[0] if b0.size == 1 else b0[index])


def ids_vacuum_field(ids, index: int) -> float:
    """Return the toroidal field one equilibrium IDS slice is read against [T].

    The IDS carries the field either once at the top level or once per slice,
    so a top-level array that is empty is read from the slice instead. That
    fallback belongs with the value rule above rather than in the readers that
    need it: a reader that restated it could drift from
    :func:`vacuum_field` and check the convention against a different field
    from the one it translates.
    """
    values = np.asarray(ids.vacuum_toroidal_field.b0).ravel()
    if values.size == 0:
        values = np.asarray(ids.time_slice[index].vacuum_toroidal_field.b0).ravel()
    return vacuum_field(values, index)


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

    @staticmethod
    def translated(signal_map, path: str, values):
        """Return one field read through the COCOS translator.

        A reader that declares its source convention serves every
        convention-bearing field through here, so the data it exposes is in
        nova's COCOS 17 whatever Data Dictionary version it was read from. The
        factors come from the signal map
        :func:`nova.io.sourcemap.translate_equilibrium` builds, so no reader
        writes a sign of its own.
        """
        (signal,) = signal_map.for_target(path)
        return signal.apply(values)

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
            raise IndexError("itime is None")
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
