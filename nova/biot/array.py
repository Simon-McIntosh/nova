"""Caching methods to enable fast access to xarray attributes."""

from dataclasses import dataclass, field

import xarray


@dataclass
class Array:
    """Cache fast access xarray attributes within array dict."""

    data: xarray.Dataset = field(repr=False, default_factory=xarray.Dataset)
    array_attrs: list[str] = field(default_factory=list)
    array: dict = field(init=False, repr=False, default_factory=dict)

    def load_arrays(self):
        """Link data arrays."""
        for attr in self.array_attrs:
            if attr in self.data:
                self.array[attr] = self.data[attr].data
        if hasattr(super(), "load_arrays"):
            super().load_arrays()

    def __getitem__(self, attr):
        """Return array attribute via dict-like access."""
        if attr in self.array_attrs:
            return self.array[attr]
        if hasattr(super(), "__getitem__"):
            return super().__getitem__(attr)
        raise KeyError(f"{attr}")
