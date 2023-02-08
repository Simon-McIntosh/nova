"""Manage fast indexed access to frame arrays."""
from dataclasses import dataclass, field

import numpy as np

from nova.frame.dataframe import DataFrame


@dataclass
class DataLocIndexer:
    """Data Loc base class."""

    name: str
    frame: DataFrame = field(repr=False)
    _data: dict = field(init=False, repr=False)

    def __setitem__(self, key, value):
        """Set data array item in metaframe.data dict."""
        try:
            self._data[key][:] = value
        except KeyError:
            self._data[key[1]][self._data[key[0]]] = value

    def __getitem__(self, key) -> np.ndarray:
        """Return data array item in metaframe.data dict."""
        try:
            return self._data[key]
        except KeyError:
            return self._data[key[1]][self._data[key[0]]]


@dataclass
class ArrayLocIndexer(DataLocIndexer):
    """Access views of cached metaframe.data arrays."""

    attrs: list[str] = field(init=False, default_factory=list)

    def __post_init__(self):
        """Update referance data arrays."""
        self.attrs = self.frame.attrs['metaframe'].array
        self._data = {attr: self.frame[attr] for attr in self.attrs}

    def __call__(self):
        """Return list of frame array attributes."""
        return list(self._data)

    def __len__(self):
        """Return frame array length."""
        return len(self._data)
