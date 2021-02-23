"""Manage fast access dataframe attributes."""
from abc import ABCMeta, abstractmethod
from dataclasses import dataclass, field
from contextlib import contextmanager

import numpy as np
import pandas

from nova.electromagnetic.metadata import MetaData


@dataclass
class MetaArray(MetaData):
    """Manage Frame metadata - accessed via Frame['attrs']."""

    array: list[str] = field(default_factory=lambda: [])
    restrict: list[str] = field(default_factory=lambda: [])
    data: dict[str, np.ndarray] = field(default_factory=dict)
    update_array: dict[bool] = field(default_factory=dict)
    update_frame: dict[bool] = field(default_factory=dict)

    _internal = ['data', 'update_array', 'update_frame']
    _lock = True

    @contextmanager
    def unlock(self):
        """Permit update to restricted variables."""
        self._lock = False
        yield
        self._lock = True

    def check_lock(self, key):
        """Check lock on restricted attributes."""
        if key in self.restrict and self._lock:
            raise PermissionError(f'Access to key: {key} is restricted. '
                                  f'Access via metadata.unlock.\n'
                                  'with frame.metaarray.unlock():\n'
                                  f'    frame.{key} = *')

    def __repr__(self):
        """Return __repr__."""
        repr_data = {field: getattr(self, field).values()
                     for field in ['update_array', 'update_frame']}
        return pandas.DataFrame(repr_data, index=self.array).__repr__()

    def validate(self):
        """Extend MetaData.validate, set default update flags."""
        super().validate()
        self.update_flag('array', True)
        self.update_flag('frame', False)

    def update_flag(self, instance, default):
        """Set flag defaults for new attributes."""
        attribute = getattr(self, f'update_{instance}')
        attribute |= {attr: default for attr in self.array
                      if attr not in attribute}
        setattr(self, f'update_{instance}',
                {attr: attribute[attr] for attr in self.array})


class Array(metaclass=ABCMeta):
    """
    Abstract base class enabling fast access to dynamic fields in array.

    Extended by Frame. Inherited alongside DataFrame.
    Fast access variables stored as np.arrays _*
    Lazy data exchange implemented with parent DataFrame.

    """

    @property
    @abstractmethod
    def metaframe(self):
        """Return MetaFrame instance."""

    @property
    def metaarray(self):
        """Return metaarray instance."""
        return self.attrs['metaarray']

    def update_attrs(self, attrs=None):
        """Extend Frame.update_attrs with metaarray instance."""
        if attrs is not None:
            self.attrs |= attrs
        self.generate_attribute('metaarray')

    @abstractmethod
    def generate_attribute(self, attribute):
        """Generate meta* attributes. Store in self.attrs."""

    def validate_metadata(self):
        """Extend Frame.validate to validate metaarray."""
        unset = [attr not in self.metaframe.columns
                 for attr in self.metaarray.array]
        if np.array(unset).any():
            raise IndexError(
                f'metaarray attributes {np.array(self.metaarray.array)[unset]}'
                f' already set in metaframe.required {self.metaframe.required}'
                f'or metaframe.additional {self.metaframe.additional}')

    def __repr__(self):
        """Extend pandas.DataFrame.__repr__."""
        self.reload_frame()
        return pandas.DataFrame.__repr__(self)

    def reload_frame(self):
        """Transfer data from metaarray.data to frame."""
        for key in self.metaarray.array:
            self._update_frame(key)

    def _update_array(self, key, value=None):
        if value is None:
            value = pandas.DataFrame.__getattr__(self, key).to_numpy()
        with self._setarray(key):
            self.metaarray.data[key] = value

    @contextmanager
    def _setarray(self, key):
        yield
        self.metaarray.update_array[key] = False

    def _update_frame(self, key):
        if self.metaarray.update_frame[key]:
            with self._setframe(key):
                pandas.DataFrame.__setitem__(self, key,
                                             self.metaarray.data[key])

    @contextmanager
    def _setframe(self, key):
        yield
        self.metaarray.update_frame[key] = False

    def __getattr__(self, key):
        """Extend pandas.DataFrame.__getattr__. (frame.*)."""
        if key in self.metaarray.array:
            if self.metaarray.update_array[key]:
                self._update_array(key)
            return self.metaarray.data[key]
        return pandas.DataFrame.__getattr__(self, key)

    def __setattr__(self, key, value):
        """Extend pandas.DataFrame.__setattr__ (frame.* = *).."""
        self.metaarray.check_lock(key)
        if key in self.metaarray.array:
            self._update_array(key, value)
            self.metaarray.update_frame[key] = True
            return None
        return pandas.DataFrame.__setattr__(self, key, value)

    def __getitem__(self, key):
        """Extend pandas.DataFrame.__getitem__. (frame.['*'])."""
        if key in self.metaarray.array:
            self._update_frame(key)
        return pandas.DataFrame.__getitem__(self, key)

    def _get_value(self, index, col, takeable=False):
        """Extend pandas.DataFrame._get_value. (frame.loc[i, '*'])."""
        if col in self.metaarray.array:
            self._update_frame(col)
        return pandas.DataFrame._get_value(self, index, col, takeable)

    def __setitem__(self, key, value):
        """Extend pandas.DataFrame.__setitem__. (frame.['*'] = *)."""
        self.metaarray.check_lock(key)
        if key in self.metaarray.array:
            self._update_array(key, value)
        pandas.DataFrame.__setitem__(self, key, value)
