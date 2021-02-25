
from contextlib import contextmanager
from dataclasses import dataclass, field

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