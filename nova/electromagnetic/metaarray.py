
from dataclasses import dataclass, field
from typing import Iterable, Union

import pandas

from nova.electromagnetic.metadata import MetaData

# pylint:disable=unsubscriptable-object


@dataclass
class MetaArray(MetaData):
    """Manage Frame metadata - accessed via Frame['attrs']."""

    array: list[str] = field(default_factory=lambda: [])
    data: dict[str, Iterable[Union[str, int, float]]] = \
        field(default_factory=dict)
    update_array: dict[str, bool] = field(default_factory=dict)
    update_frame: dict[str, bool] = field(default_factory=dict)

    _internal = ['data', 'update_array', 'update_frame']

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
