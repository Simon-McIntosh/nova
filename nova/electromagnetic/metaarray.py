"""Extend pandas.DataFrame to manage fast access attributes."""
from dataclasses import dataclass, field, InitVar
from typing import Iterable, Union

import pandas

from nova.electromagnetic.metadata import MetaData

# pylint:disable=unsubscriptable-object


@dataclass
class MetaArray(MetaData):
    """Manage Frame metadata - accessed via Frame['attrs']."""

    index: InitVar[list[str]] = field(default=None)
    array: list[str] = field(default_factory=lambda: [])
    data: dict[str, Iterable[Union[str, int, float]]] = field(init=False)

    _internal = ['index', 'data']

    def __post_init__(self, index):
        """Init update flags."""
        self.index = index
        self.data = {}

    def __repr__(self):
        """Return __repr__."""
        return pandas.DataFrame(self.data, index=self.index,
                                columns=self.array).__repr__()
