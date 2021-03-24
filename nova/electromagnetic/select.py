"""Manage frame index."""
from dataclasses import dataclass, field

import numpy as np

from nova.electromagnetic.metamethod import MetaMethod
from nova.electromagnetic.dataframe import DataFrame


@dataclass
class Select(MetaMethod):
    """Manage dependant frame energization parameters."""

    frame: DataFrame = field(repr=False)
    required: list[str] = field(default_factory=lambda: [])
    additional: list[str] = field(default_factory=lambda: [])
    match: dict[str, str] = field(default_factory=lambda: {
        'active': 'active', 'passive': 'active',
        'plasma': 'plasma', 'coil': 'plasma'
        })
    exclude: list[str] = field(default_factory=lambda: [
        'feedback', 'plasma'])

    def __post_init__(self):
        """Extend additional with unique values extracted from match."""
        self.additional.extend(
            [attr for attr in list(dict.fromkeys(self.match.values()))
             if attr not in self.additional])
        super().__post_init__()

    def __call__(self, label):
        return self.select(label)

    def select(self, label):
        """Return boolean selection index based on label."""
        if self.match[label] not in self.additional:
            raise IndexError(f'attr {self.match[label]} not specified '
                             f'in {self.additional}')



