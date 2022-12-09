"""Biot-Savart calculation base class."""
from dataclasses import dataclass, field
from importlib import import_module
from typing import ClassVar

import numpy as np

from nova.biot.biotset import BiotSet


@dataclass
class BiotMatrix(BiotSet):
    """Compute Biot interaction matricies."""

    data: dict[str, np.ndarray] = field(init=False, repr=False,
                                        default_factory=dict)
    attrs: dict[str, str] = field(default_factory=dict)

    mu_0: ClassVar[float] = import_module('scipy.constants').mu_0

    def __post_init__(self):
        """Initialize input data."""
        super().__post_init__()
        for attr in self.attrs:
            self.data[attr] = self.get_frame(attr)(self.attrs[attr])

    def __getitem__(self, attr):
        """Return attributes from data."""
        return self.data[attr]

    def get_frame(self, attr: str):
        """Return source or target frame associated with attr."""
        if attr in 'rxyz':
            return self.target
        return self.source

    def compute(self, attr: str):
        """
        Return full and unit plasma interaction matrices.

        Extract plasma (unit) interaction from full matrix.
        Multiply by source and target turns.
        Apply reduction summations.

        """
        matrix = getattr(self, attr)
        if self.target.turns:
            matrix *= self.target('nturn')
        #if isinstance(matrix, da.Array):
        #    matrix = matrix.compute()
        plasma = matrix[:, self.source.plasma]
        if self.source.turns:
            matrix *= self.source('nturn')
        # reduce
        if self.source.reduce and self.source.biotreduce.reduce:
            matrix = np.add.reduceat(
                matrix, self.source.biotreduce.indices,
                axis=1, dtype=np.float128)
        if self.target.reduce and self.target.biotreduce.reduce:
            matrix = np.add.reduceat(
                matrix, self.target.biotreduce.indices, axis=0)
            plasma = np.add.reduceat(
                plasma, self.target.biotreduce.indices, axis=0)
        # link source
        source_link = self.source.biotreduce.link
        if self.source.reduce and len(source_link) > 0:
            for link in source_link:  # sum linked columns
                ref, factor = source_link[link]
                matrix[:, ref] += factor * matrix[:, link]
            matrix = np.delete(matrix, list(source_link), 1)
        # link target
        target_link = self.target.biotreduce.link
        if self.target.reduce and len(target_link) > 0:
            for link in target_link:  # sum linked columns
                ref, factor = target_link[link]
                matrix[ref, :] += factor * matrix[link, :]
                plasma[ref, :] += factor * plasma[link, :]
            matrix = np.delete(matrix, list(target_link), 0)
            plasma = np.delete(plasma, list(target_link), 0)
        return matrix, plasma
