"""Biot-Savart calculation base class."""
from dataclasses import dataclass, field
from functools import cached_property
from importlib import import_module
from typing import ClassVar

import numpy as np

from nova.biot.groupset import GroupSet


@dataclass
class Matrix(GroupSet):
    """Compute Biot interaction matricies."""

    data: dict[str, np.ndarray] = field(init=False, repr=False, default_factory=dict)
    attrs: dict[str, str] = field(default_factory=dict)

    axisymmetric: ClassVar[bool] = True
    mu_0: ClassVar[float] = import_module("scipy.constants").mu_0

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
        if attr in "rxyz":
            return self.target
        return self.source

    @cached_property
    def phi(self):
        """Return target poloidal angle."""
        return np.arctan2(self.target.y, self.target.x)[:, np.newaxis]

    def _Bx_hat(self):
        """Return local intergration coefficent along x-axis."""
        raise NotImplementedError

    def _By_hat(self):
        """Return local intergration coefficent along y-axis."""
        raise NotImplementedError

    def _Bz_hat(self):
        """Return local intergration coefficent along z-axis."""
        raise NotImplementedError

    def _intergrate(self, data):
        """Return intergral property."""
        raise NotImplementedError

    def _vector(self, attr: str):
        """Return global attribute vector stacked along last axis."""
        local = np.stack(
            [self._intergrate(getattr(self, f"_{attr}{coord}_hat")) for coord in "xyz"],
            axis=-1,
        )
        return self.loc.rotate(local, "to_global")

    @cached_property
    def Avector(self):
        """Return global vector potential in cartesian frame."""
        if self.axisymmetric:
            raise NotImplementedError
        return self._vector("A")

    @cached_property
    def Bvector(self):
        """Return stacked global magnetic field vector, axis=-1."""
        if self.axisymmetric:
            return np.stack(
                [self.Br * np.cos(self.phi), self.Br * np.sin(self.phi), self.Bz],
                axis=-1,
            )
        return self._vector("B")

    @property
    def Bx(self):
        """Return x component of magnetic field vector."""
        return self.Bvector[..., 0]

    @property
    def By(self):
        """Return y component of magnetic field vector."""
        return self.Bvector[..., 1]

    @property
    def Bz(self):
        """Return z component of magnetic field vector."""
        if self.axisymmetric:
            raise NotImplementedError
        return self.Bvector[..., 2]

    @property
    def Br(self):
        """Return radial field array."""
        if self.axisymmetric:
            raise NotImplementedError
        return self.Bx * np.cos(self.phi) + self.By * np.sin(self.phi)

    @property
    def Fr(self):
        """Return radial force array."""
        return 2 * np.pi * self.target.x[:, np.newaxis] * self.Bz

    @property
    def Fz(self):
        """Return vertical force array."""
        return -2 * np.pi * self.target.x[:, np.newaxis] * self.Br

    @property
    def Fc(self):
        """Return first moment of vertical (crushing) force."""
        return self.target.delta_z[:, np.newaxis] * self.Fz

    def compute(self, attr: str):
        """
        Return full and unit plasma interaction matrices.

        Extract plasma (unit) interaction from full matrix.
        Multiply by source and target turns.
        Apply reduction summations.

        """
        matrix = getattr(self, attr)
        target_plasma = matrix[:, self.source.plasma]
        plasma_source = matrix[self.target.plasma]
        plasma_plasma = plasma_source[:, self.source.plasma]
        if self.target.turns:
            matrix *= self.target("nturn")
            target_plasma *= self.target("nturn")[:, self.source.plasma]
        if self.source.turns:
            matrix *= self.source("nturn")
            plasma_source *= self.source("nturn")[self.target.plasma]
        if self.source.reduce and self.source.biotreduce.reduce:
            matrix = np.add.reduceat(matrix, self.source.biotreduce.indices, axis=1)
            plasma_source = np.add.reduceat(
                plasma_source, self.source.biotreduce.indices, axis=1
            )
        if self.target.reduce and self.target.biotreduce.reduce:
            matrix = np.add.reduceat(matrix, self.target.biotreduce.indices, axis=0)
            target_plasma = np.add.reduceat(
                target_plasma, self.target.biotreduce.indices, axis=0
            )
        # link source
        source_link = self.source.biotreduce.link
        if self.source.reduce and len(source_link) > 0:
            for link in source_link:  # sum linked columns
                ref, factor = source_link[link]
                matrix[:, ref] += factor * matrix[:, link]
                plasma_source[:, ref] += factor * plasma_source[:, link]
            matrix = np.delete(matrix, list(source_link), 1)
            plasma_source = np.delete(plasma_source, list(source_link), 1)
        # link target
        target_link = self.target.biotreduce.link
        if self.target.reduce and len(target_link) > 0:
            for link in target_link:  # sum linked rows
                ref, factor = target_link[link]
                matrix[ref] += factor * matrix[link]
                target_plasma[ref] += factor * target_plasma[link]
                # plasma[ref, :] += factor * plasma[link]
            matrix = np.delete(matrix, list(target_link), 0)
            target_plasma = np.delete(target_plasma, list(target_link), 0)
        return matrix, target_plasma, plasma_source, plasma_plasma
