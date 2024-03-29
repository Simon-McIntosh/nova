"""Biot-Savart calculation base class."""

from dataclasses import dataclass, field
from functools import cached_property
from importlib import import_module
from typing import ClassVar

import numpy as np

from nova.biot.biotframe import Source, Target
from nova.biot.groupset import GroupSet


@dataclass
class CoordLocIndexer:
    """Coordinate system indexer base class."""

    source: Source = field(repr=False)
    target: Target = field(repr=False)
    coordinate_axes: np.ndarray = field(
        repr=False, default_factory=lambda: np.array([])
    )
    coordinate_origin: np.ndarray = field(
        repr=False, default_factory=lambda: np.array([])
    )
    data: dict[str, dict] = field(
        repr=False, default_factory=lambda: {"source": {}, "target": {}}
    )

    def __getitem__(self, key):
        """Return stacked point array transformed to local coordinates."""
        match key:
            case (str(frame), str(attr)) if frame in ["source", "target"]:
                return self._getdata(frame, attr)
            case _:
                raise KeyError(
                    f"malformed key {key}, require " "[source | target, attr]"
                )

    def coordinate_list(self, attr: str) -> list[str]:
        """Return coordinate list."""
        primary_coordinate = next(coord for coord in "xyz" if coord in attr)
        return [attr.replace(primary_coordinate, coord) for coord in "xyz"]

    def _getdata(self, frame: str, attr: str) -> np.ndarray:
        """Return coordinate system data."""
        try:
            return self.data[frame][attr]
        except KeyError:
            coords = self.coordinate_list(attr)
            points = self._stack(frame, coords)
            self.data[frame] |= dict(zip(coords, (points[..., i] for i in range(3))))
            return self.data[frame][attr]

    def _stack(self, frame: str, coords: list[str]) -> np.ndarray:
        """Return stacked point array in local coordinate system."""
        points = getattr(self, frame).stack(*coords)
        return self.to_local(points)

    def rotate(self, points: np.ndarray, coordinate_frame: str):
        """Rotate points to coordinate frame."""
        match coordinate_frame:
            case "to_local":
                return np.einsum("...i,...ij->...j", points, self.coordinate_axes)
            case "to_global":
                return np.einsum("...i,...ji->...j", points, self.coordinate_axes)
            case _:
                raise NotImplementedError(
                    f"coordinate rotation to {coordinate_frame} "
                    "frame not implemented"
                )

    def to_local(self, points):
        """Return 3d point array (target, source, 3) mapped to local coordinates."""
        return self.rotate(points - self.coordinate_origin, "to_local")

    def to_global(self, points):
        """Return 3d point array (target, source, 3) mapped to global coordinates."""
        return self.rotate(points, "to_global") + self.coordinate_origin


@dataclass
class Matrix(GroupSet):
    """Compute Biot interaction matricies."""

    data: dict[str, np.ndarray] = field(init=False, repr=False, default_factory=dict)
    attrs: dict[str, str] = field(default_factory=dict)
    coordinate_axes: np.ndarray = field(init=False, repr=False)
    coordinate_origin: np.ndarray = field(init=False, repr=False)

    axisymmetric: ClassVar[bool] = True
    mu_0: ClassVar[float] = import_module("scipy.constants").mu_0

    def __post_init__(self):
        """Initialize input data."""
        super().__post_init__()
        self.build_transform()
        for attr in self.attrs:
            self.data[attr] = self.get_frame(attr)(self.attrs[attr])

    def __call__(self, frame: str, attr: str):
        """Return attribute matrix, shape(target, source) from local CoordLocIndexer."""
        return self.loc[frame, attr]

    def __getitem__(self, attr):
        """Return attributes from data."""
        return self.data[attr]

    def build_transform(self):
        """Build global to local coordinate transformation matrix (target, source)."""
        self.coordinate_axes = np.tile(
            self.source.space.coordinate_axes, reps=(self.shape[0], 1, 1, 1)
        )
        self.coordinate_origin = np.tile(
            self.source.space.origin, reps=(self.shape[0], 1, 1)
        )

    @cached_property
    def loc(self):
        """Return local coordinate stack indexer."""
        return type("coord_loc_indexer", (CoordLocIndexer,), {})(
            self.source,
            self.target,
            self.coordinate_axes,
            self.coordinate_origin,
        )

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
            return np.stack(
                [
                    -self.Aphi * np.sin(self.phi),
                    self.Aphi * np.cos(self.phi),
                    np.zeros_like(self.Aphi),
                ],
                axis=-1,
            )
        return self._vector("A")

    @cached_property
    def Bvector(self):
        """Return stacked global magnetic field vector, axis=-1."""
        if self.axisymmetric:
            return np.stack(
                [self.Br * np.cos(self.phi), self.Br * np.sin(self.phi), self.Bz],
                axis=-1,
            )
        return self.mu_0 * self._vector("B")

    @property
    def Ax(self):
        """Return x component of magnetic vector potential."""
        return self.Avector[..., 0]

    @property
    def Ay(self):
        """Return y component of magnetic vector potential."""
        return self.Avector[..., 1]

    @property
    def Az(self):
        """Return z component of magnetic vector potential."""
        return self.Avector[..., 2]

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
    def Bphi(self):
        """Return toroidal field array."""
        if self.axisymmetric:
            return np.zeros(self.shape, float)
        return -self.Bx * np.sin(self.phi) + self.By * np.cos(self.phi)

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
        matrix = getattr(self, attr).copy()
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
            matrix = np.delete(matrix, list(target_link), 0)
            target_plasma = np.delete(target_plasma, list(target_link), 0)
        return matrix, target_plasma, plasma_source, plasma_plasma
