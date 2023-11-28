"""Manage source and target frames."""
from dataclasses import dataclass, field
from functools import cached_property

import numpy as np

from nova.biot.biotframe import Source, Target
from nova.graphics.plot import Plot


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
class GroupSet(Plot):
    """
    Construct Biot source/target biot frames.

    Parameters
    ----------
    source: Source
        Field source

    target: Target
        Calculation target

    turns: list[bool, bool]
        Multiply columns / rows by turn number (source, target) if True.

    reduce: list[bool, bool]
        Apply linked turn reduction (source, target) if True.

    """

    source: Source = field(repr=False, default_factory=Source)
    target: Target = field(repr=False, default_factory=Target)
    turns: list[bool] = field(default_factory=lambda: [True, False])
    reduce: list[bool] = field(default_factory=lambda: [True, True])
    coordinate_axes: np.ndarray = field(init=False, repr=False)
    coordinate_origin: np.ndarray = field(init=False, repr=False)

    def __post_init__(self):
        """Format source and target biot frames."""
        if not isinstance(self.source, Source):
            self.source = Source(self.source)
        if not isinstance(self.target, Target):
            self.target = Target(self.target, available=[])
        self.set_flags()
        self.assemble()

    def __call__(self, frame: str, attr: str):
        """Return attribute matrix, shape(target, source) from CoordLocIndexer."""
        return self.loc[frame, attr]

    def __len__(self):
        """Return interaction length."""
        return np.prod(self.shape)

    @property
    def shape(self):
        """Return interaction matrix shape."""
        return len(self.target), len(self.source)

    def set_flags(self):
        """Set turn and reduction flags on source and target biot frames."""
        if isinstance(self.turns, bool):
            self.turns = [self.turns, self.turns]
        if isinstance(self.reduce, bool):
            self.reduce = [self.reduce, self.reduce]
        self.source.turns = self.turns[0]
        self.target.turns = self.turns[1]
        self.source.reduce = self.reduce[0]
        self.target.reduce = self.reduce[1]

    def assemble(self):
        """Assemble GroupSet."""
        self.set_shape()
        self.build_index()
        self.build_transform()

    def set_shape(self):
        """Set source and target shapes."""
        self.source.set_target(len(self.target))
        self.target.set_source(len(self.source))

    def build_index(self):
        """Build index. Product of source and target biot frames."""
        self.index = range(len(self))
        # self.index = ['_'.join(label) for label
        #               in itertools.product(self.source.index,
        #                                    self.target.index)]

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

    def plot(self, axes=None):
        """Plot source and target markers."""
        self.axes = axes
        self.source.plot(
            "x", "z", "scatter", ax=self.axes, color="C1", marker="o", label="source"
        )
        self.target.plot(
            "x", "z", "scatter", ax=self.axes, color="C2", marker=".", label="target"
        )
        self.axes.axis("equal")
        self.axes.axis("off")


if __name__ == "__main__":
    from nova.geometry.polyline import PolyLine

    points = np.array([[-2, 0, 0], [-1, 0, 0], [0, 1, 0], [1, 0, 0]], float)
    polyline = PolyLine(points, minimum_arc_nodes=3)
    source = Source(polyline.path_geometry)
    target = Target({"x": np.linspace(5, 7.5, 10), "z": 0.5})
    groupset = GroupSet(source, target)
