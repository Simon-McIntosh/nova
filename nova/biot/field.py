"""Solve maximum field on coil perimiter."""

from dataclasses import dataclass, field

import numpy as np
from scipy.interpolate import interp1d

from nova.biot.biotframe import BiotFrame
from nova.biot.operate import Operate
from nova.biot.plot import Plot1D
from nova.biot.solve import Solve
from nova.graphics.plot import Plot


@dataclass
class Sample(Plot):
    """Sample boundary."""

    boundary: np.ndarray
    delta: int | float = 0
    closed: bool = False
    interp: dict[str, interp1d] = field(init=False, repr=False, default_factory=dict)
    data: dict[str, np.ndarray] = field(init=False, repr=False, default_factory=dict)

    def __post_init__(self):
        """Store segment coordinates and build boundary interpolators."""
        self.check()
        self.build()
        self.concatenate()

    def __getitem__(self, attr: str):
        """Return item from data."""
        return self.data[attr]

    def __len__(self):
        """Return sample length."""
        return len(self.data.get("radius", []))

    @property
    def number(self):
        """Return segment number."""
        return len(self.boundary) - 1

    def check(self):
        """Perform input sanity checks."""
        if self.number <= 0:
            raise IndexError(
                f"boundary length {len(self.boundary)} " "must be greater than 1"
            )
        if self.closed and not np.allclose(self.boundary[0], self.boundary[-1]):
            raise ValueError("boundary does not form closed loop")

    def build(self):
        """Build segment interpolators for delta != 0."""
        if self.delta == 0:
            return
        length = np.sqrt(
            np.diff(self.boundary[:, 0]) ** 2 + np.diff(self.boundary[:, 1]) ** 2
        )
        self.data["length"] = np.append(0, np.cumsum(length))
        for i, attr in enumerate(["radius", "height"]):
            self.interp[attr] = interp1d(self["length"], self.boundary[:, i])
        self.data["node_number"] = self.node_number

    @property
    def node_number(self):
        """Calculate node number for each segment."""
        match self.delta:
            case 0:
                return np.ones(self.number)
            case int() if self.delta < 0:
                return -self.delta * np.ones(self.number, dtype=int)
            case int() | float() if self.delta > 0:
                return np.array(
                    [
                        np.max([np.diff(self["length"][i : i + 2])[0] / self.delta, 1])
                        for i in range(self.number)
                    ],
                    dtype=int,
                )
            case _:
                raise TypeError(f"invalid delta {self.delta}")

    def concatenate(self):
        """Concatenate interpolated boundary segments."""
        if self.delta == 0:
            for i, attr in enumerate(["radius", "height"]):
                self.data[attr] = self.boundary[:, i][:-1]
            return
        for attr in ["radius", "height"]:
            segments = [
                self.interp[attr](
                    np.linspace(
                        self["length"][i],
                        self["length"][i + 1],
                        self["node_number"][i],
                        endpoint=False,
                    )
                )
                for i in range(self.number)
            ]
            self.data[attr] = np.concatenate(segments).ravel()

    def plot(self, axes=None):
        """Plot boundary and interpolant nodes."""
        self.get_axes("2d", axes=axes)
        self.axes.plot(*self.boundary.T, "C2o", ms=4)
        self.axes.plot(self["radius"], self["height"], "C1.", ms=4)


@dataclass
class Field(Plot1D, Operate):
    """
    Compute maximum field around coil perimeter.

    Parameters
    ----------
    nfield : int | -float, optional
        Boundary probe resoultion. The default is 1.

            - 0: no boundary contour probes
            - < 0: probe segment resolution
            - int >= 0: probe segment number

    """

    nfield: int | float = 1
    frame_index: str = "coil"
    target: BiotFrame = field(init=False, repr=False)

    def __len__(self):
        """Return field probe number."""
        return len(self.data.get("x", []))

    def solve(self, number=None):
        """Extract boundary and solve magnetic field around coil perimeter."""
        with self.solve_biot(number) as number:
            if number is not None:
                self.target = BiotFrame()
                for name in self.Loc[self.frame_index, :].index:
                    polyframe = self.frame.loc[name, "poly"]
                    if polyframe.poly.boundary.is_ring:
                        sample = Sample(polyframe.boundary, delta=-number)
                        self.target.insert(
                            sample["radius"],
                            sample["height"],
                            link=True,
                            label=name,
                            delim="_",
                        )
                self.data = Solve(
                    self.subframe,
                    self.target,
                    reduce=[True, False],
                    turns=[True, False],
                    attrs=["Br", "Bz"],
                    name=self.name,
                ).data
                # insert grid data
                self.data.coords["index"] = self.target.biotreduce.indices
                self.data.coords["x"] = "target", self.target.x
                self.data.coords["z"] = "target", self.target.z

    @property
    def coil_name(self):
        """Return target coil name."""
        return self.data.target[self.data.index].data

    @property
    def points(self):
        """Return point array."""
        return np.array([self.data.x, self.data.z]).T

    def plot(self, axes=None, **kwargs):
        """Plot points."""
        self.get_axes("2d", axes=axes)
        kwargs = dict(marker="o", linestyle="", color="C1", ms=4) | kwargs
        self.axes.plot(self.data.coords["x"], self.data.coords["z"], **kwargs)
