"""Build interaction matrix for a set of poloidal points."""

from dataclasses import dataclass
from typing import ClassVar

import numpy as np

from nova.biot.biotframe import Target
from nova.biot.operate import Operate
from nova.biot.solve import Solve
from nova.graphics.plot import Plot


@dataclass
class Point(Plot, Operate):
    """Compute interaction for a series of discrete points."""

    point_labels: ClassVar[dict[int, str]] = {2: "xz", 3: "xyz"}

    def extract(self, points: np.ndarray) -> dict[str, np.ndarray]:
        """Return point target dict."""
        match points.shape[-1]:
            case 2 | 3 as ndim:
                point_vector = points.reshape(-1, ndim).T
                return dict(zip(self.point_labels[ndim], point_vector))
            case _:
                raise ValueError(
                    f"Shape of point array {points.shape} must have trailing dimension "
                    "length 2 or 3."
                )

    def solve(self, points: np.ndarray):
        """Solve Biot interaction at points."""
        target = Target(self.extract(points), label="Point")
        self.data = Solve(
            self.subframe,
            target,
            reduce=[True, False],
            attrs=self.attrs,
            name=self.name,
        ).data
        for coord in "xyz":
            self.data.coords[coord] = target[coord]
        super().post_solve()

    def plot(self, axes=None, **kwargs):
        """Plot points."""
        self.axes = axes
        kwargs = dict(marker="o", linestyle="", color="C1") | kwargs
        self.axes.plot(self.data.coords["x"], self.data.coords["z"], **kwargs)
