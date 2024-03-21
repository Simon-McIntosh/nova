"""Solve intergral coil forces."""

from dataclasses import dataclass, field

import numpy as np


from nova.biot.biotframe import BiotFrame
from nova.biot.biotframe import Target
from nova.biot.operate import Operate
from nova.biot.solve import Solve
from nova.graphics.plot import Plot2D


@dataclass
class Overlap(Plot2D, Operate):
    """
    Compute error field overlap external forcing.

    Parameters
    ----------
    nloop : int, optional
        Toroidal resolution. The default is 120.

    """

    attrs: list[str] = field(init=False, default_factory=lambda: ["Br", "Bz"])
    target: BiotFrame = field(init=False, repr=False)

    def __len__(self):
        """Return force patch number."""
        return len(self.data.get("x", []))

    def solve(self, points: np.ndarray, number=None):
        """Extract boundary and solve magnetic field around coil perimeter."""
        with self.solve_biot(number) as number:
            if number is not None:

                print(points.shape)

                Target({"x": points[..., 0], "z": points[..., 1]})
                Solve()
                """

                target = Target()
                target.insert()
                points = np._r[points[-1:], points, points[:1]]
                print(points.shape)
                self.target = PolyTarget(
                    *self.frames, index=self.frame_index, delta=-number
                ).target
                self.data = Solve(
                    self.subframe,
                    self.target,
                    reduce=[True, False],
                    turns=[True, False],
                    attrs=self.attrs,
                    name=self.name,
                ).data
                self.data.coords["index"] = (
                    "target",
                    self.Loc[self.frame_index, "subref"],
                )
                self.data.coords["xo"] = "target", self.Loc[self.frame_index, "x"]
                self.data.coords["zo"] = "target", self.Loc[self.frame_index, "z"]
                self.data.coords["x"] = self.target.x
                self.data.coords["z"] = self.target.z
                """

    @property
    def coil_name(self):
        """Return target coil names."""
        return self.data.target.data

    def plot_points(self, axes=None, **kwargs):
        """Plot force intergration points."""
        self.get_axes("2d", axes=axes)
        kwargs = dict(marker="o", linestyle="", color="C2", ms=4) | kwargs
        self.axes.plot(self.data.coords["x"], self.data.coords["z"], **kwargs)

    '''
    def bar(self, attr: str, index=slice(None), axes=None, **kwargs):
        """Plot per-coil force component."""
        self.get_axes("1d", axes)
        if isinstance(index, str):
            index = [name in self.loc[index, :].index for name in self.coil_name]
        names = self.coil_name[index]
        self.axes.bar(names, 1e-6 * getattr(self, attr)[index], **kwargs)
        self.axes.set_xticklabels(names, rotation=90, ha="center")
        label = {"fr": "radial", "fz": "vertical"}
        self.axes.set_ylabel(f"{label[attr]} force MN")
    '''

    def plot(self, scale=1, norm=None, axes=None, **kwargs):
        """Plot force vectors and intergration points."""
        self.get_axes("2d", axes)
        vector = np.c_[self.fr, self.fz]
        if norm is None:
            norm = np.max(np.linalg.norm(vector, axis=1))
        length = scale * vector / norm
        patch = self.mpl["patches"].FancyArrowPatch
        if self.reduce:
            tail = np.c_[self.data.xo, self.data.zo]
        else:
            tail = np.c_[self.data.x, self.data.z]
        arrows = [
            patch(
                (x, z),
                (x + dx, z + dz),
                mutation_scale=1,
                arrowstyle="simple,head_length=0.4, head_width=0.3," " tail_width=0.1",
                shrinkA=0,
                shrinkB=0,
            )
            for x, z, dx, dz in zip(tail[:, 0], tail[:, 1], length[:, 0], length[:, 1])
        ]
        collections = self.mpl.collections.PatchCollection(
            arrows, facecolor="black", edgecolor="darkgray"
        )
        self.axes.add_collection(collections)
        return norm


if __name__ == "__main__":

    from nova.frame.coilset import CoilSet

    coilset = CoilSet(noverlap=120)

    coilset.coil.insert(3, 4, 0.05, 0.05, ifttt=False, segment="cylinder", Ic=1e3)

    points = np.stack(
        np.meshgrid(np.linspace(3, 5, 21), np.linspace(-2, 2), 32, indexing="ij"),
        axis=-1,
    )
    coilset.overlap.solve(points)
