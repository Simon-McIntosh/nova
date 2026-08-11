"""Solve intergral coil forces."""

from dataclasses import dataclass, field

import numpy as np

from nova.biot.biotframe import BiotFrame
from nova.biot.operate import Operate
from nova.biot.plot import Plot1D
from nova.biot.solve import Solve
from nova.frame.polygrid import PolyTarget


@dataclass
class Force(Plot1D, Operate):
    """
    Compute coil force interaction matricies.

    Parameters
    ----------
    nforce : int | -float, optional
        Coil force segment resoultion. The default is 500.

            - < 0: coil segment resolution
            - int >= 0: coil segment number

    """

    reduce: bool = True
    attrs: list[str] = field(default_factory=lambda: ["Fr", "Fz", "Fc"])
    frame_index: str = "coil"
    target: BiotFrame = field(init=False, repr=False)

    def __len__(self):
        """Return force patch number."""
        return len(self.data.get("x", []))

    def solve(self, number=None):
        """Extract boundary and solve magnetic field around coil perimeter."""
        with self.solve_biot(number) as number:
            if number is not None:
                self.target = PolyTarget(
                    *self.frames, index=self.frame_index, delta=-number
                ).target
                self.bind_moment_arm()
                self.data = Solve(
                    self.subframe,
                    self.target,
                    reduce=[True, self.reduce],
                    turns=[True, True],
                    attrs=self.attrs,
                    name=self.name,
                ).data
                # insert grid data
                if self.reduce:
                    self.data.coords["index"] = (
                        "target",
                        self.Loc[self.frame_index, "subref"],
                    )
                    self.data.coords["xo"] = "target", self.Loc[self.frame_index, "x"]
                    self.data.coords["zo"] = "target", self.Loc[self.frame_index, "z"]
                    self.data.coords["x"] = self.target.x
                    self.data.coords["z"] = self.target.z
                else:
                    self.data.coords["index"] = (
                        "target",
                        self.loc[self.frame_index, "subref"],
                    )
                    self.data.coords["x"] = "target", self.target.x
                    self.data.coords["z"] = "target", self.target.z

    def bind_moment_arm(self):
        """Measure the moment arm against the conductor rather than the cell.

        The tiling hands every cell its own width and height, and a first moment
        divided by them is divided by a length that shrinks as the tiling refines,
        so it grows without bound instead of converging. The arm a crushing moment
        turns about is the conductor's own extent, which is the same body whose
        centre the target already carries.
        """
        label = np.asarray(self.target.index, dtype=object)
        link = np.asarray(self.target["link"], dtype=object)
        parent = np.where(link == "", label, link)
        extent = {
            str(name): (width, height)
            for name, width, height in zip(
                np.asarray(self.frame.index, dtype=object),
                np.asarray(self.frame["dx"], dtype=float),
                np.asarray(self.frame["dz"], dtype=float),
            )
        }
        self.target["dx"] = np.array([extent[str(name)][0] for name in parent])
        self.target["dz"] = np.array([extent[str(name)][1] for name in parent])

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
                arrowstyle="simple,head_length=0.4, head_width=0.3, tail_width=0.1",
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
