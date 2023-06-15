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
    target: BiotFrame = field(init=False, repr=False)

    def __len__(self):
        """Return force patch number."""
        return len(self.data.get("x", []))

    def solve(self, number=None):
        """Extract boundary and solve magnetic field around coil perimeter."""
        with self.solve_biot(number) as number:
            if number is not None:
                self.target = PolyTarget(
                    *self.frames, index="coil", delta=-number
                ).target
                self.data = Solve(
                    self.subframe,
                    self.target,
                    reduce=[True, self.reduce],
                    turns=[True, True],
                    attrs=self.attrs,
                    name=self.name,
                ).data
                # insert grid data
                self.data.coords["index"] = "target", self.Loc["coil", "subref"]
                if self.reduce:
                    self.data.coords["xo"] = "target", self.Loc["coil", "x"]
                    self.data.coords["zo"] = "target", self.Loc["coil", "z"]
                    self.data.coords["x"] = self.target.x
                    self.data.coords["z"] = self.target.z
                else:
                    self.data.coords["x"] = "target", self.target.x
                    self.data.coords["z"] = "target", self.target.z

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
