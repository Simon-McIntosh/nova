"""Solve intergral coil forces."""

from collections.abc import Mapping
from dataclasses import dataclass, field, replace

import numpy as np

from nova.biot.biotframe import BiotFrame
from nova.biot.operate import Operate
from nova.biot.plot import Plot1D
from nova.biot.solve import Solve
from nova.biot.target import (
    DEFAULT_FORCE_TARGET_POLICY,
    ForceTargetPolicy,
    section_force_target,
)
from nova.frame.polygrid import PolyTarget


@dataclass
class Force(Plot1D, Operate):
    """
    Compute coil force interaction matricies.

    The force a conductor carries is an area integral over the material its
    current occupies, so the target rule is a quadrature rule for that integral.
    ``target_policy`` selects it: the shipped ``positive_material`` rule averages
    the integrand over the Gauss fan of :mod:`nova.biot.sectionaverage`, and
    ``subdivision`` samples it at the centroid of every cell of an exact tiling
    instead. A target selection holding plasma material integrates on the tiling
    whichever rule is named -- see :meth:`material_rule`.

    Parameters
    ----------
    nforce : int | -float, optional
        Coil force segment resoultion.

            - < 0: coil segment resolution
            - int >= 0: coil segment number

        This sets the subdivision rule's resolution. Under the shipped rule the
        quadrature order sets it and the segment number survives only as the
        solve trigger.

    """

    reduce: bool = True
    attrs: list[str] = field(default_factory=lambda: ["Fr", "Fz", "Fc"])
    frame_index: str = "coil"
    target_policy: ForceTargetPolicy | Mapping | str = DEFAULT_FORCE_TARGET_POLICY
    _configured_target_policy: str = field(init=False, repr=False, default="")
    target: BiotFrame = field(init=False, repr=False)

    def __post_init__(self):
        """Freeze the target rule to the owning CoilSet's cache identity."""
        self._configured_target_policy = ForceTargetPolicy.resolve(
            self.target_policy
        ).key
        self.target_policy = self._configured_target_policy
        if hasattr(super(), "__post_init__"):
            super().__post_init__()

    def __len__(self):
        """Return force patch number."""
        return len(self.data.get("x", []))

    def solve(self, number=None):
        """Extract boundary and solve magnetic field around coil perimeter."""
        with self.solve_biot(number) as number:
            if number is not None:
                policy = ForceTargetPolicy.resolve(self.target_policy)
                if policy.key != self._configured_target_policy:
                    raise ValueError(
                        "force target policy is fixed by its CoilSet constructor"
                    )
                policy = self.material_rule(policy)
                if policy.rule == "subdivision":
                    self.solve_subdivision(number)
                else:
                    self.solve_section_quadrature(policy)
                self.data.attrs["force_target_policy"] = policy.key

    @property
    def target_carries_plasma(self) -> bool:
        """Return whether the selected force targets hold plasma material."""
        return bool(
            np.any(np.asarray(self.Loc[self.frame_index, "plasma"], dtype=bool))
        )

    def material_rule(self, policy: ForceTargetPolicy) -> ForceTargetPolicy:
        """Return the target rule the selected material admits.

        The fan spreads one uniform turn density over a whole section, which is
        what a conductor's turns are. A plasma allocates its turns cell by cell,
        and its force is the sum of cell forces each weighted by its own turn
        number, so the section mean the fan returns is not the force the plasma
        carries. That is a property of the material rather than a limit of the
        rule, so a selection holding plasma integrates on the tiling whichever
        rule was named, and the whole selection follows it because one operator
        carries one rule. The rule that ran is what the solved data records.
        """
        if policy.rule == "subdivision" or not self.target_carries_plasma:
            return policy
        return replace(policy, rule="subdivision")

    def solve_subdivision(self, number):
        """Sample the force density at the centroid of every cell of a tiling."""
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

    def solve_section_quadrature(self, policy):
        """Average the force density over positive material with the Gauss fan."""
        if not self.reduce:
            raise ValueError(
                "the positive material force rule integrates whole sections and "
                "cannot resolve force within one; a map inside a section is a "
                'different quantity, so name ForceTargetPolicy(rule="subdivision")'
            )
        quadrature = section_force_target(
            self.frame, self.Loc[self.frame_index, :].index, policy
        )
        self.target = quadrature.logical
        self.data = Solve(
            self.subframe,
            quadrature.nodes,
            reduce=[True, True],
            turns=[True, True],
            attrs=self.attrs,
            name=self.name,
            target_quadrature=quadrature,
        ).data
        self.data.coords["index"] = (
            "target",
            self.Loc[self.frame_index, "subref"],
        )
        self.data.coords["xo"] = "target", self.Loc[self.frame_index, "x"]
        self.data.coords["zo"] = "target", self.Loc[self.frame_index, "z"]
        self.data.coords["x"] = quadrature.nodes.x
        self.data.coords["z"] = quadrature.nodes.z

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
