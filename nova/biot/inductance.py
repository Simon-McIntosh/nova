"""Build interaction matrix toroidal loops."""

from collections.abc import Mapping
from dataclasses import dataclass, field

from nova.biot.operate import Operate
from nova.biot.solve import Solve
from nova.biot.target import (
    DEFAULT_TARGET_QUADRATURE_POLICY,
    TargetQuadraturePolicy,
    linked_flux_target,
)
from nova.graphics.plot import Plot


@dataclass
class Inductance(Plot, Operate):
    """Compute self interaction."""

    target_policy: TargetQuadraturePolicy | Mapping | str = (
        DEFAULT_TARGET_QUADRATURE_POLICY
    )
    _configured_target_policy: str = field(init=False, repr=False, default="")

    def __post_init__(self):
        """Freeze target integration to the owning CoilSet's cache identity."""
        self._configured_target_policy = TargetQuadraturePolicy.resolve(
            self.target_policy
        ).key
        self.target_policy = self._configured_target_policy
        if hasattr(super(), "__post_init__"):
            super().__post_init__()

    def solve(self, number=None):
        """Solve Biot interaction across subframe."""
        with self.solve_biot(number) as number:
            if number is not None:
                policy = TargetQuadraturePolicy.resolve(self.target_policy).key
                if policy != self._configured_target_policy:
                    raise ValueError(
                        "inductance target policy is fixed by its CoilSet constructor"
                    )
                quadrature = linked_flux_target(*self.frames, policy=policy)
                self.target = quadrature.logical
                self.data = Solve(
                    self.subframe,
                    quadrature.nodes,
                    turns=[True, True],
                    reduce=[True, True],
                    attrs=self.attrs,
                    name=self.name,
                    target_quadrature=quadrature,
                ).data
                self.data.attrs["target_quadrature_policy"] = quadrature.policy.key

    def plot(self, axes=None, **kwargs):
        """Plot points."""
        self.axes = axes
        kwargs = dict(marker=".", linestyle="") | kwargs
        self.axes.plot(self.subframe["x"], self.subframe["z"], **kwargs)
        kwargs = dict(marker=".", linestyle="", color="C1") | kwargs
        self.axes.plot(self.target["x"], self.target["z"], **kwargs)
