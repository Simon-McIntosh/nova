"""Construct coilset with frameset and biot factories."""

from dataclasses import dataclass, field
from typing import ClassVar

from nova.biot.biot import Biot
from nova.control.control import Control
from nova.frame.frame import Frame
from nova.geometry.polygon import Polygon


@dataclass(repr=False)
class CoilSet(Biot, Control, Frame):
    """
    Manage coilset.

    See Frame for insert methods.

    """

    dirname: str = field(default=".nova", repr=False)
    coil_polysection_policy: object = ""
    plasma_polysection_policy: object = ""
    inductance_target_policy: object = ""
    force_target_policy: object = ""
    _configured_route_attrs: dict[str, str] = field(
        init=False, repr=False, default_factory=dict
    )
    _route_restore: bool = field(init=False, repr=False, default=False)

    _route_fields: ClassVar[tuple[str, ...]] = (
        "coil_polysection_policy",
        "plasma_polysection_policy",
        "inductance_target_policy",
        "force_target_policy",
    )

    def __setattr__(self, name, value):
        """Reject source or target route changes after construction."""
        configured = self.__dict__.get("_configured_route_attrs", {})
        if (
            name in self._route_fields
            and configured
            and not self.__dict__.get("_route_restore", False)
        ):
            from nova.biot.polysection import PolySectionPolicy
            from nova.biot.target import ForceTargetPolicy, TargetQuadraturePolicy

            resolver = {
                "inductance_target_policy": TargetQuadraturePolicy,
                "force_target_policy": ForceTargetPolicy,
            }.get(name, PolySectionPolicy)
            canonical = resolver.resolve(value).key
            if canonical != configured[name]:
                raise ValueError(
                    "CoilSet route policies are fixed by its constructor or "
                    "restored cache identity"
                )
            value = canonical
        super().__setattr__(name, value)

    def __post_init__(self):
        """Resolve immutable source and target routes before cache identity."""
        self._resolve_route_attrs()
        super().__post_init__()

    def _resolve_route_attrs(self):
        """Canonicalize route attributes supplied by constructors or storage."""
        self._restore_route_attrs({})

    def _restore_route_attrs(self, attrs):
        """Bind canonical routes while constructing or loading one cache identity."""
        from nova.biot.polysection import PolySectionPolicy
        from nova.biot.target import ForceTargetPolicy, TargetQuadraturePolicy

        object.__setattr__(self, "_route_restore", True)
        try:
            for name in self._route_fields:
                if name in attrs:
                    setattr(self, name, attrs[name])
            canonical = {
                "coil_polysection_policy": PolySectionPolicy.resolve(
                    self.coil_polysection_policy
                ).key,
                "plasma_polysection_policy": PolySectionPolicy.resolve(
                    self.plasma_polysection_policy
                ).key,
                "inductance_target_policy": TargetQuadraturePolicy.resolve(
                    self.inductance_target_policy
                ).key,
                "force_target_policy": ForceTargetPolicy.resolve(
                    self.force_target_policy
                ).key,
            }
            for name, value in canonical.items():
                setattr(self, name, value)
            object.__setattr__(self, "_configured_route_attrs", canonical)
        finally:
            object.__setattr__(self, "_route_restore", False)

    @property
    def route_attrs(self) -> dict[str, str]:
        """Return complete cache identities for distinct source and target lanes."""
        return {
            "coil_polysection_policy": self.coil_polysection_policy,
            "plasma_polysection_policy": self.plasma_polysection_policy,
            "inductance_target_policy": self.inductance_target_policy,
            "force_target_policy": self.force_target_policy,
        }

    @property
    def coilset_attrs(self):
        """Return coilset attrs."""
        return self.frameset_attrs | self.biot_attrs | self.route_attrs

    def frame_factory_kwargs(self, name: str) -> dict:
        """Bind each cached factory to its explicit immutable route."""
        kwargs = super().frame_factory_kwargs(name)
        self._check_subframe_route_identity()
        route_attrs = self.route_attrs
        if name == "coil":
            return kwargs | {
                "polysection_policy": route_attrs["coil_polysection_policy"]
            }
        if name == "firstwall":
            return kwargs | {
                "polysection_policy": route_attrs["plasma_polysection_policy"]
            }
        if name == "inductance":
            return kwargs | {"target_policy": route_attrs["inductance_target_policy"]}
        if name == "force":
            return kwargs | {"target_policy": route_attrs["force_target_policy"]}
        return kwargs

    def _check_subframe_route_identity(self):
        """Reject executable row routes that disagree with the root cache key."""
        import numpy as np

        from nova.biot.polysection import PolySectionPolicy

        if len(self.subframe) == 0 or "polysection_policy" not in self.subframe:
            return
        segment = np.asarray(self.subframe["segment"], dtype=object)
        positions = np.flatnonzero(segment == "polysection")
        if len(positions) == 0:
            return
        plasma = np.asarray(self.subframe.plasma, dtype=bool)
        route_attrs = self.route_attrs
        expected = np.where(
            plasma[positions],
            route_attrs["plasma_polysection_policy"],
            route_attrs["coil_polysection_policy"],
        )
        actual = np.asarray(
            [
                PolySectionPolicy.resolve(value).key
                for value in np.asarray(
                    self.subframe["polysection_policy"], dtype=object
                )[positions]
            ],
            dtype=object,
        )
        mismatch = positions[actual != expected]
        if len(mismatch) > 0:
            labels = np.asarray(self.subframe.index, dtype=object)[mismatch].tolist()
            raise ValueError(
                "subframe polygon-section routes differ from the CoilSet cache "
                f"identity at {labels}"
            )

    def _check_route_compatibility(self, other):
        """Reject unions whose cached operators have different route meanings."""
        if self.route_attrs != other.route_attrs:
            raise ValueError(
                "cannot combine coilsets with different Biot route policies"
            )

    def restore_store_metadata(self):
        """Restore cache identity before frames and cached factories are rebuilt."""
        frameset = {
            name: self.data.attrs[name]
            for name in self.frameset_attrs
            if name in self.data.attrs
        }
        self.frameset_attrs = frameset
        for name in self._biot_attrs.keys() & self.data.attrs.keys():
            setattr(self, name, self.data.attrs[name])
        self._restore_route_attrs(
            {
                name: self.data.attrs[name]
                for name in self._route_fields
                if name in self.data.attrs
            }
        )

    def store(self):
        """Persist route identity beside frames and solved operator groups."""
        self._check_subframe_route_identity()
        self.data.attrs.update(self.coilset_attrs)
        return super().store()

    def __add__(self, other):
        """Return framset union of self and other."""
        self._check_route_compatibility(other)
        frame = self.frame + other.frame
        subframe = self.subframe + other.subframe
        circuit = self.circuit + other.circuit
        coilset = CoilSet(**self.route_attrs)
        coilset.frames = frame, subframe
        coilset.circuit = circuit
        return coilset

    def __iadd__(self, other):
        """Return coilset augmented by other."""
        self._check_route_compatibility(other)
        self.clear_biot()
        self.frame += other.frame
        self.subframe += other.subframe
        self.circuit += other.circuit
        return self


if __name__ == "__main__":
    reload = True
    if reload:
        coilset = CoilSet(dcoil=-5, dplasma=-500, tplasma="hex")
        coilset.coil.insert(
            1, 0.5, 0.95, 0.95, section="r", turn="r", nturn=-5.8, delta=-1, part="pf"
        )
        coilset.coil.insert(
            1,
            -0.5,
            0.95,
            0.95,
            section="hex",
            turn="c",
            tile=True,
            delta=-6,
            name="bubble",
        )
        coilset.coil.insert(2, 0, 0.95, 0.1, section="sk", nturn=-1.8)
        coilset.coil.insert(3, 0, 0.6, 0.9, section="r", turn="sk")
        coilset.firstwall.insert({"ellip": [4.2, -0.4, 1.25, 4.2]})
        coilset.shell.insert(
            {"e": [2.5, -1.25, 1.75, 1.0]}, 13, 0.05, delta=-4, part="vv"
        )

        coilset.sloc["Ic"] = 1
        coilset.sloc["Shl0", "Ic"] = -5

        coilset.grid.solve(500, 0.1)  # , 'plasma'
        coilset.plasmagrid.solve()

        coilset.plasma.separatrix = dict(c=[5, 0.25, 0.9])

        coilset.sloc["Ic"] = 6
        coilset.sloc["bubble", "Ic"] = 5
        coilset.sloc["Shl0", "Ic"] = -5
        coilset.sloc["plasma", "Ic"] = -1
        # coilset.store()
    else:
        coilset = CoilSet().load()

    separatrix = Polygon(dict(c=[4.2, -0.75, 0.9])).boundary
    coilset.plasma.separatrix = separatrix

    coilset.sloc["bubble", "Ic"] = 8
    coilset.sloc["passive", "Ic"] = 4
    coilset.sloc["plasma", "Ic"] = 1

    coilset.plot()

    coilset.grid.plot(levels=51)  # , nulls=False
    coilset.plasma.plot(levels=coilset.grid.levels)
    coilset.plasmagrid.plot(levels=coilset.grid.levels, colors="C6")

    coilset.plasmagrid.svd_rank = 50
    coilset.plasmagrid.plot_svd(levels=coilset.grid.levels)
