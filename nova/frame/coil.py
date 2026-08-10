"""Mesh poloidal coils."""

from dataclasses import dataclass, field

from nova.frame.poloidalgrid import PoloidalGrid


@dataclass
class Coil(PoloidalGrid):
    """Generate poloidal field coils (CS and PF)."""

    tile: bool = False
    fill: bool = False
    turn: str = "rectangle"
    section: str = "rectangle"
    segment: str = "polysection"
    polysection_policy: str = ""
    _configured_polysection_policy: str = field(init=False, repr=False, default="")
    """Biot element every subframe filament is coupled through.

    :class:`nova.biot.polysection.PolySection` spreads each filament's current over
    its own section polygon and evaluates the exact kernel on every pair, so a coil
    operator carries no filament far field and no seam between the two treatments.
    Source sub-sections compose into the authored conductor through their exact
    area fractions. Linked flux is separately averaged over positive nodes in each
    target material cell, so its reduced value converges with the target mesh and
    target-rule order rather than claiming subdivision independence.
    """
    required: list[str] = field(default_factory=lambda: ["x", "z", "dl", "dt"])
    attributes: list[str] = field(
        init=False,
        default_factory=lambda: [
            "trim",
            "fill",
            "delta",
            "turn",
            "section",
            "segment",
            "polysection_policy",
            "tile",
            "ifttt",
        ],
    )
    """Fields carried from this class onto an insert that does not name them.

    ``segment`` is here so the field above is what an unqualified insert gets; a
    name absent from this list falls through to the frame metadata's own default
    instead, whatever the class declares.
    """
    default: dict = field(
        init=False,
        default_factory=lambda: {"label": "Coil", "part": "coil", "active": True},
    )

    def __post_init__(self):
        """Freeze the constructor route used by every insert from this factory."""
        from nova.biot.polysection import PolySectionPolicy

        self._configured_polysection_policy = PolySectionPolicy.resolve(
            self.polysection_policy
        ).key
        self.polysection_policy = self._configured_polysection_policy
        if hasattr(super(), "__post_init__"):
            super().__post_init__()

    def insert(self, *args, required=None, iloc=None, **additional):
        """Insert coils with a canonical immutable polygon-source policy."""
        from nova.biot.polysection import PolySectionPolicy

        current = PolySectionPolicy.resolve(self.polysection_policy).key
        requested = PolySectionPolicy.resolve(
            additional.get("polysection_policy", current)
        ).key
        if current != self._configured_polysection_policy or requested != current:
            raise ValueError(
                "coil polygon-section policy is fixed by its CoilSet constructor"
            )
        additional["polysection_policy"] = current
        return super().insert(*args, required=required, iloc=iloc, **additional)

    def set_conditional_attributes(self):
        """Set conditional attrs."""
        # an undivided rectangular coil is one axis-aligned box, which
        # nova.biot.cylinder integrates in four corner antiderivatives -- the same
        # exact section integral the polygon kernel returns, for a fraction of its
        # cost, and the guard is what keeps it off any other section shape
        self.ifthen(
            ["delta", "section", "tile", "fill"],
            [-1, "rectangle", False, False],
            "segment",
            "cylinder",
        )
        self.ifthen("turn", "hexagon", "tile", True)
        self.ifthen("turn", "hexagon", "scale", 1)
        self.ifthen("delta", 0, "turn", "skin")
        self.ifthen("delta", 0, "tile", False)
        self.ifthen("delta", 0, "fill", False)
