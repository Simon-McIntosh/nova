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
    """Biot element every subframe filament is coupled through.

    :class:`nova.biot.polysection.PolySection` spreads each filament's current over
    its own section polygon and evaluates the exact kernel on every pair, so a coil
    operator carries no filament far field and no seam between the two treatments.
    A tiling of sub-sections sums to the whole section's integral identically, which
    makes the reduced coil-coil terms independent of ``delta``.
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

    def set_conditional_attributes(self):
        """Set conditional attrs."""
        # an undivided rectangular coil is one axis-aligned box, which
        # nova.biot.cylinder integrates in four corner antiderivatives -- the same
        # exact section integral the polygon kernel returns, for a fraction of its
        # cost, and the guard is what keeps it off any other section shape
        self.ifthen(["delta", "section"], [-1, "rectangle"], "segment", "cylinder")
        self.ifthen("turn", "hexagon", "tile", True)
        self.ifthen("turn", "hexagon", "scale", 1)
        self.ifthen("delta", 0, "turn", "skin")
        self.ifthen("delta", 0, "tile", False)
        self.ifthen("delta", 0, "fill", False)
