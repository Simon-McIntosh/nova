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
    segment: str = "ring"
    required: list[str] = field(default_factory=lambda: ["x", "z", "dl", "dt"])
    default: dict = field(
        init=False,
        default_factory=lambda: {"label": "Coil", "part": "coil", "active": True},
    )

    def set_conditional_attributes(self):
        """Set conditional attrs."""
        self.ifthen(["delta", "section"], [-1, "rectangle"], "segment", "cylinder")
        self.ifthen("turn", "hexagon", "tile", True)
        self.ifthen("turn", "hexagon", "scale", 1)
        self.ifthen("delta", 0, "turn", "skin")
        self.ifthen("delta", 0, "tile", False)
        self.ifthen("delta", 0, "fill", False)
