"""Mesh poloidal coils."""
from dataclasses import dataclass, field

from nova.electromagnetic.frame import Frame
from nova.electromagnetic.poloidalgrid import PoloidalGrid


@dataclass
class Coil(PoloidalGrid):
    """Generate poloidal field coils (CS and PF)."""

    frame: Frame = field(repr=False)
    subframe: Frame = field(repr=False)
    delta: float
    tile: bool = False
    fill: bool = False
    turn: str = 'rectangle'

    def set_conditional_attributes(self):
        """Set conditional attrs."""
        self.ifthen('turn', 'hexagon', 'tile', True)
        self.ifthen('turn', 'hexagon', 'scale', 1)
        self.ifthen('delta', -1, 'turn', 'skin')
        self.ifthen('delta', -1, 'tile', False)
        self.ifthen('delta', -1, 'fill', False)
