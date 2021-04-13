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

    def update_conditionals(self):
        """Update conditional attrs."""
        self.ifthen('tile', True, 'turn', 'hexagon')
        self.ifthen('fill', True, 'turn', 'hexagon')
        self.ifthen('turn', 'hexagon', 'tile', True)
        self.ifthen('delta', -1, 'turn', 'skin')
        self.ifthen('delta', -1, 'tile', False)
        self.ifthen('delta', -1, 'fill', False)
