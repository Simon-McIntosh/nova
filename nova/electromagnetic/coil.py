"""Mesh poloidal coils."""
from dataclasses import dataclass, field

from nova.electromagnetic.framespace import FrameSpace
from nova.electromagnetic.poloidalgrid import PoloidalGrid


@dataclass
class Coil(PoloidalGrid):
    """Generate poloidal field coils (CS and PF)."""

    frame: FrameSpace = field(repr=False)
    subframe: FrameSpace = field(repr=False)
    delta: float
    tile: bool = False
    fill: bool = False
    turn: str = 'rectangle'
    default: dict = field(init=False, default_factory=lambda: {
        'label': 'Coil', 'part': 'coil', 'active': True})

    def set_conditional_attributes(self):
        """Set conditional attrs."""
        self.ifthen('delta', -1, 'turn', 'rectangle')
        self.ifthen('turn', 'hexagon', 'tile', True)
        self.ifthen('turn', 'hexagon', 'scale', 1)
        self.ifthen('delta', 0, 'turn', 'skin')
        self.ifthen('delta', 0, 'tile', False)
        self.ifthen('delta', 0, 'fill', False)
