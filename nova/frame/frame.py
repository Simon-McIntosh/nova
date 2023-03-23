"""Manage frame factroy methods."""
from dataclasses import dataclass

from nova.frame import (Coil, Ferritic, FirstWall, Shell, Turn, Winding)
from nova.frame.frameset import FrameSet, frame_factory
from nova.geometry.polyshape import PolyShape


@dataclass
class Frame(FrameSet):
    """Manage methods for frameset construction."""

    delta: float = -1
    dcoil: float = -1
    dplasma: float = -1
    dshell: float = 0
    tcoil: str = 'rectangle'
    tplasma: str = 'rectangle'

    def __post_init__(self):
        """Update turn attribute names."""
        self._expand_polyattrs()
        super().__post_init__()

    def _expand_polyattrs(self):
        """Expand polyshape attrbutes."""
        for attr in ['tplasma', 'tcoil']:
            setattr(self, attr, PolyShape(getattr(self, attr)).shape)

    @property
    def frameset_attrs(self):
        """Return frame attributes."""
        return dict(dcoil=self.dcoil, dplasma=self.dplasma, dshell=self.dshell,
                    tcoil=self.tcoil, tplasma=self.tplasma, delta=self.delta)

    @frameset_attrs.setter
    def frameset_attrs(self, attrs: dict[str, int | float | str]):
        """Set frame attributes."""
        for attr in attrs:
            if hasattr(self, attr):
                setattr(self, attr, attrs[attr])
        self._expand_polyattrs()

    @frame_factory(Coil)
    def coil(self):
        """Return coil constructor."""
        return dict(turn=self.tcoil, delta=self.dcoil)

    @frame_factory(Ferritic)
    def ferritic(self):
        """Return ferritic insert constructor."""
        return dict(delta=self.delta)

    @frame_factory(FirstWall)
    def firstwall(self):
        """Return plasma firstwall constructor."""
        return dict(turn=self.tplasma, delta=self.dplasma)

    @frame_factory(Shell)
    def shell(self):
        """Return shell constructor."""
        return dict(delta=self.dshell)

    @frame_factory(Turn)
    def turn(self):
        """Return 2D/3D coil turn constructor."""
        return dict(delta=self.delta)

    @frame_factory(Winding)
    def winding(self):
        """Return winding constructor."""
        return dict(delta=self.delta)
