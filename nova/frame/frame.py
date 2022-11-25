"""Manage frame factroy methods."""
from dataclasses import dataclass
from functools import cached_property
from typing import Union

from nova.frame.coil import Coil
from nova.frame.firstwall import FirstWall
from nova.frame.frameset import FrameSet
from nova.frame.winding import Winding
from nova.frame.shell import Shell
from nova.frame.turn import Turn
from nova.frame.ferritic import Ferritic
from nova.geometry.polygen import PolyGen


@dataclass
class Frame(FrameSet):
    """Manage methods for frameset construction."""

    delta: float = -1
    dcoil: float = -1
    nplasma: float = 1000
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
            setattr(self, attr, PolyGen.polyshape[getattr(self, attr)])

    @property
    def frame_attrs(self):
        """Return frame attributes."""
        return dict(dcoil=self.dcoil, nplasma=self.nplasma, dshell=self.dshell,
                    tcoil=self.tcoil, tplasma=self.tplasma, delta=self.delta)

    @frame_attrs.setter
    def frame_attrs(self, attrs: dict[str, Union[int, float, str]]):
        """Set frame attributes."""
        for attr in attrs:
            if hasattr(self, attr):
                setattr(self, attr, attrs[attr])
        self._expand_polyattrs()

    @cached_property
    def coil(self):
        """Return coil constructor."""
        return Coil(*self.frames, turn=self.tcoil, delta=self.dcoil)

    @cached_property
    def ferritic(self):
        """Return ferritic insert constructor."""
        return Ferritic(*self.frames, delta=self.delta)

    @cached_property
    def firstwall(self):
        """Return plasma firstwall constructor."""
        return FirstWall(*self.frames, turn=self.tplasma, delta=-self.nplasma)

    @cached_property
    def shell(self):
        """Return shell constructor."""
        return Shell(*self.frames, delta=self.dshell)

    @cached_property
    def turn(self):
        """Return 2D/3D coil turn constructor."""
        return Turn(*self.frames, delta=self.delta)

    @cached_property
    def winding(self):
        """Return winding constructor."""
        return Winding(*self.frames, delta=self.delta)
