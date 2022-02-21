"""Manage frame factroy methods."""
from dataclasses import dataclass
from functools import cached_property

from nova.electromagnetic.coil import Coil
from nova.electromagnetic.frameset import FrameSet
from nova.electromagnetic.winding import Winding
from nova.electromagnetic.shell import Shell
from nova.electromagnetic.turn import Turn
from nova.electromagnetic.plasma import Plasma
from nova.electromagnetic.ferritic import Ferritic


@dataclass
class FrameFactory(FrameSet):
    """Manage methods for frameset construction."""

    delta: float = -1
    dcoil: float = -1
    dplasma: float = 0.25
    dshell: float = 0
    tplasma: str = 'hex'
    tcoil: str = 'rect'

    @property
    def frame_attrs(self):
        """Return frame attributes."""
        return dict(dcoil=self.dcoil, dplasma=self.dplasma, dshell=self.dshell,
                    tcoil=self.tcoil, tplasma=self.tplasma)

    @cached_property
    def coil(self):
        """Return coil constructor."""
        return Coil(*self.frames, turn=self.tcoil, delta=self.dcoil)

    @cached_property
    def ferritic(self):
        """Return ferritic insert constructor."""
        return Ferritic(*self.frames, delta=self.delta)

    @cached_property
    def plasma(self):
        """Return plasma constructor."""
        return Plasma(*self.frames, turn=self.tplasma, delta=self.dplasma)

    @cached_property
    def shell(self):
        """Return shell constructor."""
        return Shell(*self.frames, delta=self.dshell)

    @cached_property
    def turn(self):
        """Return 3D coil turn constructor."""
        return Turn(*self.frames, delta=self.delta)

    @cached_property
    def winding(self):
        """Return winding constructor."""
        return Winding(*self.frames, delta=self.delta)
