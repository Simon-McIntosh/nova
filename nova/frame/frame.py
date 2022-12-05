"""Manage frame factroy methods."""
from dataclasses import dataclass
from functools import cached_property
import inspect
from typing import ClassVar, Union

from nova.frame.frameset import FrameSet
from nova.geometry.polyshape import PolyShape


@dataclass
class Frame(FrameSet):
    """Manage methods for frameset construction."""

    delta: float = -1
    dcoil: float = -1
    nplasma: float = 1000
    dshell: float = 0
    tcoil: str = 'rectangle'
    tplasma: str = 'rectangle'

    _framemethods: ClassVar[dict[str, str]] = dict(
        coil='.coil.Coil',
        firstwall='.firstwall.FirstWall',
        winding='winding.Winding',
        shell='.shell.Shell',
        turn='.turn.Turn',
        ferritic='.ferritic.Ferritic',
        )

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
        return dict(dcoil=self.dcoil, nplasma=self.nplasma, dshell=self.dshell,
                    tcoil=self.tcoil, tplasma=self.tplasma, delta=self.delta)

    @frameset_attrs.setter
    def frameset_attrs(self, attrs: dict[str, Union[int, float, str]]):
        """Set frame attributes."""
        for attr in attrs:
            if hasattr(self, attr):
                setattr(self, attr, attrs[attr])
        self._expand_polyattrs()

    def _framefactory(self, **kwargs):
        """Return nammed biot instance."""
        name = inspect.getframeinfo(inspect.currentframe().f_back, 0)[2]
        method = self.import_method(self._framemethods[name], 'nova.frame')
        return method(*self.frames, **kwargs)

    @cached_property
    def coil(self):
        """Return coil constructor."""
        return self._framefactory(turn=self.tcoil, delta=self.dcoil)

    @cached_property
    def ferritic(self):
        """Return ferritic insert constructor."""
        return self._framefactory(delta=self.delta)

    @cached_property
    def firstwall(self):
        """Return plasma firstwall constructor."""
        return self._framefactory(turn=self.tplasma, delta=-self.nplasma)

    @cached_property
    def shell(self):
        """Return shell constructor."""
        return self._framefactory(delta=self.dshell)

    @cached_property
    def turn(self):
        """Return 2D/3D coil turn constructor."""
        return self._framefactory(delta=self.delta)

    @cached_property
    def winding(self):
        """Return winding constructor."""
        return self._framefactory(delta=self.delta)
