"""Manage biot methods."""
from dataclasses import dataclass
from functools import cached_property
import inspect
from typing import ClassVar

from nova.electromagnetic.biotdata import BiotData
from nova.electromagnetic.biotgrid import BiotGrid
from nova.electromagnetic.biotinductance import BiotInductance
from nova.electromagnetic.biotloop import BiotLoop
from nova.electromagnetic.biotpoint import BiotPoint
from nova.electromagnetic.frameset import FrameSet
from nova.electromagnetic.plasmagrid import PlasmaGrid


@dataclass
class BiotFactory(FrameSet):
    """Expose biot methods as cached properties."""

    dfield: float = -1
    biot_class: ClassVar[dict[str, BiotData]] = \
        dict(grid=BiotGrid, plasmagrid=PlasmaGrid, point=BiotPoint,
             probe=BiotPoint, loop=BiotLoop, inductance=BiotInductance)

    def _biotfactory(self):
        """Return nammed biot instance."""
        attr = inspect.stack()[1][3]  # name of caller
        return self.biot_class[attr](*self.frames, path=self.path, name=attr)

    @cached_property
    def grid(self):
        """Return grid biot instance."""
        return self._biotfactory()

    @cached_property
    def plasmagrid(self):
        """Return plasma grid biot instance."""
        return self._biotfactory()

    @cached_property
    def point(self):
        """Return point biot instance."""
        return self._biotfactory()

    @cached_property
    def probe(self):
        """Return biot probe instance."""
        return self._biotfactory()

    @cached_property
    def loop(self):
        """Return biot loop instance."""
        return self._biotfactory()

    @cached_property
    def inductance(self):
        """Return biot inductance instance."""
        return self._biotfactory()

    def clear_biot(self):
        """Clear all biot attributes."""
        delattrs = []
        for attr in self.__dict__:
            if isinstance(getattr(self, attr), BiotData):
                delattrs.append(attr)
        for attr in delattrs:
            delattr(self, attr)
