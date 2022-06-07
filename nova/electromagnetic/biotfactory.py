"""Manage biot methods."""
from dataclasses import dataclass, field
from functools import cached_property
import inspect
from typing import ClassVar

from nova.database.netcdf import netCDF
from nova.electromagnetic.biotdata import BiotData
from nova.electromagnetic.biotgrid import BiotGrid
from nova.electromagnetic.biotinductance import BiotInductance
from nova.electromagnetic.biotloop import BiotLoop
from nova.electromagnetic.biotplasmagrid import BiotPlasmaGrid
from nova.electromagnetic.biotplasmaboundary import BiotPlasmaBoundary
from nova.electromagnetic.biotpoint import BiotPoint
from nova.electromagnetic.frameset import FrameSet
from nova.electromagnetic.plasma import Plasma
from nova.electromagnetic.circuit import Circuit


@dataclass
class BiotFactory(FrameSet):
    """Expose biot methods as cached properties."""

    dfield: float = field(default=-1, repr=False)
    biot_class: ClassVar[dict[str, BiotData]] = \
        dict(point=BiotPoint, grid=BiotGrid,
             plasmaboundary=BiotPlasmaBoundary, plasmagrid=BiotPlasmaGrid,
             probe=BiotPoint, loop=BiotLoop, inductance=BiotInductance,
             circuit=Circuit)

    def _biotfactory(self):
        """Return nammed biot instance."""
        attr = inspect.stack()[1][3]  # name of caller
        return self.biot_class[attr](*self.frames, path=self.path, name=attr,
                                     filename=self.filename)

    @property
    def biot_attrs(self):
        """Return frame attributes."""
        return dict(dfield=self.dfield)

    @property
    def biot_methods(self):
        """Return list of active biot methods."""
        attrs = []
        for attr in self.__dict__:
            if isinstance(getattr(self, attr), netCDF):
                attrs.append(attr)
        return attrs

    @cached_property
    def plasma(self):
        """Return plasma instance."""
        return Plasma(*self.frames, path=self.path,
                      grid=self.plasmagrid, boundary=self.plasmaboundary)

    @cached_property
    def circuit(self):
        """Return power supply instance."""
        return self._biotfactory()

    @cached_property
    def grid(self):
        """Return grid biot instance."""
        return self._biotfactory()

    @cached_property
    def plasmagrid(self):
        """Return plasma grid biot instance."""
        return self._biotfactory()

    @cached_property
    def plasmaboundary(self):
        """Return plasma firstwall biot instance."""
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
