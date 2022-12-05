"""Manage biot methods."""
from dataclasses import dataclass, field
from functools import cached_property
import inspect
from typing import ClassVar

from nova.biot.biotdata import BiotData
from nova.database.netcdf import netCDF
from nova.frame.frameset import FrameSet


@dataclass
class Biot(FrameSet):
    """Expose biot methods as cached properties."""

    field_attrs: list[str] = field(default_factory=lambda: ['Br', 'Bz', 'Psi'])
    dfield: float = field(default=-1, repr=False)

    _biotmethods: ClassVar[dict[str, str]] = dict(
        plasma='.plasma.Plasma',
        point='.biotpoint.BiotPoint',
        grid='.biotgrid.BiotGrid',
        plasmaboundary='.biotplasmaboundary.BiotPlasmaBoundary',
        plasmagrid='.biotplasmagrid.BiotPlasmaGrid',
        probe='.biotpoint.BiotPoint',
        loop='.biotloop.BiotLoop',
        inductance='.boitinductance.BiotInductance',
        )

    def _biotfactory(self):
        """Return nammed biot instance."""
        name = inspect.getframeinfo(inspect.currentframe().f_back, 0)[2]
        method = self.import_method(self._biotmethods[name], 'nova.biot')
        return method(*self.frames, path=self.path, name=name,
                      filename=self.filename, attrs=self.field_attrs)

    @property
    def biot_attrs(self):
        """Return frame attributes."""
        return dict(dfield=self.dfield, field_attrs=self.field_attrs)

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
        Plasma = self.import_method(self._biotmethods['plasma'], 'nova.biot')
        return Plasma(*self.frames, path=self.path,
                      grid=self.plasmagrid, boundary=self.plasmaboundary)

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
