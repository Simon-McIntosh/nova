"""Manage biot methods."""
from dataclasses import dataclass, field

from nova.biot import (BiotGrid, BiotInductance, BiotLoop, BiotFirstWall,
                       BiotPlasmaGrid, BiotPoint, Field, Force, Plasma)
from nova.biot.biotdata import BiotData
from nova.database.netcdf import netCDF
from nova.frame.frameset import FrameSet, frame_factory


@dataclass
class Biot(FrameSet):
    """Expose biot methods as cached properties."""

    field_attrs: list[str] = field(default_factory=lambda: ['Br', 'Bz', 'Psi'])
    force_attrs: list[str] = field(default_factory=lambda: ['Fr', 'Fz', 'Fc'])
    nfield: int | float = field(default=0, repr=False)
    nforce: int | float = field(default=0, repr=False)

    @property
    def field_kwargs(self):
        """Return default biot factory kwargs."""
        return dict(attrs=self.field_attrs)

    @property
    def force_kwargs(self):
        """Return default biot factory kwargs."""
        return dict(attrs=self.force_attrs)

    @property
    def biot_attrs(self):
        """Return frame attributes."""
        return dict(nfield=self.nfield, nforce=self.nforce,
                    field_attrs=self.field_attrs,
                    force_attrs=self.force_attrs)

    @property
    def biot_methods(self):
        """Return list of active biot methods."""
        attrs = []
        for attr in self.__dict__:
            if isinstance(getattr(self, attr), netCDF):
                attrs.append(attr)
        return attrs

    @frame_factory(Plasma)
    def plasma(self):
        """Return plasma instance."""
        return dict(dirname=self.path, grid=self.plasmagrid,
                    wall=self.plasmawall)

    @frame_factory(BiotGrid)
    def grid(self):
        """Return grid biot instance."""
        return self.field_kwargs

    @frame_factory(BiotPlasmaGrid)
    def plasmagrid(self):
        """Return plasma grid biot instance."""
        return self.field_kwargs

    @frame_factory(BiotFirstWall)
    def plasmawall(self):
        """Return plasma firstwall biot instance."""
        return self.field_kwargs

    @frame_factory(BiotPoint)
    def point(self):
        """Return point biot instance."""
        return self.field_kwargs

    @frame_factory(BiotPoint)
    def probe(self):
        """Return biot probe instance."""
        return self.field_kwargs

    @frame_factory(BiotLoop)
    def loop(self):
        """Return biot loop instance."""
        return self.field_kwargs

    @frame_factory(Field)
    def field(self):
        """Return boundary field instance."""
        return dict(nfield=self.nfield)

    @frame_factory(Force)
    def force(self):
        """Return force field instance."""
        return dict(nforce=self.nforce, attrs=self.force_attrs)

    @frame_factory(BiotInductance)
    def inductance(self):
        """Return biot inductance instance."""
        return self.field_kwargs

    def clear_biot(self):
        """Clear all biot attributes."""
        delattrs = []
        for attr in self.__dict__:
            if isinstance(getattr(self, attr), BiotData):
                delattrs.append(attr)
        for attr in delattrs:
            delattr(self, attr)
