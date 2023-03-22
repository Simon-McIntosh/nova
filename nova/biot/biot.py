"""Manage biot methods."""
from dataclasses import dataclass, field

from nova.biot import (BiotGap, BiotGrid, BiotInductance, BiotLoop,
                       BiotFirstWall, BiotPlasmaGrid, BiotPoint,
                       Field, Force, LevelSet, Plasma)
from nova.biot.biotdata import BiotData
from nova.database.netcdf import netCDF
from nova.frame.frameset import FrameSet, frame_factory


@dataclass
class WallGap:
    """Manage wallgap biot probe attributes."""

    mingap: int | float = 1e-3
    maxgap: int | float = 5
    ngap: int | float = 150

    @property
    def gap_kwargs(self):
        """Return gap kwargs."""
        return {'mingap': self.mingap,
                'maxgap': self.maxgap,
                'ngap': self.ngap}

    @frame_factory(BiotGap)
    def wallgap(self):
        """Return biot wall-gap probe instance."""
        return self.gap_kwargs


@dataclass
class Biot(WallGap, FrameSet):
    """Expose biot methods as cached properties."""

    force_attrs: list[str] = field(default_factory=lambda: ['Fr', 'Fz', 'Fc'])
    field_attrs: list[str] = field(default_factory=lambda: ['Br', 'Bz', 'Psi'])
    nforce: int | float = None
    nfield: int | float = None
    ninductance: int | float = None
    nlevelset: int = 500

    @property
    def field_kwargs(self):
        """Return field kwargs."""
        return {'attrs': self.field_attrs}

    @property
    def force_kwargs(self):
        """Return force kwargs."""
        return {'attrs': self.force_attrs}

    @property
    def biot_attrs(self):
        """Return frame attributes."""
        kwargs = {attr: value for attr in
                  ['field_attrs', 'force_attrs', 'nfield', 'nforce',
                   'ninductance', 'nlevelset']
                  if (value := getattr(self, attr)) is not None}
        return kwargs | self.gap_kwargs

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
        return {'dirname': self.path, 'grid': self.plasmagrid,
                'wall': self.plasmawall, 'levelset': self.levelset}

    @frame_factory(BiotGrid)
    def grid(self):
        """Return grid biot instance."""
        return self.field_kwargs

    @frame_factory(LevelSet)
    def levelset(self):
        """Return plasma grid biot instance."""
        return {'attrs': ['Psi'], 'nlevelset': self.nlevelset}

    @frame_factory(BiotPlasmaGrid)
    def plasmagrid(self):
        """Return plasma grid biot instance."""
        return self.field_kwargs

    @frame_factory(BiotFirstWall)
    def plasmawall(self):
        """Return plasma firstwall biot instance."""
        return {'attrs': ['Psi']}

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
        return {'number': self.nfield}

    @frame_factory(Force)
    def force(self):
        """Return force field instance."""
        return {'number': self.nforce, 'attrs': self.force_attrs}

    @frame_factory(BiotInductance)
    def inductance(self):
        """Return biot inductance instance."""
        return {'number': self.ninductance, 'attrs': ['Psi']}

    def clear_biot(self):
        """Clear all biot attributes."""
        delattrs = []
        for attr in self.__dict__:
            if isinstance(getattr(self, attr), BiotData):
                delattrs.append(attr)
        for attr in delattrs:
            delattr(self, attr)
