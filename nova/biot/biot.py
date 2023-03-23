"""Manage biot methods."""
from dataclasses import dataclass
from typing import ClassVar

from nova.biot import (Gap, Grid, Inductance, Loop,
                       PlasmaWall, PlasmaGrid, Point,
                       Field, Force, LevelSet, Plasma)
from nova.biot.data import Data
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

    @frame_factory(Gap)
    def wallgap(self):
        """Return biot wall-gap probe instance."""
        return self.gap_kwargs


@dataclass
class Biot(WallGap, FrameSet):
    """Expose biot methods as cached properties."""

    ngrid: int | float = None
    nwall: int | float = None
    nlevelset: int | float = None
    nforce: int | float = None
    nfield: int | float = None
    ninductance: int | float = None

    force_attrs: ClassVar[list[str]] = ['Fr', 'Fz', 'Fc']
    field_attrs: ClassVar[list[str]] = ['Br', 'Bz', 'Psi']

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

    @frame_factory(Grid)
    def grid(self):
        """Return grid biot instance."""
        return {'number': self.ngrid} | self.field_kwargs

    @frame_factory(LevelSet)
    def levelset(self):
        """Return plasma grid biot instance."""
        return {'number': self.nlevelset, 'attrs': ['Psi']}

    @frame_factory(PlasmaGrid)
    def plasmagrid(self):
        """Return plasma grid biot instance."""
        return self.field_kwargs

    @frame_factory(PlasmaWall)
    def plasmawall(self):
        """Return plasma firstwall biot instance."""
        return {'number': self.nwall, 'attrs': ['Psi']}

    @frame_factory(Point)
    def point(self):
        """Return point biot instance."""
        return self.field_kwargs

    @frame_factory(Point)
    def probe(self):
        """Return biot probe instance."""
        return self.field_kwargs

    @frame_factory(Loop)
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

    @frame_factory(Inductance)
    def inductance(self):
        """Return biot inductance instance."""
        return {'number': self.ninductance, 'attrs': ['Psi']}

    def clear_biot(self):
        """Clear all biot attributes."""
        delattrs = []
        for attr in self.__dict__:
            if isinstance(getattr(self, attr), Data):
                delattrs.append(attr)
        for attr in delattrs:
            delattr(self, attr)
