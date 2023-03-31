"""Manage biot methods."""
from dataclasses import dataclass, field
from typing import ClassVar

from nova.biot import (Gap, Grid, Inductance, Loop,
                       Wall, PlasmaGrid, Point, Select,
                       Field, Force, LevelSet, Plasma)
from nova.biot.data import Data
from nova.database.netcdf import netCDF
from nova.frame.frameset import FrameSet, frame_factory

Nbiot = int | float | None


@dataclass
class BiotBase(FrameSet):
    """Biot methods base class."""

    _biot_attrs: dict[str, list[str] | Nbiot] = field(
        init=False, default_factory=dict)
    force_attrs: ClassVar[list[str]] = ['Fr', 'Fz', 'Fc']
    field_attrs: ClassVar[list[str]] = ['Br', 'Bz', 'Psi']

    def __post_init__(self):
        """Append biot attrs."""
        self.append_biot_attrs(['field_attrs', 'force_attrs'])
        super().__post_init__()

    def append_biot_attrs(self, attrs: list[str]):
        """Append biot attributes."""
        self._biot_attrs |= {attr: getattr(self, attr) for attr in attrs}

    @property
    def field_kwargs(self):
        """Return field kwargs."""
        return {'attrs': self.field_attrs}

    @property
    def force_kwargs(self):
        """Return force kwargs."""
        return {'attrs': self.force_attrs}

    @property
    def biot_methods(self):
        """Return list of active biot methods."""
        attrs = []
        for attr in self.__dict__:
            if isinstance(getattr(self, attr), netCDF):
                attrs.append(attr)
        return attrs

    def clear_biot(self):
        """Clear all biot attributes."""
        delattrs = []
        for attr in self.__dict__:
            if isinstance(getattr(self, attr), Data):
                delattrs.append(attr)
        for attr in delattrs:
            delattr(self, attr)


@dataclass
class BiotPlasma(BiotBase):
    """Group plasma biot methods."""

    nwall: Nbiot = None
    nselect: Nbiot = None
    nlevelset: Nbiot = None

    def __post_init__(self):
        """Append biot attrs."""
        self.append_biot_attrs(['nwall', 'nselect', 'nlevelset'])
        super().__post_init__()

    @frame_factory(Plasma)
    def plasma(self):
        """Return plasma instance."""
        return {'dirname': self.path, 'grid': self.plasmagrid,
                'wall': self.plasmawall, 'levelset': self.levelset,
                'select': self.select}

    @frame_factory(Select)
    def select(self):
        """Return unstructured grid instance for fast nearest node queries."""
        return {'number': self.nselect, 'attrs': ['Psi']}

    @frame_factory(LevelSet)
    def levelset(self):
        """Return plasma grid biot instance."""
        return {'number': self.nlevelset, 'attrs': ['Psi']}

    @frame_factory(PlasmaGrid)
    def plasmagrid(self):
        """Return plasma grid biot instance."""
        return self.field_kwargs

    @frame_factory(Wall)
    def plasmawall(self):
        """Return plasma firstwall biot instance."""
        return {'number': self.nwall, 'attrs': ['Psi']}


@dataclass
class BiotCoil(BiotBase):
    """Group coil biot methods."""

    nforce: Nbiot = None
    nfield: Nbiot = None
    ninductance: Nbiot = None

    def __post_init__(self):
        """Append biot attrs."""
        self.append_biot_attrs(['nforce', 'nfield', 'ninductance'])
        super().__post_init__()

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


@dataclass
class BiotGap(BiotBase):
    """Manage biot gap probe methods."""

    ngap: Nbiot = None
    mingap: int | float = 0
    maxgap: int | float = 3

    def __post_init__(self):
        """Append biot attrs."""
        self.append_biot_attrs(['ngap', 'mingap', 'maxgap'])
        super().__post_init__()

    @frame_factory(Gap)
    def plasmagap(self):
        """Return biot wall-gap probe instance."""
        return {'ngap': self.ngap,
                'mingap': self.mingap,
                'maxgap': self.maxgap}


@dataclass
class Biot(BiotPlasma, BiotCoil, BiotGap):
    """Expose biot methods as cached properties."""

    ngrid: Nbiot = None

    def __post_init__(self):
        """Append biot attrs."""
        self.append_biot_attrs(['ngrid'])
        super().__post_init__()

    @property
    def biot_attrs(self):
        """Return frame attributes."""
        return {attr: value for attr, value in self._biot_attrs.items()
                if value is not None}

    @frame_factory(Grid)
    def grid(self):
        """Return grid biot instance."""
        return {'number': self.ngrid} | self.field_kwargs

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
