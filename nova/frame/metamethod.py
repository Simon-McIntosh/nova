"""AbstractBaseClass Extended FrameSpace._methods."""
from dataclasses import dataclass, field
from importlib import import_module
from typing import ClassVar

import numpy as np
import pandas


@dataclass
class MetaMethod:
    """Manage DataFrame methods."""

    frame: pandas.DataFrame = field(repr=False)
    required: list[str] = field(default_factory=list)
    base: list[str] = field(default_factory=list)
    additional: list[str] = field(default_factory=list)
    require_all: bool = True

    _subclass: ClassVar[str] = '.frame.metamethod.MetaMethod'

    def __post_init__(self):
        """Update metadata."""
        if self.generate:
            self.update_base()
            self.update_additional()

    def __call__(self):
        """Return metamethod subclass."""
        _module = '.'.join(self._subclass.split('.')[:-1])
        _method = self._subclass.split('.')[-1]
        return getattr(import_module(_module, 'nova'), _method)(self.frame)

    def initialize(self):
        """Init metamethod."""
        raise NotImplementedError()

    @property
    def generate(self):
        """Return True if required attributes set else False."""
        if not self.required:  # required attributes empty
            return True
        if self.require_all:
            return self.required_attributes.all()
        return self.required_attributes.any()

    @property
    def required_attributes(self):
        """Return boolean status of attributes found in frame.columns."""
        return np.array([attr in self.frame
                         or attr in self.frame.metaframe.available
                         for attr in self.required])

    def unset(self, attributes: list[str]) -> list[str]:
        """Return unset attributes."""
        return [attr for attr in list(dict.fromkeys(attributes))
                if attr not in self.frame.metaframe.columns]

    def update_base(self):
        """Update base attributes."""
        if self.base:
            self.frame.metaframe.metadata = {'base': self.base}

    def update_additional(self):
        """Update additional attributes if subset exsists in frame.columns."""
        additional = self.unset(self.base + self.required + self.additional)
        if not self.require_all:
            additional.extend(self.unset(self.required))
        if additional:
            self.frame.metaframe.metadata = {'additional': additional}

    def update_available(self, attrs):
        """Update metaframe.available if attrs unset and available."""
        available = [attr for attr in self.unset(attrs)
                     if attr in self.frame.metaframe.available]
        if available:
            self.frame.metaframe.metadata = {'additional': available}


@dataclass
class VtkGeo(MetaMethod):
    """Volume vtk geometry metamethod."""

    name: str = field(init=False, default='vtkgeo')
    required: list[str] = field(default_factory=lambda: ['vtk'])

    _subclass: ClassVar[str] = '.frame.geometry.VtkGeo'


@dataclass
class PolyGeo(MetaMethod):
    """
    Polygon geometrical methods for FrameSpace.

    Extract geometric features from shapely polygons.
    """

    name: str = field(init=False, default='polygeo')
    required: list[str] = field(default_factory=lambda: [
        'segment', 'section', 'poly'])
    require_all: bool = False

    _subclass: ClassVar[str] = '.frame.geometry.PolyGeo'


@dataclass
class PolyPlot(MetaMethod):
    """Methods for ploting FrameSpace data."""

    name: str = field(init=False, default='polyplot')
    required: list[str] = field(default_factory=lambda: ['poly'])

    _subclass: ClassVar[str] = '.frame.polyplot.PolyPlot'


@dataclass
class VtkPlot(MetaMethod):
    """Methods for ploting 3D FrameSpace data."""

    name: str = field(init=False, default='vtkplot')
    required: list[str] = field(default_factory=lambda: ['vtk'])

    _subclass: ClassVar[str] = '.frame.vtkplot.VtkPlot'


@dataclass
class Energize(MetaMethod):
    """Manage dependant frame energization parameters."""

    name: str = field(init=False, default='energize')
    required: list[str] = field(default_factory=lambda: ['It', 'nturn'])
    require_all: bool = False
    additional: list[str] = field(default_factory=lambda: ['Ic'])
    available: dict[str, bool] = field(default_factory=lambda: {
        'Ic': False, 'nturn': False})

    _subclass: ClassVar[str] = '.frame.energize.Energize'

    def __post_init__(self):
        """Update energize key."""
        if self.generate:
            self.frame.metaframe.energize = ['It']  # set metaframe key
            if np.array([attr in self.frame.metaframe.subspace
                         for attr in self.required]).any():
                self.frame.metaframe.metadata = \
                    {'subspace': self.required+self.additional}
        else:
            self.update_available(self.additional)
        super().__post_init__()


@dataclass
class MultiPoint(MetaMethod):
    """Manage multi-point constraints applied across frame.index."""

    name: str = field(init=False, default='multipoint')
    required: list[str] = field(default_factory=lambda: ['link'])
    require_all: bool = True

    _subclass: ClassVar[str] = '.frame.multipoint.MultiPoint'


@dataclass
class Select(MetaMethod):
    """Manage dependant frame energization parameters."""

    name: str = field(init=False, default='select')
    required: list[str] = field(default_factory=lambda: [
        'active', 'plasma', 'fix', 'ferritic'], repr=False)
    require_all: bool = False

    _subclass: ClassVar[str] = '.frame.select.Select'


@dataclass
class CrossSection(MetaMethod):
    """
    Cross-section methods for Biot Frame.

    Set cross-section factors used in Biot_Savart calculations.
    """

    name: str = field(init=False, default='biotsection')
    required: list[str] = field(default_factory=lambda: ['section'])

    _subclass: ClassVar[str] = '.biot.crosssection.CrossSection'


@dataclass
class Shape(MetaMethod):
    """Shape methods for Biot Frame."""

    name: str = field(init=False, default='biotshape')

    _subclass: ClassVar[str] = '.biot.shape.Shape'


@dataclass
class Reduce(MetaMethod):
    """Calculate reduction indices for reduceat."""

    name: str = field(init=False, default='biotreduce')

    _subclass: ClassVar[str] = '.biot.reduce.Reduce'
