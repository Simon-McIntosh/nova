"""Manage access to IMAS machine data."""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import string
from typing import ClassVar, Union

import numpy as np
import numpy.typing as npt
import shapely

from nova.electromagnetic.coil import Coil
from nova.electromagnetic.coilset import CoilSet
from nova.electromagnetic.shell import Shell
from nova.geometry.polygon import Polygon
from nova.imas.database import Database
from nova.utilities.pyplot import plt


# pylint: disable=too-many-ancestors


@dataclass
class GeomData(ABC):
    """Geometry data baseclass."""

    ids_data: object = field(repr=False)
    data: dict[str, Union[int, float]] = field(init=False, repr=False,
                                               default_factory=dict)
    attrs: ClassVar[list[str]] = []

    def __post_init__(self):
        """Extract attributes from ids_data."""
        self.extract()

    @property
    @abstractmethod
    def name(self) -> str:
        """Return instance name."""

    @property
    @abstractmethod
    def poly(self) -> shapely.geometry.Polygon:
        """Return patch polygon."""

    def extract(self):
        """Extract attributes from ids_data and store in data."""
        data = getattr(self.ids_data, self.name)
        for attr in self.attrs:
            self.data[attr] = getattr(data, attr)

    def __getattr__(self, attr: str):
        """Return attributes from data."""
        return self.data[attr]

    @property
    def area(self):
        """Return section area."""
        return self.poly.area


@dataclass
class Outline(GeomData):
    """Polygonal poloidal patch."""

    name: str = 'outline'
    attrs: ClassVar[list[str]] = ['r', 'z']

    @property
    def poly(self):
        """Return shapely polygon."""
        return Polygon([self.data['r'], self.data['z']]).poly


@dataclass
class Rectangle(GeomData):
    """Rectangular poloidal patch."""

    name: str = 'rectangle'
    attrs: ClassVar[list[str]] = ['r', 'z', 'width', 'height']

    @property
    def poly(self):
        """Return shapely polygon."""
        return Polygon({'r': [self.data['r'], self.data['z'],
                              self.data['width'], self.data['height']]}).poly


@dataclass
class Oblique(GeomData):
    """Oblique poloidal patch (parallelogram)."""

    name: str = 'oblique'
    attrs: ClassVar[list[str]] = ['r', 'z', 'length_alpha', 'length_beta',
                                  'alpha', 'beta']

    @property
    def poly(self):
        """Return skewed shapely polygon."""
        radius = self.r + np.array(
            [0, self.length_alpha * np.cos(self.alpha),
             self.length_alpha * np.cos(self.alpha)
             - self.length_beta * np.sin(self.beta),
             -self.length_beta * np.sin(self.beta)])
        height = self.z + np.array(
            [0, self.length_alpha * np.sin(self.alpha),
             self.length_alpha * np.sin(self.alpha)
             + self.length_beta * np.cos(self.beta),
             self.length_beta * np.cos(self.beta)])
        return Polygon([radius, height]).poly

    @property
    def start(self):
        """Return oblique geometry start point."""
        if self.length_alpha > self.length_beta:
            return np.array([self.r - self.length_beta/2 * np.sin(self.beta),
                             self.z + self.length_beta/2 * np.cos(self.beta)])
        return np.array([self.r + self.length_alpha/2 * np.cos(self.alpha),
                         self.z + self.length_alpha/2 * np.sin(self.alpha)])

    @property
    def end(self):
        """Return oblique geometry end point."""
        if self.length_alpha > self.length_beta:
            return self.start + np.array(
                [self.length_alpha * np.cos(self.alpha),
                 self.length_alpha * np.sin(self.alpha)])
        return self.start + np.array(
            [-self.length_beta * np.sin(self.beta),
             self.length_beta * np.cos(self.beta)])

    @property
    def points(self):
        """Return start and end points."""
        return self.start, self.end

    @property
    def length(self):
        """Return oblique pannel length."""
        return np.linalg.norm(self.end - self.start)

    @property
    def thickness(self):
        """Return oplique pannel thickness."""
        return self.area / self.length

    def plot(self):
        """Plot oblique patch verticies and start/end points."""
        axes = plt.gca()
        axes.plot(*self.poly.boundary.xy, 'o', label='vertex')
        axes.plot(*self.start, 'C1o', label='start')
        axes.plot(*self.end, 'C3o', label='end')
        axes.axis('equal')
        axes.legend()


@dataclass
class Arcs(GeomData):
    """Polygonal poloidal patch."""

    name: str = 'arcs'
    attrs: ClassVar[list[str]] = []

    @property
    def poly(self):
        """Return shapely polygon."""
        raise NotImplementedError


@dataclass
class Geometry:
    """Manage poloidal cross-sections."""

    ids_data: object = field(repr=False)
    data: GeomData = field(init=False)
    transform: ClassVar[dict[int, str]] = \
        {1: Outline, 2: Rectangle, 3: Oblique, 4: Arcs}

    def __post_init__(self):
        """Build geometry instance."""
        self.data = self.transform[self.ids_data.geometry_type](self.ids_data)

    def __getattr__(self, attr):
        """Return data attributes."""
        return getattr(self.data, attr)


@dataclass
class Loop:
    """Poloidal loop."""

    ids_data: object = field(repr=False)
    name: str = field(init=False)
    label: str = field(init=False)
    resistance: float = field(init=False)

    def __post_init__(self):
        """Extract data from loop ids."""
        self.name = self.ids_data.name.strip()
        self.label = self.name.rstrip(string.digits + '_')
        self.resistance = self.ids_data.resistance


@dataclass
class ActiveLoop(Loop):
    """Poloidal coil."""

    identifier: str = field(init=False)

    def __post_init__(self):
        """Extract data from loop ids."""
        super().__post_init__()
        self.identifier = self.ids_data.identifier


@dataclass
class Element:
    """Poloidal element."""

    ids_data: object = field(repr=False)
    index: int = 0
    name: str = field(init=False)
    nturn: float = field(init=False)
    geometry: Geometry = field(init=False)

    def __post_init__(self):
        """Extract element data from ids."""
        self.name = self.ids_data.name.strip()
        self.nturn = self.ids_data.turns_with_sign
        self.geometry = Geometry(self.ids_data.geometry)

    def is_oblique(self) -> bool:
        """Return geometry.name == 'oblique'."""
        return self.geometry.name == 'oblique'

    def is_rectangular(self) -> bool:
        """Return geometry.name == 'rectangle'."""
        return self.geometry.name == 'rectangle'


@dataclass
class FrameData(ABC):
    """Frame data base class."""

    data: dict[str, list[float]] = field(init=False)
    element_attrs: ClassVar[list[str]] = []
    geometry_attrs: ClassVar[list[str]] = []
    loop_attrs: ClassVar[list[str]] = []

    def __post_init__(self):
        """Init data dict."""
        self.data = {attr: [] for attr in self.attrs}

    @property
    def attrs(self):
        """Return attribute list."""
        return self.element_attrs + self.geometry_attrs + self.loop_attrs

    def append(self, loop: Loop, element: Element):
        """Append data to internal structrue."""
        for attr in self.element_attrs:
            self.data[attr].append(getattr(element, attr))
        for attr in self.geometry_attrs:
            self.data[attr].append(getattr(element.geometry, attr))
        for attr in self.loop_attrs:
            self.data[attr].append(getattr(loop, attr))

    @property
    def part(self):
        """Return part name."""
        try:
            label = self.data['name'][0]
        except KeyError:
            label = self.data['identifier'][0]
        if isinstance(label, list):
            label = label[0]
        if 'VES' in label:
            return 'vv'
        if 'TRI' in label:
            return 'trs'
        if label == 'INB_RAIL':
            return 'dir'
        if 'CS' in label:
            return 'cs'
        if 'PF' in label:
            return 'pf'
        if 'VS3' in label:
            return 'vs3'
        return ''

    @staticmethod
    def update_resistivity(index, frame, subframe, resistance):
        """Update frame and subframe resistivity."""
        rho = resistance * frame.loc[index, 'area'] / frame.loc[index, 'dy']
        frame.loc[index, 'rho'] = rho
        for i, name in enumerate(index):
            subindex = subframe.frame == name
            subframe.loc[subindex, 'rho'] = rho[i]


@dataclass
class PassiveShellData(FrameData):
    """Extract oblique shell geometries from pf_passive ids."""

    length: float = 0
    points: list[npt.ArrayLike] = field(init=False, repr=False,
                                        default_factory=list)
    loop_attrs: ClassVar[list[str]] = ['name', 'resistance']
    geometry_attrs: ClassVar[list[str]] = ['thickness']

    def __len__(self):
        """Return loop number."""
        return len(self.points)

    def append(self, loop: Loop, element: Element):
        """Check start/end point colocation."""
        assert element.is_oblique()
        if not self.points:
            return self._new(loop, element)
        if np.allclose(self.points[-1][-1], element.geometry.start):
            return self._end(loop, element)
        return self._new(loop, element)

    def _new(self, loop: Loop, element: Element):
        """Start new loop."""
        geometry = element.geometry
        self.points.append(np.c_[geometry.start, geometry.end].T)
        for attr in self.loop_attrs:
            self.data[attr].append([getattr(loop, attr)])
        for attr in self.geometry_attrs:
            self.data[attr].append([getattr(geometry, attr)])

    def _end(self, loop: Loop, element: Element):
        """Append endpoint to current loop."""
        geometry = element.geometry
        self.points[-1] = np.append(
            self.points[-1], geometry.end.reshape(1, -1), axis=0)
        for attr in self.loop_attrs:
            self.data[attr][-1].append(getattr(loop, attr))
        for attr in self.geometry_attrs:
            self.data[attr][-1].append(getattr(geometry, attr))

    def insert(self, shell: Shell):
        """Insert data into shell instance."""
        for i in range(len(self)):
            thickness = np.mean(self.data['thickness'][i])
            index = shell.insert(*self.points[i].T, self.length, thickness,
                                 rho=0,
                                 name=self.data['name'][i], part=self.part)
            self.update_resistivity(index, *shell.frames,
                                    self.data['resistance'][i])

    def plot(self):
        """Plot shell centerlines."""
        axes = plt.gca()
        for loop in self.points:
            axes.plot(loop[:, 0], loop[:, 1], 'o-')
        plt.axis('equal')
        plt.axis('off')


@dataclass
class CoilData(FrameData):
    """Extract coildata from ids."""

    name: list[str] = field(init=False, default_factory=list)
    geometry_attrs: ClassVar[list[str]] = ['r', 'z', 'width', 'height']
    loop_attrs: ClassVar[list[str]] = ['name', 'resistance']

    def append(self, loop: Loop, element: Element):
        """Append coil data to internal structrue."""
        super().append(loop, element)

    def insert(self, coil: Coil, **kwargs):
        """Insert data via coil method."""
        index = coil.insert(*[self.data[attr] for attr in self.geometry_attrs],
                            part=self.part, rho=0, **kwargs)
        self.update_resistivity(index, *coil.frames, self.data['resistance'])
        return index


@dataclass
class PassiveCoilData(CoilData):
    """Extract coildata from passive ids."""

    geometry_attrs: ClassVar[list[str]] = ['r', 'z', 'width', 'height']
    loop_attrs: ClassVar[list[str]] = ['name', 'resistance']

    def insert(self, coil: Coil, **kwargs):
        """Insert data via coil method."""
        kwargs = {'active': False, 'name': self.data['name'],
                  'turn': 'rect'} | kwargs
        return super().insert(coil, **kwargs)


@dataclass
class ActiveCoilData(CoilData):
    """Extract coildata from active ids."""

    element_attrs: ClassVar[list[str]] = ['nturn', 'index', 'name']
    geometry_attrs: ClassVar[list[str]] = ['r', 'z', 'width', 'height']
    loop_attrs: ClassVar[list[str]] = ['identifier', 'resistance']

    def __post_init__(self):
        """Init data dict."""
        self.data = {attr: [] for attr in self.attrs}

    def insert(self, coil: Coil, **kwargs):
        """Insert data via coil method."""
        factor = np.sign(self.data['nturn'])
        self.data['nturn'] = np.abs(self.data['nturn'])
        link = [''] + [self.data['name'][i-index+1] if index > 0 else ''
                       for i, index in enumerate(self.data['index'][1:])]
        kwargs = {'active': True, 'name': self.data['name'],
                  'nturn': self.data['nturn'],
                  'link': link, 'factor': factor} | kwargs
        return super().insert(coil, **kwargs)


@dataclass
class ActiveObliqueCoilData(ActiveCoilData):
    """Extract coildata from active ids."""

    geometry_attrs: ClassVar[list[str]] = ['poly']


@dataclass
class ContourData:
    """Extract contour data from ids_data."""

    data: dict[str, npt.ArrayLike] = field(init=False, default_factory=dict)

    def append(self, unit):
        """Append contour data."""
        self.data[unit.name] = np.array([unit.outline.r, unit.outline.z]).T

    def plot(self):
        """Plot contours."""
        for component in self.data.items():
            plt.plot(*self.data[component].T, label=component)
        plt.axis('equal')
        plt.despine()
        plt.axis('off')
        plt.legend()


@dataclass
class Contour:
    """Extract closed contour from multiple unordered segments."""

    data: dict[str, npt.ArrayLike]
    loop: npt.ArrayLike = field(init=False, default_factory=lambda:
                                np.ndarray((0, 2), float))
    segments: list[npt.ArrayLike] = field(init=False)

    def __post_init__(self):
        """Create segments list."""
        self.segments = list(self.data.values())
        self.loop = self.segments.pop(0)
        self.extract()

    def gap(self, index: int):
        """Return length of gap to next segment."""
        return [np.linalg.norm(segment[index] - self.loop[-1])
                for segment in self.segments]

    def append(self, index: int, flip=False):
        """Pop matching segment and append loop."""
        segment = self.segments.pop(index)
        if flip:
            segment = segment[::-1]
        self.loop = np.append(self.loop, segment, axis=0)

    def select(self):
        """Select matching segment and join to loop."""
        start = self.gap(0)
        end = self.gap(-1)
        if np.min(start) < np.min(end):
            return self.append(np.argmin(start))
        return self.append(np.argmin(end), flip=True)

    def extract(self):
        """Extract closed contour."""
        while len(self.segments) > 0:
            self.select()
        self.loop = np.append(self.loop, self.loop[:1], axis=0)
        assert shapely.geometry.LinearRing(self.loop).is_valid

    def plot(self):
        """Plot closed contour."""
        plt.plot(*self.loop.T, 'C3-')


@dataclass
class MachineDescription(CoilSet, Database):
    """Manage access to machine data."""

    def __post_init__(self):
        """Build geometry."""
        super().__post_init__()
        self.build()

    @abstractmethod
    def build(self):
        """Build geometry."""


@dataclass
class PF_Passive_Geometry(MachineDescription):
    """Manage passive poloidal loop ids, pf_passive."""

    shot: int = 115005
    run: int = 2
    ids_name: str = 'pf_passive'

    def build(self):
        """Build pf passive geometroy."""
        shelldata = PassiveShellData()
        coildata = PassiveCoilData()
        for ids_loop in self.load_ids_data().loop:
            loop = Loop(ids_loop)
            for i, ids_element in enumerate(ids_loop.element):
                element = Element(ids_element, i)
                if element.is_oblique():
                    shelldata.append(loop, element)
                    continue
                if element.is_rectangular():
                    coildata.append(loop, element)
                    continue
                raise NotImplementedError(f'geometory {element.geometry.name} '
                                          'not implemented')
        coildata.insert(self.coil)
        shelldata.insert(self.shell)


@dataclass
class PF_Active_Geometry(MachineDescription):
    """Manage active poloidal loop ids, pf_passive."""

    shot: int = 111001
    run: int = 1
    ids_name: str = 'pf_active'

    def build(self):
        """Build pf active geometroy."""
        coildata = ActiveCoilData()
        obliquecoildata = ActiveObliqueCoilData()
        for ids_loop in self.load_ids_data().coil:
            loop = ActiveLoop(ids_loop)
            for i, ids_element in enumerate(ids_loop.element):
                element = Element(ids_element, i)
                if element.is_rectangular():
                    coildata.append(loop, element)
                    continue
                if element.is_oblique():
                    obliquecoildata.append(loop, element)
                    continue
                raise NotImplementedError(f'geometory {element.geometry.name} '
                                          'not implemented')
        coildata.insert(self.coil)
        obliquecoildata.insert(self.coil)


@dataclass
class Wall_Geometry(MachineDescription):
    """Manage plasma boundary, wall ids."""

    shot: int = 116000
    run: int = 1
    ids_name: str = 'wall'

    def build(self):
        """Build plasma bound by firstwall contour."""
        firstwall = ContourData()
        limiter = self.load_ids_data().description_2d.array[0].limiter
        for unit in limiter.unit:
            firstwall.append(unit)
        contour = Contour(firstwall.data)  # extract closed loop
        self.plasma.insert(contour.loop)


@dataclass
class Machine(CoilSet, Database):
    """Manage ITER machine geometry."""

    shot: int = 135011
    run: int = 7
    tokamak: str = 'iter'
    datapath: str = 'data/Nova'

    geometry: Union[dict[str, tuple[int, int]], list[str]] = field(
        default_factory=lambda: ['pf_active', 'pf_passive', 'wall'])
    ids_name: str = field(init=False, default='multiple')

    machine_description: ClassVar[dict[str, MachineDescription]] = dict(
        pf_active=PF_Active_Geometry,
        pf_passive=PF_Passive_Geometry,
        wall=Wall_Geometry)

    def __post_init__(self):
        """Load coilset, build if not found."""
        super().__post_init__()
        try:
            self.load(self.filename)
        except (FileNotFoundError, OSError, KeyError):
            self.build()

    def load(self, filename: str, path=None):
        """Load machine geometry and data. Re-build if metadata diffrent."""
        super().load(filename, path)
        self.metadata = self.load_metadata(filename, path)

    def check_geometry(self):
        """Check geometry ids referances."""
        if isinstance(self.geometry, list):
            self.geometry = {attr: [self.shot, self.run]
                             for attr in self.geometry}

    def solve_biot(self):
        """Solve biot instances."""
        if self.sloc['plasma'].sum() > 0:
            self.plasmaboundary.solve(self.Loc['plasma', 'poly'][0].boundary)
            self.plasmagrid.solve()
            wall = self.Loc['plasma', :].iloc[0]
            self.plasma.update_separatrix(
                dict(e=[wall.x, wall.z, 0.7*wall.dx, 0.5*wall.dz]))

    @property
    def metadata(self):
        """Return machine metadata."""
        metadata = self.ids_attrs | self.frame_attrs
        metadata['geometry'] = list(self.geometry)
        for attr in self.geometry:
            metadata[attr] = self.geometry[attr]
        return metadata

    @metadata.setter
    def metadata(self, metadata: dict):
        """Set instance metadata."""
        for attr in list(self.ids_attrs) + list(self.frame_attrs):
            setattr(self, attr, metadata[attr])
        self.geometry = {}
        for attr in metadata['geometry']:
            self.geometry[attr] = list(metadata[attr])

    def build(self, **kwargs):
        """Build dataset, frameset and, biotset and save to file."""
        super().__post_init__()
        self.frame_attrs = kwargs
        self.clear_frameset()
        self.check_geometry()
        for attr in self.geometry:
            self += self.machine_description[attr](
                *self.geometry[attr], tokamak=self.tokamak, **self.frame_attrs)
        self.solve_biot()
        self.store(self.filename, metadata=self.metadata)


if __name__ == '__main__':

    coilset = Machine(135011, 7)
    # coilset.build(dcoil=0.25, dshell=0.5, dplasma=-500, tcoil='hex')

    # coilset.plasma.update_separatrix(dict(e=[6, -0.5, 1.5, 2.2]))

    coilset.sloc['Ic'] = 1
    coilset.plot()
    coilset.plasma.plot()
