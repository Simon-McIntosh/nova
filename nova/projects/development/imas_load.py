"""Manage access to IMAS machine data."""
from abc import ABC, abstractmethod
from contextlib import contextmanager
from dataclasses import dataclass, field
import string
from typing import ClassVar, Union

import numpy as np
import numpy.typing as npt
import shapely

from imas import imasdef, DBEntry
from nova.electromagnetic.coil import Coil
from nova.electromagnetic.coilset import CoilSet
from nova.electromagnetic.shell import Shell
from nova.geometry.polygon import Polygon
from nova.utilities.pyplot import plt


# pylint: disable=too-many-ancestors

@dataclass
class MachineDescription:
    """Methods to access IMAS machine description data."""

    user: str = 'public'
    tokamak: str = 'iter_md'
    backend: int = imasdef.MDSPLUS_BACKEND

    @contextmanager
    def database(self, shot: int, run: int, ids_name: str):
        """Database context manager."""
        database = DBEntry(self.backend, self.tokamak, shot, run,
                           user_name=self.user)
        database.open()
        yield database.get(ids_name, 0)
        database.close()

    def ids(self, shot: int, run: int, ids_name: str):
        """Return filled ids from dataabase."""
        with self.database(shot, run, ids_name) as database:
            return database


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

    @abstractmethod
    def append(self, loop: Loop, element: Element):
        """Append data to internal structures."""

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

    delta: float
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
                                 rho=0, delta=self.delta,
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

    delta: float
    name: list[str] = field(init=False, default_factory=list)
    geometry_attrs: ClassVar[list[str]] = ['r', 'z', 'width', 'height']
    loop_attrs: ClassVar[list[str]] = ['name', 'resistance']

    def append(self, loop: Loop, element: Element):
        """Append coil data to internal structrue."""
        assert element.is_rectangular()
        for attr in self.element_attrs:
            self.data[attr].append(getattr(element, attr))
        for attr in self.geometry_attrs:
            self.data[attr].append(getattr(element.geometry, attr))
        for attr in self.loop_attrs:
            self.data[attr].append(getattr(loop, attr))

    def insert(self, coil: Coil, **kwargs):
        """Insert data via coil method."""
        index = coil.insert(self.data['r'], self.data['z'],
                            self.data['width'], self.data['height'],
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
        kwargs = {'passive': True, 'name': self.data['name']} | kwargs
        return super().insert(coil, **kwargs)


@dataclass
class ActiveCoilData(CoilData):
    """Extract coildata from active ids."""

    element_attrs: ClassVar[list[str]] = ['nturn']
    geometry_attrs: ClassVar[list[str]] = ['r', 'z', 'width', 'height']
    loop_attrs: ClassVar[list[str]] = ['identifier', 'resistance']

    def __post_init__(self):
        """Init data dict."""
        self.data = {attr: [] for attr in self.attrs}

    def insert(self, coil: Coil, **kwargs):
        """Insert data via coil method."""
        kwargs = {'active': True, 'name': self.data['identifier'],
                  'nturn': self.data['nturn']} | kwargs
        return super().insert(coil, **kwargs)


@dataclass
class MachineData(CoilSet, MachineDescription):
    """Manage access to machine data."""

    shot: int = None
    run: int = None
    ids_name: str = None

    def __post_init__(self):
        """Build geometry."""
        super().__post_init__()
        self.build()

    def load_ids_data(self):
        """Return ids_data."""
        return self.ids(self.shot, self.run, self.ids_name)

    @abstractmethod
    def build(self):
        """Build geometry."""


@dataclass
class Passive(MachineData):
    """Manage passive poloidal loop ids, pf_passive."""

    shot: int = 115005
    run: int = 2
    ids_name: str = 'pf_passive'

    def build(self):
        """Build pf passive geometroy."""
        shelldata = PassiveShellData(self.dshell)
        coildata = PassiveCoilData(self.dcoil)
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
class Active(MachineData):
    """Manage active poloidal loop ids, pf_passive."""

    shot: int = 111001
    run: int = 1
    ids_name: str = 'pf_active'

    def build(self):
        """Build pf active geometroy."""
        coildata = ActiveCoilData(self.dcoil)
        for ids_loop in self.load_ids_data().coil:
            loop = ActiveLoop(ids_loop)
            for i, ids_element in enumerate(ids_loop.element):
                element = Element(ids_element, i)
                if element.is_rectangular():
                    coildata.append(loop, element)
                    continue
                raise NotImplementedError(f'geometory {element.geometry.name} '
                                          'not implemented')
        coildata.insert(self.coil)


@dataclass
class Machine(CoilSet):
    """Manage machine geometry."""


if __name__ == '__main__':

    passive = Passive(dshell=0.25)
    #passive.plot()

    active = Active(dcoil=0.25, tcoil='hex')
    #active.plot()

    active += passive
    active.plot()

    #coilset = passive + active
    #coilset.plot()

    #loop = Loop(pf_passive.loop)
    #loop.frameset.plot()

    #pf_passive = MachineDescription().ids(115005, 2, 'pf_passive')
    #pf_active = MachineDescription().ids(111001, 1, 'pf_active')
    #el = Element(pf_passive.loop[1].element[0])
    #el.geom.plot()

    #element = pf_passive.loop[1].element[0]
    #geom = Geometry(element.geometry)
