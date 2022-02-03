"""Manage access to IMAS machine data."""
from abc import ABC, abstractmethod
from contextlib import contextmanager
from dataclasses import dataclass, field
import string
from typing import ClassVar, Union

import numpy as np
import numpy.typing as npt
import shapely

import imas
from nova.electromagnetic.coil import Coil
from nova.electromagnetic.coilset import CoilSet
from nova.electromagnetic.shell import Shell
from nova.geometry.polygon import Polygon
from nova.utilities.pyplot import plt


@dataclass
class MachineDescription:
    """Methods to access IMAS machine description data."""

    user: str = 'public'
    tokamak: str = 'iter_md'
    backend: int = imas.imasdef.MDSPLUS_BACKEND

    @contextmanager
    def database(self, shot: int, run: int, ids_name: str):
        """Database context manager."""
        database = imas.DBEntry(self.backend, self.tokamak,
                                shot, run, user_name=self.user)
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

    def __post_init__(self):
        """Build oblique geometory."""
        super().__post_init__()

    @property
    def poly(self):
        """Return shapely polygon."""
        return Polygon([self.data['r'], self.data['z']]).poly


@dataclass
class Rectangle(GeomData):
    """Rectangular poloidal patch."""

    name: str = 'rectangle'
    attrs: ClassVar[list[str]] = ['r', 'z', 'width', 'height']

    def __post_init__(self):
        """Build oblique geometory."""
        super().__post_init__()

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

    def __post_init__(self):
        """Build oblique geometory."""
        super().__post_init__()

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

    def __post_init__(self):
        """Build oblique geometory."""
        super().__post_init__()

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


class FrameData(ABC):
    """Frame data base class."""

    @abstractmethod
    def append(self, loop: Loop, element: Element):
        """Append data to internal structures."""

    @abstractmethod
    def insert(self, method: object):
        """Insert geometry into frameset."""

    def get_part(self, name: str):
        """Return part name."""
        if 'VES' in name:
            return 'vv'
        if 'TRI' in name:
            return 'trs'
        return 'shell'


@dataclass
class ShellData(FrameData):
    """Extract oblique shell geometries from pf_passive ids."""

    delta: float
    length: float = 0
    points: list[npt.ArrayLike] = field(init=False, repr=False,
                                        default_factory=list)
    name: list[list[float]] = field(init=False, repr=False,
                                    default_factory=list)
    thickness: list[list[float]] = field(init=False, repr=False,
                                         default_factory=list)
    resistance: list[list[float]] = field(init=False, repr=False,
                                          default_factory=list)

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
        self.name.append([loop.name])
        self.thickness.append([geometry.thickness])
        self.resistance.append([loop.resistance])

    def _end(self, loop: Loop, element: Element):
        """Append endpoint to current loop."""
        geometry = element.geometry
        self.points[-1] = np.append(
            self.points[-1], geometry.end.reshape(1, -1), axis=0)
        self.name[-1].append(loop.name)
        self.thickness[-1].append(geometry.thickness)
        self.resistance[-1].append(loop.resistance)

    def insert(self, shell: Shell):
        """Insert data into shell instance."""
        for i in range(len(self)):
            thickness = np.mean(self.thickness[i])
            part = self.get_part(self.name[i][0])
            index = shell.insert(*self.points[i].T, self.length, thickness,
                                 rho=0, delta=self.delta, name=self.name[i],
                                 part=part)
            rho = self.resistance[i] * shell.frame.loc[index, 'area'] / \
                shell.frame.loc[index, 'dy']
            shell.frame.loc[index, 'rho'] = rho
            for j, frame in enumerate(index):
                subindex = shell.subframe.frame == frame
                shell.subframe.loc[subindex, 'rho'] = rho[j]

    def plot(self):
        """Plot shell centerlines."""
        axes = plt.gca()
        for loop in self.points:
            axes.plot(loop[:, 0], loop[:, 1], 'o-')
        plt.axis('equal')
        plt.axis('off')


@dataclass
class CoilData(FrameData):
    """Extract coildata from passive ids."""

    def append(self, loop: Loop, element: Element):
        """Append coil data to internal structrue."""
        assert element.is_rectangular()
        # TODO finish creating append / insert structure for passive coils.

    def insert(self, coil: Coil):
        """Insert data via coil method."""
        coil.insert(self.radius, self.height, self.width, self.thickness)



@dataclass
class MachineData(CoilSet, MachineDescription):
    """Manage access to machine data."""

    shot: int = None
    run: int = None
    ids_name: str = None

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

    #ids_data: object = field(repr=False)
    #shell: ShellLoop = field(init=False, default_factory=ShellLoop)

    def __post_init__(self):
        """Load data from cache - build if not found"""
        super().__post_init__()

    def build(self):
        """Build pf passive geometroy."""
        shelldata = ShellData(self.dshell)
        coildata = CoilData(self.dcoil)
        for ids_loop in self.load_ids_data().loop:
            loop = Loop(ids_loop)
            for i, ids_element in enumerate(ids_loop.element):
                element = Element(ids_element, i)
                if element.is_oblique():
                    shelldata.append(loop, element)
                    continue
                if element.is_rectangular():
                    # TO DO implement rectanglar build
                    coildata.append(loop, element)
                    continue

        shelldata.insert(self.shell)


@dataclass
class Machine(CoilSet):

    def pf_passive(self):
        """ """
        #ids = self.ids(shot, run, 'pf_passive')


if __name__ == '__main__':

    passive = Passive(dshell=0.25)
    passive.build()
    passive.plot()

    #loop = Loop(pf_passive.loop)
    #loop.frameset.plot()

    #pf_passive = MachineDescription().ids(115005, 2, 'pf_passive')
    #el = Element(pf_passive.loop[15].element[0])
    #el.geom.plot()

    #element = pf_passive.loop[1].element[0]
    #geom = Geometry(element.geometry)


'''
ids = imas.DBEntry(imas.imasdef.MDSPLUS_BACKEND,
                   'iter_md', 115005, 2, user_name='public')
ids.open()
vessel = ids.get('pf_passive', 0)
ids.close()

coils = [coil.identifier for coil in pf_active.coil.array]





def get_part(name: str) -> str:
    if (prefix := name[:2].lower()) in ['cs', 'pf']:
        return prefix
    return 'coil'

coilset = CoilSet(dcoil=-15)

for coil in pf_active.coil.array[:13]:
    for element in coil.element:
        nturn = element.turns_with_sign
        poly = load_geometory(element.geometry)
        name = element.identifier
        if name == '':
            name = coil.identifier
        part = get_part(name)
        coilset.coil.insert(poly, nturn=nturn, name=name, part=part)

for coil in vessel.loop:
    for element in coil.element:
        nturn = element.turns_with_sign
        poly = load_geometory(element.geometry)
        name = element.name
        if name == '':
            name = coil.identifier
        part = get_part(name)
        coilset.coil.insert(poly, nturn=nturn, name=name, part=part,
                            delta=1)


coilset.plot()
'''
