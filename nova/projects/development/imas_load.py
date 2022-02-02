"""Manage access to IMAS machine data."""
from abc import ABC, abstractmethod
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import ClassVar, Union

import numpy as np
import numpy.typing as npt
import shapely

import imas
from nova.electromagnetic.coil import Coil
from nova.electromagnetic.frameset import FrameSet
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
class Element:
    """Poloidal element."""

    ids_data: object = field(repr=False)
    index: int = 0
    name: str = field(init=False)
    nturn: float = field(init=False)
    geom: Geometry = field(init=False)

    def __post_init__(self):
        """Extract element data from ids."""
        self.name = self.ids_data.name.strip()
        self.nturn = self.ids_data.turns_with_sign
        self.geom = Geometry(self.ids_data.geometry)


@dataclass
class ShellGeom:
    """Extract shell geometries from pf_passive ids."""

    number: int = field(init=False, default=0)
    points: list[npt.ArrayLike] = field(init=False, repr=False,
                                        default_factory=list)
    thickness: list[npt.ArrayLike] = field(init=False, repr=False,
                                           default_factory=list)

    def append(self, start, end, thickness):
        """Check start/end point colocation."""
        if not self.points:
            return self._new(start, end, thickness)
        if np.allclose(self.points[-1][-1], start):
            return self._end(end, thickness)
        return self._new(start, end, thickness)

    def _new(self, start, end, thickness):
        """Start new loop."""
        self.number += 1
        self.points.append(np.c_[start, end].T)
        self.thickness.append(thickness * np.ones(2))

    def _end(self, end, thickness):
        """Append endpoint to current loop."""
        self.points[-1] = np.append(self.points[-1], end.reshape(1, -1),
                                    axis=0)
        self.thickness[-1] = np.append(self.thickness[-1], thickness)

    def plot(self):
        """Plot shell centerlines."""
        axes = plt.gca()
        for loop in self.points:
            axes.plot(loop[:, 0], loop[:, 1], 'o-')
        plt.axis('equal')
        plt.axis('off')


@dataclass
class Loop:
    """Manage passive poloidal loop ids, pf_passive."""

    ids_data: object = field(repr=False)
    frameset: FrameSet = field(init=False, default_factory=FrameSet)

    def __post_init__(self):
        coil_geom = []
        shellgeom = ShellGeom()
        #coil = Coil(*self.frameset.frames)
        shell = Shell(*self.frameset.frames)
        for loop in self.ids_data:
            for i, element in enumerate(loop.element):
                elem = Element(element, i)
                if elem.geom.name == 'oblique':
                    shellgeom.append(*elem.geom.points, elem.geom.thickness)
                else:
                    coil_geom.append(elem)
        for i in range(shellgeom.number):
            thickness = np.mean(shellgeom.thickness[i])
            shell.insert(*shellgeom.points[i].T, 0.5, thickness, delta=-3)
            #coil.insert(Element(loop.element[0]).poly, delta=1)



@dataclass
class Machine:


    def pf_passive(self):
        """ """
        #ids = self.ids(shot, run, 'pf_passive')


if __name__ == '__main__':

    md = MachineDescription()
    pf_passive = md.ids(115005, 2, 'pf_passive')

    loop = Loop(pf_passive.loop)

    loop.frameset.plot()
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
