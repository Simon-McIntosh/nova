"""Manage access to IMAS machine data."""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import string
from typing import ClassVar

import numpy as np
import numpy.typing as npt
import shapely

from nova.electromagnetic.coilset import CoilSet
from nova.electromagnetic.shell import Shell
from nova.geometry.polygon import Polygon
from nova.imas.database import CoilData, Database, Ids
from nova.utilities.pyplot import plt


# pylint: disable=too-many-ancestors


@dataclass
class GeomData:
    """Geometry data baseclass."""

    ids: object = field(repr=False)
    data: dict[str, int | float] = field(init=False, repr=False,
                                         default_factory=dict)
    attrs: ClassVar[list[str]] = []

    def __post_init__(self):
        """Extract attributes from ids."""
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
        """Extract attributes from ids and store in data."""
        data = getattr(self.ids, self.name)
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
class Annulus(GeomData):
    """Annulus patch."""

    name: str = 'annulus'
    attrs: ClassVar[list[str]] = ['r', 'z', 'radius_inner', 'radius_outer']

    def __post_init__(self):
        """Caclulate derived attributes."""
        super().__post_init__()
        self.data['width'] = self.data['height'] = 2*self.data['radius_outer']
        self.data['factor'] = \
            1 - self.data['radius_inner'] / self.data['radius_outer']

    @property
    def poly(self):
        """Return shapely polygon."""
        return Polygon({'skin': [self.data['r'], self.data['z'],
                                 self.data['width'],
                                 self.data['factor']]}).poly


@dataclass
class Geometry:
    """Manage poloidal cross-sections."""

    ids: object = field(repr=False)
    data: GeomData = field(init=False)
    transform: ClassVar[dict[int, object]] = \
        {1: Outline, 2: Rectangle, 3: Oblique, 4: Arcs, 5: Annulus}

    def __post_init__(self):
        """Build geometry instance."""
        self.data = self.transform[self.ids.geometry_type](self.ids)

    def __getattr__(self, attr):
        """Return data attributes."""
        return getattr(self.data, attr)


@dataclass
class Loop:
    """Poloidal loop."""

    ids: object = field(repr=False)
    name: str = field(init=False)
    label: str = field(init=False)
    resistance: float = field(init=False)

    def __post_init__(self):
        """Extract data from loop ids."""
        self.name = self.ids.name.strip()
        self.label = self.name.rstrip(string.digits + '_')
        self.resistance = self.ids.resistance


@dataclass
class ActiveLoop(Loop):
    """Poloidal coil."""

    identifier: str = field(init=False)

    def __post_init__(self):
        """Extract data from loop ids."""
        super().__post_init__()
        self.identifier = self.ids.identifier


@dataclass
class Element:
    """Poloidal element."""

    ids: object = field(repr=False)
    index: int = 0
    name: str = field(init=False)
    nturn: float = field(init=False)
    geometry: Geometry = field(init=False)

    def __post_init__(self):
        """Extract element data from ids."""
        self.name = self.ids.name.strip()
        self.nturn = self.ids.turns_with_sign
        self.geometry = Geometry(self.ids.geometry)

    def is_poly(self) -> bool:
        """Return True if geometry.name == 'oblique' or 'annulus'."""
        return self.geometry.name in ['oblique', 'annulus']

    def is_rectangular(self) -> bool:
        """Return geometry.name == 'rectangle'."""
        return self.geometry.name == 'rectangle'

    def is_oblique(self) -> bool:
        """Return geometry.name == 'oblique'."""
        return self.geometry.name == 'oblique'

    def is_point(self) -> bool:
        """Return geometry validity flag."""
        return np.isclose(self.geometry.data.poly.area, 0)


@dataclass
class FrameData(ABC):
    """Frame data base class."""

    data: dict[str, list[list[float]]] = field(init=False)
    element_attrs: ClassVar[list[str]] = []
    geometry_attrs: ClassVar[list[str]] = []
    loop_attrs: ClassVar[list[str]] = []

    def __post_init__(self):
        """Init data dict."""
        self.data = {attr: [] for attr in self.attrs}

    @property
    def empty(self) -> bool:
        """Return empty boolean."""
        return len(self.data[self.attrs[0]]) == 0

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
            self.data[attr] = getattr(loop, attr)

    @property
    def part(self):
        """Return part name."""
        try:
            label = self.data['identifier']
        except KeyError:
            label = self.data['name']
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
class IdsCoilData(FrameData):
    """Extract coildata from ids."""

    name: list[str] = field(init=False, default_factory=list)
    geometry_attrs: ClassVar[list[str]] = ['r', 'z', 'width', 'height']
    loop_attrs: ClassVar[list[str]] = ['identifier', 'resistance']

    def append(self, loop: Loop, element: Element):
        """Append coil data to internal structrue."""
        super().append(loop, element)

    def insert(self, constructor, **kwargs):
        """Insert data via Coilset.constructor method."""
        attrs = kwargs.pop('attrs', self.geometry_attrs)
        index = constructor.insert(*[self.data[attr] for attr in attrs],
                                   part=self.part, rho=0, **kwargs)
        self.update_resistivity(
            index, *constructor.frames, self.data['resistance'])
        super().__post_init__()
        return index


@dataclass
class PassiveShellData(FrameData):
    """Extract oblique shell geometries from pf_passive ids."""

    length: float = 0
    points: list[np.ndarray] = field(init=False, repr=False,
                                     default_factory=list)
    loop_attrs: ClassVar[list[str]] = ['name', 'resistance']
    geometry_attrs: ClassVar[list[str]] = ['thickness']

    def reset(self):
        """Reset instance state."""
        self.__init__()

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
            self.data[attr].append(getattr(loop, attr))
        for attr in self.geometry_attrs:
            self.data[attr].append([getattr(geometry, attr)])

    def _end(self, loop: Loop, element: Element):
        """Append endpoint to current loop."""
        geometry = element.geometry
        self.points[-1] = np.append(
            self.points[-1], geometry.end.reshape(1, -1), axis=0)
        for attr in self.geometry_attrs:
            self.data[attr][-1].append(getattr(loop, attr))

    def insert(self, shell: Shell):
        """Insert data into shell instance."""
        if self.empty:
            return
        for i in range(len(self)):
            thickness = np.mean(self.data['thickness'][i])
            index = shell.insert(*self.points[i].T, self.length, thickness,
                                 rho=0, name=self.data['name'][i],
                                 part=self.part)
            self.update_resistivity(index, shell.frame, shell.subframe,
                                    self.data['resistance'][i])
        self.reset()

    def plot(self):
        """Plot shell centerlines."""
        axes = plt.gca()
        for loop in self.points:
            axes.plot(loop[:, 0], loop[:, 1], 'o-')
        plt.axis('equal')
        plt.axis('off')


@dataclass
class PassiveCoilData(IdsCoilData):
    """Extract coildata from passive ids."""

    geometry_attrs: ClassVar[list[str]] = ['r', 'z', 'width', 'height']
    loop_attrs: ClassVar[list[str]] = ['name', 'resistance']

    def insert(self, constructor, **kwargs):
        """Insert data via coil method."""
        if self.empty:
            return
        kwargs = {'active': False, 'name': self.data['name'],
                  'turn': 'rect'} | kwargs
        return super().insert(constructor, **kwargs)


@dataclass
class CoilDatabase(CoilSet, CoilData, Database):
    """Manage coilset construction from ids structures."""

    machine: str = 'iter_md'

    @property
    def group_attrs(self) -> dict:
        """
        Return group attrs.

        Extends :func:`~nova.imas.database.CoilData`.
        """
        return self.coilset_attrs | self.ids_attrs


@dataclass
class PoloidalFieldPassive(CoilDatabase):
    """Manage passive poloidal loop ids, pf_passive."""

    pulse: int = 115005
    run: int = 2
    name: str = 'pf_passive'

    def build(self):
        """Build pf passive geometroy."""
        shelldata = PassiveShellData()
        coildata = PassiveCoilData()
        for ids_loop in getattr(self.ids, 'loop'):
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

            coildata.insert(self.coil, delta=self.dshell)
            shelldata.insert(self.shell)


@dataclass
class ActiveCoilData(IdsCoilData):
    """Extract coildata from active ids."""

    element_attrs: ClassVar[list[str]] = ['nturn', 'index', 'name']
    geometry_attrs: ClassVar[list[str]] = ['r', 'z', 'width', 'height']
    loop_attrs: ClassVar[list[str]] = ['identifier', 'resistance']

    def insert(self, constructor, **kwargs):
        """Insert data via coil method."""
        if self.empty:
            return
        self.data['nturn'] = np.abs(self.data['nturn'])
        kwargs = {'active': True, 'fix': False,
                  'name': self.data['identifier'],
                  'delim': '_', 'nturn': self.data['nturn'],
                  } | kwargs
        return super().insert(constructor, **kwargs)


@dataclass
class ActivePolyCoilData(ActiveCoilData):
    """Extract coildata from active ids."""

    geometry_attrs: ClassVar[list[str]] = ['poly']


@dataclass
class PoloidalFieldActive(CoilDatabase):
    """Manage active poloidal loop ids, pf_passive."""

    pulse: int = 111001
    run: int = 202
    name: str = 'pf_active'

    def build(self):
        """Build pf active."""
        self.build_coil()
        self.build_circuit()

    def build_coil(self):
        """Build pf active coil geometroy."""
        for ids_loop in getattr(self.ids, 'coil'):
            loop = ActiveLoop(ids_loop)
            coildata = ActiveCoilData()
            polydata = ActivePolyCoilData()
            for i, ids_element in enumerate(ids_loop.element):
                element = Element(ids_element, i)
                if element.is_point():
                    continue
                if element.is_rectangular():
                    coildata.append(loop, element)
                    continue
                if element.is_poly():
                    polydata.append(loop, element)
                    continue
                raise NotImplementedError(f'geometory {element.geometry.name} '
                                          'not implemented')
            if i == 0:
                constructor = self.coil
            else:
                constructor = self.turn
            coildata.insert(constructor)
            polydata.insert(constructor)

    def build_circuit(self):
        """Build circuit influence matrix."""
        supply = [supply.identifier
                  for supply in getattr(self.ids, 'supply')]
        nodes = max([len(circuit.connections)
                     for circuit in getattr(self.ids, 'circuit')])
        self.circuit.initialize(supply, nodes)
        for circuit in getattr(self.ids, 'circuit'):
            self.circuit.insert(circuit.identifier, circuit.connections)
        self.circuit.link()  # link single loop circuits


@dataclass
class ContourData:
    """Extract contour data from ids."""

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
    loop: np.ndarray = field(init=False, default_factory=lambda:
                             np.ndarray((0, 2), float))
    segments: list[np.ndarray] = field(init=False)

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
class Wall(CoilDatabase):
    """Manage plasma boundary, wall ids."""

    pulse: int = 116000
    run: int = 2
    name: str = 'wall'

    def build(self):
        """Build plasma bound by firstwall contour."""
        firstwall = ContourData()
        limiter = getattr(self.ids, 'description_2d').array[0].limiter
        for unit in limiter.unit:
            firstwall.append(unit)
        contour = Contour(firstwall.data)  # extract closed loop
        self.firstwall.insert(contour.loop)


@dataclass
class MachineGeometry:
    """
    Manage IDS geometry attributes.

    Parameters
    ----------
    pf_active: Ids | bool, optional
        pf active IDS. The default is True
    pf_passive: Ids | bool, optional
        pf passive IDS. The default is True
    wall: Ids | bool, optional
        wall IDS. The default is True


    Examples
    --------
    Dissable wall geometry via boolean input:

    >>> geometry = MachineGeometry(wall=False)
    >>> geometry.wall
    False
    >>> geometry.pf_active == PoloidalFieldActive.default_ids_attrs()
    True
    >>> geometry.pf_passive == PoloidalFieldPassive.default_ids_attrs()
    True

    Modify pf_active attrs via dict input:

    >>> pf_active = MachineGeometry(pf_active=dict(run=101)).pf_active
    >>> pf_active == PoloidalFieldActive.default_ids_attrs() | dict(run=101)
    True

    Specify pf_active as an ids:

    >>> database = Database(111001, 202, 'pf_active', machine='iter_md')
    >>> pf_active = MachineGeometry(database.ids).pf_active
    >>> pf_active['run']
    1072318551

    Specify pf_active as an itterable:

    >>> pf_active = MachineGeometry(pf_active=(111001, 202)).pf_active
    >>> tuple(pf_active[attr] for attr in ['pulse', 'run', 'name', 'machine'])
    (111001, 202, 'pf_active', 'iter_md')

    """

    pf_active: Ids | bool = True
    pf_passive: Ids | bool = True
    wall: Ids | bool = True

    geometry: ClassVar[dict] = dict(pf_active=PoloidalFieldActive,
                                    pf_passive=PoloidalFieldPassive,
                                    wall=Wall)

    def __post_init__(self):
        """Map geometry parameters to dict attributes."""
        for attr in self.geometry:
            attrs = self.geometry[attr].update_ids_attrs(getattr(self, attr))
            setattr(self, attr, attrs)
        if hasattr(super(), '__post_init__'):
            super().__post_init__()

    @property
    def geometry_attrs(self) -> dict:
        """Return geometry attributes."""
        return {attr: getattr(self, attr) for attr in self.geometry}


@dataclass
class Machine(CoilSet, MachineGeometry, CoilData):
    """Manage ITER machine geometry."""

    dcoil: float = -1
    dshell: float = 0.5
    nplasma: float = 1000
    tcoil: str = 'rectangle'
    tplasma: str = 'rectangle'

    filename: str = 'iter'

    @property
    def metadata(self):
        """Manage machine metadata.

        Raises
        ------
            AssertionError:
                A change in the metadata group hash has been detected.
        """
        metadata = self.coilset_attrs
        for geometry in self.geometry:
            if (attrs := self.geometry_attrs[geometry]) is False:
                continue
            metadata[geometry] = ','.join([str(attrs[attr]) for attr in attrs])
        return metadata

    @metadata.setter
    def metadata(self, metadata: dict):
        """Set instance metadata, assert consistent attr_hash."""
        attr_hash = self.hash_attrs(self.group_attrs)
        for attr in self.coilset_attrs:
            setattr(self, attr, metadata[attr])
        for geometry in self.geometry:
            if geometry not in metadata:
                setattr(self, geometry, False)
                continue
            values = [self._format_geometry_attrs(attr)
                      for attr in metadata[geometry].split(',')]
            setattr(self, geometry, dict(zip(Database.attrs, values)))
        assert attr_hash == self.hash_attrs(self.group_attrs)

    @staticmethod
    def _format_geometry_attrs(attr: str) -> str | int | float:
        """Return formated attr. Try int conversion except return str."""
        if '.' in attr:
            return float(attr)
        try:
            return int(attr)
        except ValueError:
            return attr

    @property
    def group_attrs(self) -> dict:
        """
        Return group attrs.

        Extends :func:`~nova.imas.database.CoilData.group_attrs`.
        """
        return self.coilset_attrs | self.geometry_attrs

    def solve_biot(self):
        """Solve biot instances."""
        if self.sloc['plasma'].sum() > 0:
            self.plasmaboundary.solve(self.Loc['plasma', 'poly'][0].boundary)
            self.plasmagrid.solve()

    def build(self, **kwargs):
        """Build dataset, frameset and, biotset and save to file."""
        self.frame_attrs = kwargs
        self.clear_frameset()
        for attr in self.geometry:
            if (geometry := getattr(self, attr)):
                coilset = self.geometry[attr](**geometry, **self.frame_attrs)
                self += coilset
            for attr in coilset.biot_methods:
                getattr(self, attr).data = getattr(coilset, attr).data
        self.solve_biot()
        return self.store()

    def load(self, filename=None, path=None):
        """Load machine geometry and data."""
        super().load(filename, path)
        self.metadata = self.load_metadata(filename, path)
        return self

    def store(self, filename=None, path=None, metadata=None):
        """Store frameset, biot attributes and metadata."""
        super().store(filename, path)
        self.store_metadata(filename, path, self.metadata)
        return self


if __name__ == '__main__':

    import doctest
    doctest.testmod()

    machine = Machine(pf_passive=False, nplasma=500)

    machine.sloc['Ic'] = 1
    machine.sloc['plasma', 'Ic'] = -10000
    machine.plot()
    machine.plasmagrid.plot()

    #coilset.plot()
    #coilset.circuit.plot('CS1')

    # coilset.plasma.separatrix = dict(e=[6, -0.5, 2.5, 2.5])
    # coilset.sloc['Ic'] = 1
    # coilset.sloc['plasma', 'Ic'] = -450
    # coilset.plot()
