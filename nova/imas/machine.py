"""Manage access to IMAS machine data."""

from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from functools import cached_property
from importlib import import_module
import itertools
import string
from typing import ClassVar, TYPE_CHECKING
from warnings import warn

import numpy as np
import xarray

from nova.graphics.plot import Plot
from nova.frame.coilset import CoilSet
from nova.imas.database import CoilData
from nova.imas.dataset import IdsBase, Ids, ImasIds, EMPTY_FLOAT
from nova.imas.ids_index import IdsIndex
from nova.geometry.polygon import Polygon

if TYPE_CHECKING:
    from nova.frame.shell import Shell

# pylint: disable=too-many-ancestors


@dataclass
class GeomData:
    """Geometry data baseclass."""

    ids: ImasIds = field(repr=False)
    data: dict[str, int | float] = field(init=False, repr=False, default_factory=dict)
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
    def poly(self):
        """Return patch polygon."""

    def extract(self):
        """Extract attributes from ids and store in data."""
        data = getattr(self.ids, self.name)
        for attr in self.attrs:
            self.data[attr] = getattr(data, attr).value

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

    name: str = "outline"
    attrs: ClassVar[list[str]] = ["r", "z"]

    @property
    def poly(self):
        """Return shapely polygon."""
        return Polygon([self.data["r"], self.data["z"]], metadata=self.data).poly


@dataclass
class Rectangle(GeomData):
    """Rectangular poloidal patch."""

    name: str = "rectangle"
    attrs: ClassVar[list[str]] = ["r", "z", "width", "height"]

    def __post_init__(self):
        """Enforce positive dimensions."""
        super().__post_init__()
        for attr in [
            "width",
            "height",
        ]:  # TODO remove negative check once MastU IDS fixed
            if self.data[attr] <= 0:
                self.data[attr] *= -1
                warn(
                    "negative width or height "
                    f'{self.data["width"], self.data["height"]}'
                )

    @property
    def poly(self):
        """Return shapely polygon."""
        return Polygon(
            {
                "r": [
                    self.data["r"],
                    self.data["z"],
                    self.data["width"],
                    self.data["height"],
                ]
            },
            metadata=self.data,
        ).poly


@dataclass
class Oblique(Plot, GeomData):
    """Oblique poloidal patch (parallelogram)."""

    name: str = "oblique"
    attrs: ClassVar[list[str]] = [
        "r",
        "z",
        "length_alpha",
        "length_beta",
        "alpha",
        "beta",
    ]

    @property
    def poly(self):
        """Return skewed shapely polygon."""
        radius = self.r + np.array(
            [
                0,
                self.length_alpha * np.cos(self.alpha),
                self.length_alpha * np.cos(self.alpha)
                - self.length_beta * np.sin(self.beta),
                -self.length_beta * np.sin(self.beta),
            ]
        )
        height = self.z + np.array(
            [
                0,
                self.length_alpha * np.sin(self.alpha),
                self.length_alpha * np.sin(self.alpha)
                + self.length_beta * np.cos(self.beta),
                self.length_beta * np.cos(self.beta),
            ]
        )
        return Polygon([radius, height], metadata=self.data).poly

    @property
    def start(self):
        """Return oblique geometry start point."""
        if self.length_alpha > self.length_beta:
            return np.array(
                [
                    self.r - self.length_beta / 2 * np.sin(self.beta),
                    self.z + self.length_beta / 2 * np.cos(self.beta),
                ]
            )
        return np.array(
            [
                self.r + self.length_alpha / 2 * np.cos(self.alpha),
                self.z + self.length_alpha / 2 * np.sin(self.alpha),
            ]
        )

    @property
    def end(self):
        """Return oblique geometry end point."""
        if self.length_alpha > self.length_beta:
            return self.start + np.array(
                [
                    self.length_alpha * np.cos(self.alpha),
                    self.length_alpha * np.sin(self.alpha),
                ]
            )
        return self.start + np.array(
            [
                -self.length_beta * np.sin(self.beta),
                self.length_beta * np.cos(self.beta),
            ]
        )

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

    def plot(self, axes=None):
        """Plot oblique patch verticies and start/end points."""
        self.set_axes("2d", axes=axes)
        self.axes.plot(*self.poly.boundary.xy, "o", label="vertex")
        self.axes.plot(*self.start, "C1o", label="start")
        self.axes.plot(*self.end, "C3o", label="end")
        self.axes.legend()


@dataclass
class Arcs(GeomData):
    """Polygonal poloidal patch."""

    name: str = "arcs"
    attrs: ClassVar[list[str]] = []

    @property
    def poly(self):
        """Return shapely polygon."""
        raise NotImplementedError


@dataclass
class Annulus(GeomData):
    """Annulus patch."""

    name: str = "annulus"
    attrs: ClassVar[list[str]] = ["r", "z", "radius_inner", "radius_outer"]

    def __post_init__(self):
        """Caclulate derived attributes."""
        super().__post_init__()
        self.data["width"] = self.data["height"] = 2 * self.data["radius_outer"]
        self.data["factor"] = 1 - self.data["radius_inner"] / self.data["radius_outer"]

    @property
    def poly(self):
        """Return shapely polygon."""
        return Polygon(
            {
                "skin": [
                    self.data["r"],
                    self.data["z"],
                    self.data["width"],
                    self.data["factor"],
                ]
            },
            metadata=self.data,
        ).poly


@dataclass
class ThickLine(GeomData):
    """Thick line patch."""

    name: str = "thick_line"
    attrs: ClassVar[list[str]] = ["thickness"]

    def extract(self):
        """Extend GeomData.extract."""
        super().extract()
        data = getattr(self.ids, self.name)
        for attr, label in zip(["start", "end"], ["first", "second"]):
            point = getattr(data, f"{label}_point")
            self.data[attr] = np.array([point.r.value, point.z.value])


@dataclass
class CrossSection:
    """Manage poloidal cross-sections."""

    ids: ImasIds = field(repr=False)
    data: GeomData = field(init=False)
    transform: ClassVar[dict[int, object]] = {
        1: Outline,
        2: Rectangle,
        3: Oblique,
        4: Arcs,
        5: Annulus,
        6: ThickLine,
    }

    def __post_init__(self):
        """Build geometry instance."""
        self.data = self.transform[self.ids.geometry_type.value](self.ids)
        for attr in self.data.data:
            if isinstance(self.data.data[attr], float) and not np.isfinite(
                self.data.data[attr]
            ):
                self.data.data[attr] = 0.1  # TODO remove once MastU IDS is fixed

    def __getattr__(self, attr):
        """Return data attributes."""
        return getattr(self.data, attr)


@dataclass
class Loop:
    """Poloidal loop."""

    ids: ImasIds = field(repr=False)
    name: str = field(init=False)
    label: str = field(init=False)
    resistance: float = field(init=False)

    def __post_init__(self):
        """Extract data from loop ids."""
        self.name = self.ids.name.value.strip()
        self.label = self.name.rstrip(string.digits + "_")
        self.resistance = self.ids.resistance.value


@dataclass
class ActiveLoop(Loop):
    """Poloidal coil."""

    identifier: str = field(init=False)

    def __post_init__(self):
        """Extract data from loop ids."""
        super().__post_init__()
        self.identifier = self.ids.identifier.value


@dataclass
class Element:
    """Poloidal element."""

    ids: ImasIds = field(repr=False)
    index: int = 0
    name: str = field(init=False)
    identifier: str = field(init=False)
    nturn: float = field(init=False)
    cross_section: CrossSection = field(init=False)

    def __post_init__(self):
        """Extract element data from ids."""
        self.name = self.ids.name.value.strip()
        self.identifier = self.ids.identifier.value.strip()
        self.nturn = self.ids.turns_with_sign.value
        if np.isclose(self.nturn, EMPTY_FLOAT):
            warn("nturn unset, setting turn number equal to one")
            self.nturn = 1
        self.cross_section = CrossSection(self.ids.geometry)

    @property
    def section(self):
        """Return section name."""
        return self.cross_section.name

    def is_poly(self) -> bool:
        """Return True if geometry.name == 'oblique' or 'annulus'."""
        return self.section in ["oblique", "annulus"]

    def is_rectangular(self) -> bool:
        """Return geometry.name == 'rectangle'."""
        return self.section == "rectangle"

    def is_oblique(self) -> bool:
        """Return geometry.name == 'oblique'."""
        return self.section == "oblique"

    def is_point(self) -> bool:
        """Return geometry validity flag."""
        return np.isclose(self.cross_section.data.poly.area, 0)

    def is_thickline(self) -> bool:
        """Return geometry.name == 'thick_line'."""
        return self.section == "thick_line"


@dataclass
class FrameData(ABC):
    """Frame data base class."""

    data: dict[str, list[list[float]]] = field(init=False)
    element_attrs: ClassVar[list[str]] = []
    geometry_attrs: ClassVar[list[str]] = []
    loop_attrs: ClassVar[list[str]] = []

    _count: ClassVar[itertools.count] = itertools.count(0)

    def __post_init__(self):
        """Init data dict."""
        self.data = {attr: [] for attr in self.attrs}

    @property
    def coil_name(self):
        """Return coil name."""
        identifier = self.data.get("identifier", "")
        if not isinstance(identifier, str):
            identifier = identifier[0]
        if identifier != "":
            return identifier
        name = self.data["name"]
        if not isinstance(name, str):
            name = name[0]
        if identifier == "" and name == "":
            name = f"_{next(self._count)}"
            return name  # TODO remove return and raise error once MastU ids fixed
            # raise ValueError("Nether name nor identifier set.")
        if len(name.split()) == 1:
            return name
        label = "".join(name.split()[:2]).rstrip(string.punctuation)
        digit = name.split()[-1].lstrip(string.ascii_letters + string.punctuation)
        return "".join([part for part in [label, digit] if part != ""])

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
            self.data[attr].append(getattr(element.cross_section, attr))
        for attr in self.loop_attrs:
            self.data[attr] = getattr(loop, attr)

    @property
    def part(self):
        """Return part name."""
        label = self.coil_name
        if isinstance(label, list):
            label = label[0]
        if "VES" in label or "VV" in label:
            return "vv"
        if "TRI" in label or "Tri" in label:
            return "trs"
        if label == "INB_RAIL" or "Div" in label:
            return "dir"
        if "CS" in label or "PF" in label:
            return label[:2].lower()
        if "SS" in label:
            return "vs3j"
        if "VS" in label:
            return "vs3"
        return ""

    @staticmethod
    def update_resistivity(index, frame, subframe, resistance):
        """Update frame and subframe resistivity."""
        rho = resistance * frame.loc[index, "area"] / frame.loc[index, "dy"]
        frame.loc[index, "rho"] = rho
        for name in index:
            subindex = subframe.frame == name
            subframe.loc[subindex, "rho"] = rho[name]


@dataclass
class IdsCoilData(FrameData):
    """Extract coildata from ids."""

    geometry_attrs: ClassVar[list[str]] = ["r", "z", "width", "height"]
    loop_attrs: ClassVar[list[str]] = ["identifier", "resistance"]

    def insert(self, constructor, **kwargs):
        """Insert data via Coilset.constructor method."""
        attrs = kwargs.pop("attrs", self.geometry_attrs)
        index = constructor.insert(
            *[self.data[attr] for attr in attrs], part=self.part, rho=0, **kwargs
        )
        self.update_resistivity(index, *constructor.frames, self.data["resistance"])
        super().__post_init__()
        return index


@dataclass
class PassiveShellData(Plot, FrameData):
    """Extract oblique shell geometries from pf_passive ids."""

    length: float = 0
    points: list[np.ndarray] = field(init=False, repr=False, default_factory=list)
    loop_attrs: ClassVar[list[str]] = ["name", "resistance"]
    geometry_attrs: ClassVar[list[str]] = ["thickness"]

    def reset(self):
        """Reset instance state."""
        self.length = 0
        self.points = []
        super().__init__()

    def __len__(self):
        """Return loop number."""
        return len(self.points)

    def append(self, loop: Loop, element: Element):
        """Check start/end point colocation."""
        assert element.is_oblique() or element.is_thickline()
        if not self.points:
            return self._new(loop, element)
        if np.allclose(self.points[-1][-1], element.cross_section.start):
            return self._end(loop, element)
        return self._new(loop, element)

    def _new(self, loop: Loop, element: Element):
        """Start new loop."""
        geometry = element.cross_section
        self.points.append(np.c_[geometry.start, geometry.end].T)
        for attr in self.loop_attrs:
            self.data[attr].append(getattr(loop, attr))
        for attr in self.geometry_attrs:
            self.data[attr].append([getattr(geometry, attr)])

    def _end(self, loop: Loop, element: Element):
        """Append endpoint to current loop."""
        geometry = element.cross_section
        self.points[-1] = np.append(
            self.points[-1], geometry.end.reshape(1, -1), axis=0
        )
        for attr in self.geometry_attrs:
            self.data[attr][-1].append(getattr(geometry, attr))

    def insert(self, shell: Shell):
        """Insert data into shell instance."""
        if self.empty:
            return
        for i in range(len(self)):
            thickness = np.mean(self.data["thickness"][i])
            index = shell.insert(
                *self.points[i].T,
                self.length,
                thickness,
                rho=0,
                name=self.coil_name,
                part=self.part,
            )
            self.update_resistivity(
                index, shell.frame, shell.subframe, self.data["resistance"][i]
            )
        self.reset()

    def plot(self, axes=None):
        """Plot shell centerlines."""
        self.set_axes("2d", axes=axes)
        for loop in self.points:
            self.axes.plot(loop[:, 0], loop[:, 1], "o-")


@dataclass
class PassiveCoilData(IdsCoilData):
    """Extract coildata from passive ids."""

    element_attrs: ClassVar[list[str]] = ["identifier", "section"]
    geometry_attrs: ClassVar[list[str]] = ["r", "z", "width", "height"]
    loop_attrs: ClassVar[list[str]] = ["name", "resistance"]

    def insert(self, constructor, **kwargs):
        """Insert data via coil method."""
        if self.empty:
            return None
        kwargs = {
            "active": False,
            "name": self.coil_name,
            "section": self.data["section"],
        } | kwargs
        return super().insert(constructor, **kwargs)


@dataclass
class PassivePolyCoilData(PassiveCoilData):
    """Extract coildata from passive ids."""

    geometry_attrs: ClassVar[list[str]] = ["poly"]


@dataclass
class CoilDatabase(CoilSet, CoilData):
    """Manage coilset construction from ids structures."""

    machine: str = "iter_md"
    ids_node: str = ""

    @cached_property
    def ids_index(self):
        """Return cached ids_index instance."""
        return IdsIndex(self.ids, self.ids_node)

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

    pulse: int = 115004  # 115005
    run: int = 5  # 2
    occurrence: int = 0
    name: str = "pf_passive"

    def build(self):
        """Build pf passive geometroy."""
        for ids_loop in getattr(self.ids, "loop"):
            loop = Loop(ids_loop)
            shelldata = PassiveShellData()
            coildata = PassiveCoilData()
            polydata = PassivePolyCoilData()
            for i, ids_element in enumerate(ids_loop.element):
                element = Element(ids_element, i)
                if element.is_thickline():  # element.is_oblique() or
                    shelldata.append(loop, element)
                    continue
                if element.is_rectangular():
                    coildata.append(loop, element)
                    continue
                if element.is_oblique() or element.is_poly():
                    polydata.append(loop, element)
                    continue
                raise NotImplementedError(
                    f"geometory {element.section} " "not implemented"
                )
            coildata.insert(self.coil, delta=-1)
            polydata.insert(self.coil, delta=-10)
            shelldata.insert(self.shell)


@dataclass
class ActiveCoilData(IdsCoilData):
    """Extract coildata from active ids."""

    element_attrs: ClassVar[list[str]] = ["nturn", "index", "name", "section"]
    geometry_attrs: ClassVar[list[str]] = ["r", "z", "width", "height"]
    loop_attrs: ClassVar[list[str]] = ["identifier", "resistance"]

    def insert(self, constructor, **kwargs):
        """Insert data via coil method."""
        if self.empty:
            return None
        self.data["nturn"] = self.data["nturn"]

        kwargs = {
            "active": True,
            "fix": False,
            "name": self.coil_name,
            "delim": "_",
            "nturn": self.data["nturn"],
            "section": self.data["section"],
        } | kwargs
        return super().insert(constructor, **kwargs)


@dataclass
class ActivePolyCoilData(ActiveCoilData):
    """Extract coildata from active ids."""

    geometry_attrs: ClassVar[list[str]] = ["poly"]


@dataclass
class PoloidalFieldActive(CoilDatabase):
    """Manage active poloidal loop ids, pf_passive."""

    pulse: int = 111001
    run: int = 203
    occurrence: int = 0
    name: str = "pf_active"

    def build(self):
        """Build pf active."""
        self.build_coil()
        self.build_circuit()

    def build_coil(self):
        """Build pf active coil geometry."""
        maximum_current = {}
        for ids_loop in getattr(self.ids, "coil"):
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
                raise NotImplementedError(
                    f"geometory {element.name} " "not implemented"
                )
            if len(ids_loop.element) == 1:
                constructor = self.coil
            else:
                constructor = self.turn
            coildata.insert(constructor)
            polydata.insert(constructor)
            current_limit_max = ids_loop.current_limit_max
            if len(current_limit_max) > 0:
                maximum_current[ids_loop.identifier.value] = current_limit_max[-1, 0]
        self.frame.loc[:, ["Imax"]] = maximum_current
        self.frame.loc[:, ["Imin"]] = {
            coil: -limit for coil, limit in maximum_current.items()
        }

    def build_circuit(self):
        """Build circuit influence matrix."""
        if len(self.ids.circuit) == 0:  # no circuit
            return
        supply = [supply.identifier.value for supply in self.ids.supply]
        nodes = max(len(circuit.connections) for circuit in self.ids.circuit)
        self.circuit.initialize(supply, nodes)
        for circuit in getattr(self.ids, "circuit"):
            if len(circuit.connections) == 0:
                continue
            self.circuit.insert(circuit.identifier.value, circuit.connections.value)

        self.circuit.link()  # link single loop circuits

        if len(self.ids.supply) == 0:  # no supplies
            return
        with self.ids_index.node("supply"):
            name = self.ids_index.array("identifier")
            if self.ids_index.empty("resistance"):
                resistance = np.zeros(len(name))
            try:
                resistance = self.ids_index.array("resistance")
            except ValueError:  # resistance field is empty
                resistance = np.zeros(len(name))
            self.supply.insert(resistance, name=name)

            for attr, label in zip(["I", "V"], ["current", "voltage"]):
                for minmax in ["min", "max"]:
                    supply = f"{attr}{minmax}"
                    node = f"{label}_limit_{minmax}"
                    try:
                        self.supply[supply] = self.ids_index.array(node)
                    except ValueError:  # node is empty
                        self.supply[supply] = 0


@dataclass
class ContourData(Plot):
    """Extract contour data from ids."""

    data: dict[str, np.ndarray] = field(init=False, default_factory=dict)

    def append(self, unit):
        """Append contour data."""
        self.data[unit.name.value] = np.array([unit.outline.r, unit.outline.z]).T

    def plot(self, axes=None):
        """Plot contours."""
        self.axes.set_axes("2d", axes=axes)
        for component in self.data.items():
            self.axes.plot(*self.data[component].T, label=component)
        self.axes.legend()


@dataclass
class Contour(Plot):
    """Extract closed contour from multiple unordered segments."""

    data: dict[str, np.ndarray]
    loop: np.ndarray = field(
        init=False, default_factory=lambda: np.ndarray((0, 2), float)
    )
    segments: list[np.ndarray] = field(init=False)

    def __post_init__(self):
        """Create segments list."""
        self.segments = list(self.data.values())
        self.loop = self.segments.pop(0)
        self.extract()

    def gap(self, index: int):
        """Return length of gap to next segment."""
        return [
            np.linalg.norm(segment[index] - self.loop[-1]) for segment in self.segments
        ]

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
        assert import_module("shapely.geometry").LinearRing(self.loop).is_valid

    def plot(self, axes=None):
        """Plot closed contour."""
        self.set_axes("2d", axes=axes)
        self.axes.plot(*self.loop.T, "C3-")


@dataclass
class Wall(CoilDatabase):
    """Manage plasma boundary, wall ids."""

    pulse: int = 116000
    run: int = 2
    name: str = "wall"
    occurrence: int = 0

    @cached_property
    def limiter(self):
        """Return limiter units."""
        return getattr(self.ids, "description_2d")[0].limiter

    @cached_property
    def boundary(self):
        """Return closed firstwall boundary contour."""
        firstwall = ContourData()
        for unit in self.limiter.unit:
            firstwall.append(unit)
        return Contour(firstwall.data).loop

    def segment(self, index=0):
        """Return indexed firstwall segment."""
        return np.array(
            [self.limiter.unit[index].outline.r, self.limiter.unit[index].outline.z]
        ).T

    @cached_property
    def segments(self):
        """Return first wall segments."""
        return [self.segment(i) for i in range(len(self.limiter.unit))]

    @cached_property
    def outline(self):
        """Return first wall xz outline."""
        return {
            "x": [segment[:, 0] for segment in self.segments],
            "z": [segment[:, 1] for segment in self.segments],
        }

    def build(self):
        """Build plasma bound by firstwall contour."""
        self.firstwall.insert(self.boundary)

    def insert(self, data: xarray.Dataset):
        """Insert wall and divertor geometory into dataset structure."""
        for i, attr in enumerate(["wall", "divertor"]):
            data[attr] = (f"{attr}_index", "point"), self.segment(i)
            index = np.arange(data[attr].shape[0])
            data.coords[f"{attr}_index"] = index
        data.attrs["wall_md"] = ",".join(
            [str(value) for _, value in self.ids_attrs.items()]
        )


@dataclass
class ELM(CoilDatabase):
    """Construct axisymetric ELM coils.

    Centerline data extracted from: Data_for_study_of_ITER_plasma_magnetic_33NHXN_v3.20

    """

    pulse: int = 0
    run: int = 0
    occurrence: int = 0
    name: str = "elm"
    mode: str | None = None

    def build(self):
        """Build axisymmetric elm coil geometry."""
        # lower ELM (feed / return)
        lower_nturn = 30 / 40 * 6  # 30 degrees per sector
        self.coil.insert(
            {"r": [7.8320, -2.4100, 0.0689, 0.122]},
            name="lELM",
            nturn=lower_nturn,
            active=False,
            part="elm",
        )
        self.coil.insert(
            {"r": [8.2280, -1.5450, 0.0689, 0.122]},
            name="lELMr",
            nturn=3 / 4 * 6,
            active=False,
            part="elm",
        )
        self.linkframe(["lELM", "lELMr"], factor=-1)
        # mid ELM (feed / return)
        mid_nturn = 20 / 40 * 6  # 20 degrees per sector
        self.coil.insert(
            {"r": [8.677, -0.5560, 0.0689, 0.122]},
            name="mELM",
            nturn=mid_nturn,
            active=False,
            part="elm",
        )
        self.coil.insert(
            {"r": [8.6340, 1.7960, 0.0689, 0.122]},
            name="mELMr",
            nturn=mid_nturn,
            active=False,
            part="elm",
        )
        self.linkframe(["mELM", "mELMr"], factor=-1)
        # upper ELM (feed / return)
        upper_nturn = 28.9 / 40 * 6  # 28.5 degrees per sector
        self.coil.insert(
            {"r": [8.2600, 2.6250, 0.0689, 0.122]},
            name="uELM",
            nturn=upper_nturn,
            active=False,
            part="elm",
        )
        self.coil.insert(
            {"r": [7.7330, 3.3780, 0.0689, 0.122]},
            name="uELMr",
            nturn=upper_nturn,
            active=False,
            part="elm",
        )
        self.linkframe(["uELM", "uELMr"], factor=-1)


@dataclass
class Geometry:
    """
    Manage IDS coil geometry attributes.

    Parameters
    ----------
    pf_active: Ids | bool | str, default=True
        pf active IDS.
    pf_passive: Ids | bool | str, default=True
        pf passive IDS.
    wall: Ids | bool | str, default = True
        wall IDS.
    elm: bool, default = False


    Examples
    --------
    Skip doctest if IMAS instalation or requisite IDS(s) not found.

    >>> import pytest
    >>> from nova.imas.database import Database
    >>> try:
    ...     _ = Database(111001, 202, 'pf_active', machine='iter_md')
    ...     _ = Database(115005, 2, 'pf_passive', machine='iter_md')
    ... except:
    ...     pytest.skip('IMAS not found or 111001/202, 115005/2 unavailable')

    Dissable wall geometry via boolean input:

    >>> geometry = Geometry(wall=False)
    >>> geometry.wall
    False
    >>> geometry.pf_active == PoloidalFieldActive.default_ids_attrs()
    True
    >>> geometry.pf_passive == PoloidalFieldPassive.default_ids_attrs()
    True

    Modify pf_active attrs via dict input:

    >>> pf_active = Geometry(pf_active=dict(run=101)).pf_active
    >>> pf_active == PoloidalFieldActive.default_ids_attrs() | dict(run=101)
    True

    Specify pf_active as an ids with pulse, run, and name:

    >>> database = Database(111001, 202, 'pf_active', machine='iter_md')
    >>> pf_active = Geometry(database.ids).pf_active
    >>> pf_active['pulse'] == 111001
    True

    Specify pf_active as an itterable:

    >>> pf_active = Geometry(pf_active=(111001, 202)).pf_active
    >>> tuple(pf_active[attr] for attr in ['pulse', 'run', 'name'])
    (111001, 202, 'pf_active')

    """

    pf_active: Ids | bool | str = True
    pf_passive: Ids | bool | str = True
    wall: Ids | bool | str = "iter_md"
    elm: bool = False
    filename: str = ""
    ids: ImasIds | None = field(default=None, repr=False)

    geometry: ClassVar[dict] = dict(
        pf_active=PoloidalFieldActive,
        pf_passive=PoloidalFieldPassive,
        wall=Wall,
        elm=ELM,
    )

    def __post_init__(self):
        """Map geometry parameters to dict attributes."""
        self.set_filename()
        for attr, geometry in self.geometry.items():
            ids_attrs = self.get_ids_attrs(getattr(self, attr), geometry)
            setattr(self, attr, ids_attrs)
        if hasattr(super(), "__post_init__"):
            super().__post_init__()

    def set_filename(self):
        """Set filename when all geometry attrs is str or False."""
        if (
            np.all(
                [
                    isinstance(getattr(self, attr), str) or getattr(self, attr) is False
                    for attr in self.geometry
                ]
            )
            and self.filename == ""
        ):
            self.filename = "machine_description"

    def get_ids_attrs(self, attrs, geometry) -> dict:
        """Return default ids attributes.

        Parameters
        ----------
        attrs : Ids | bool | str
            Descriptor for geometry ids.

        geometry : IDS
            Geometry class derived from IDS parent. When `attrs` != 'iter_md'
            child classes of the geometry class that define an `ids_attrs`
            attribute (such as the `nova.imas.Database`) merge the `attrs`
            parameter with `ids_attrs` with `attrs` taking priority in
            cases where an IDS attribute is pressent in both.

        Returns
        -------
        dict[str, int | str]
            IDS descriptor containing all the parameters required by the
            `nova.imas.database.IDS` class \
            (pulse, run, machine, occurrence, user, name, backend).
            The `attrs` attribute is matched according to its type and value
            as follows:

                - 'iter_md' : Return default IDS stored in `ids_attrs` \
                    attribute of the 'geometry' class.
                - True : Return  `geometry.ids_attrs` | `self.ids_attrs`
                - dict[str, int | str] : Return `geometry.ids_attrs` | attrs
                - tuple[int | str] : Return `geometry.ids_attrs` | \
                    `IDS(*attrs)`

        Examples
        --------
        Instantiate the geometry class.

        >>> geometry = Geometry()

        Get default IDS for ITER's wall.

        >>> wall_md = geometry.get_ids_attrs('iter_md', Wall)
        >>> list(wall_md.values())
        [116000, 2, 'wall', 0, 'iter_md', 'public', 'hdf5']

        Extend geometry instance with an `ids_attrs` attribute.

        >>> geometry.ids_attrs = wall_md | {'pulse': 20, 'run': 25}

        Confrim that new attrs overide the defaults profided by `Wall`.

        >>> list(geometry.get_ids_attrs(True, Wall).values())
        [20, 25, 'wall', 0, 'iter_md', 'public', 'hdf5']

        Request ids_attrs with a dict input replacing machine and occurence.

        >>> attrs = {'machine': 'iter', 'occurrence': 3}
        >>> list(geometry.get_ids_attrs(attrs, Wall).values())
        [20, 25, 'wall', 3, 'iter', 'public', 'hdf5']

        Request ids_attrs with a tuple input replacing the first 4 IDS attrs.

        >>> attrs = (20, 38, 'pf_passive', 3)
        >>> list(geometry.get_ids_attrs(attrs, Wall).values())
        [20, 38, 'pf_passive', 3, 'iter_md', 'public', 'hdf5']
        """
        match attrs:
            case "iter_md":  # update from iter_md
                ids_attrs = geometry.update_ids_attrs(True)
            case str():
                raise ValueError(f"attr str input {attrs} != iter_md")
            case attrs if hasattr(self, "ids_attrs") and self.ids is None:
                ids_attrs = geometry.merge_ids_attrs(attrs, self.ids_attrs)
            case attrs:
                ids_attrs = geometry.update_ids_attrs(attrs)
        return ids_attrs

    @property
    def geometry_attrs(self) -> dict:
        """Return geometry attributes."""
        return {
            f"{attr}_md": (
                value
                if isinstance(value := getattr(self, attr), (bool, np.integer))
                or "ids" not in value
                else "ids"  # Database(ids=value["ids"]).ids_hash
            )
            for attr in self.geometry
        }


@dataclass
class Machine(CoilSet, Geometry, CoilData):
    """Manage ITER machine geometry."""

    mode: str | None = None

    @property
    def metadata(self):
        """Manage machine metadata.

        Raises
        ------
            AssertionError:
                A change in the metadata group hash has been detected.
        """
        metadata = self.coilset_attrs
        for geometry in self.geometry_attrs:
            if (attrs := self.geometry_attrs[geometry]) is False:
                continue
            if isinstance(attrs, (int, bytes, str)):  # ids input
                metadata[geometry] = attrs
                continue
            metadata[geometry] = ",".join([str(attrs[attr]) for attr in attrs])
        return metadata

    @metadata.setter
    def metadata(self, metadata: dict):
        """Set instance metadata, assert consistent attr_hash."""
        print(self.group_attrs)
        attr_hash = self.hash_attrs(self.group_attrs)
        for attr in self.coilset_attrs:
            setattr(self, attr, metadata[attr])
        for geometry in self.geometry_attrs:
            if geometry not in metadata:
                setattr(self, geometry[:-3], False)
                continue
            print(geometry, metadata[geometry])
            if isinstance(metadata[geometry], np.integer):
                setattr(self, geometry[:-3], metadata[geometry])
                continue
            values = [
                self._format_geometry_attrs(attr)
                for attr in metadata[geometry].split(",")
            ]
            setattr(self, geometry[:-3], dict(zip(IdsBase.database_attrs, values)))
        print(self.group_attrs)
        assert attr_hash == self.hash_attrs(self.group_attrs)

    @staticmethod
    def _format_geometry_attrs(attr: str) -> str | int | float:
        """Return formated attr. Try int conversion except return str."""
        if "." in attr:
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
        if self.sloc["plasma"].sum() > 0:
            boundary = self.geometry["wall"](**self.wall).boundary
            self.plasma.solve(boundary)
        self.inductance.solve()
        self.field.solve()
        self.force.solve()

    def build(self, **kwargs):
        """Build dataset, frameset and, biotset and save to file."""
        self.frameset_attrs = kwargs
        self.clear_frameset()
        for attr, geometry in self.geometry.items():
            geometry_attrs = getattr(self, attr)
            if isinstance(geometry_attrs, dict):
                coilset = geometry(**geometry_attrs, **self.frameset_attrs)
                self += coilset

        if hasattr(super(), "build"):
            super().build()
        self.solve_biot()
        return self.store()

    def load(self):
        """Load machine geometry and data."""
        super().load()
        self.metadata = self.data.attrs
        return self

    def store(self):
        """Store frameset, biot attributes and metadata."""
        self.data.attrs = self.metadata | self.data.attrs
        super().store()
        return self


if __name__ == "__main__":

    import doctest

    doctest.testmod(verbose=False)

    kwargs = {"pulse": 105028, "run": 1, "machine": "iter"}  # DINA
    # kwargs = {"pulse": 45272, "run": 1, "machine": "mast_u"}  # MastU

    machine = Machine(
        **kwargs,
        pf_active="iter_md",
        pf_passive=False,
        elm=False,
        wall=True,
        tplasma="h",
        nwall=10,
        dplasma=-2000,
        ngrid=2000,
        ninductance=-0.05,
    )

    # from nova.imas.coils_non_axisymmetric import CoilsNonAxisymmetyric

    # machine += CoilsNonAxisymmetyric(111003, 2)  # CC
    # machine += CoilsNonAxisymmetyric(115001, 1)  # ELM

    machine.ferritic.insert("Fi")

    machine.frame.vtkplot()
