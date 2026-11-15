"""Manage access to IMAS machine data."""

from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from functools import cached_property
import itertools
import os
import string
from typing import ClassVar, TYPE_CHECKING
from warnings import warn

import numpy as np
import xarray

from nova.graphics.plot import Plot
from nova.frame.coilset import CoilSet
from nova.imas.database import CoilData, Database, IdsData
from nova.imas.dataset import IdsBase, Ids, ImasIds, EMPTY_FLOAT, open_db_entry
from nova.imas.ids_index import IdsIndex
from nova.geometry.polygon import Polygon
from nova.equilibrium.wall_mask import WallUnit, inside_polygon, wall_units_from_ids


if TYPE_CHECKING:
    from nova.frame.shell import Shell

# pylint: disable=too-many-ancestors


@dataclass
class GeomData:
    """Geometry data baseclass."""

    ids: ImasIds = field(repr=False)
    data: dict[str, int | float] = field(repr=False, default_factory=dict)
    attrs: ClassVar[list[str]] = []

    def __post_init__(self):
        """Extract attributes from ids."""
        if not self.data:
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
                warn(f"negative {attr} {self.data[attr]}")
                self.data[attr] *= -1

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
class MachineGeometryReader(ABC):
    """Provider seam presenting a poloidal geometry element to CrossSection.

    Decouples the geometry-type dispatch and attribute extraction from the
    backing store so that an IMAS IDS geometry node and a (FAIR-MAST) zarr
    source are interchangeable providers of the same :class:`GeomData`
    sections. A concrete reader reports the element's geometry type and builds
    the requested section from whatever it wraps.
    """

    source: object = field(repr=False)

    @property
    @abstractmethod
    def geometry_type(self) -> int:
        """Return the poloidal geometry-type index of the element."""

    @abstractmethod
    def section(self, geometry: type[GeomData]) -> GeomData:
        """Return the ``geometry`` section built from the backing source."""


@dataclass
class ImasGeometryReader(MachineGeometryReader):
    """Present an IMAS geometry IDS node through the reader seam."""

    @property
    def geometry_type(self) -> int:
        """Return the geometry-type index read from the IDS node."""
        return self.source.geometry_type.value

    def section(self, geometry: type[GeomData]) -> GeomData:
        """Build the section by extracting attributes from the IDS node."""
        return geometry(self.source)

    def iron_segment(self, segment: object) -> MachineIronSegment:
        """Read an iron segment's raw outline and material table together.

        The geometry source is ``segment.geometry``. Retain closing and repeated
        vertices exactly as stored rather than normalising through a polygon.
        """
        outline = self.section(Outline).data
        return MachineIronSegment(
            name=str(segment.name),
            description=str(segment.description),
            outline=MachineContour.from_record(
                {"kind": "iron", "r": outline["r"], "z": outline["z"]}
            ),
            material=MachineIronMaterial(
                b_field=tuple(float(value) for value in segment.b_field),
                permeability_relative=tuple(
                    float(value) for value in segment.permeability_relative
                ),
            ),
        )


@dataclass
class TabularGeometryReader(MachineGeometryReader):
    """Present a flat mapping record through the reader seam.

    The provider skeleton for facility geometry tables that are not IMAS
    IDSs — a FAIR-MAST zarr row, a columnar record, or any per-element
    mapping carrying ``geometry_type`` plus the section's attributes as flat
    scalar fields. A facility plugin normalises its native store into such
    records and drives :class:`CrossSection` through this reader; the
    facility tables themselves stay with the plugin.

    A record missing a required attribute raises ``KeyError`` naming the
    attribute — verify-and-flag, never a fabricated geometry value.
    """

    @property
    def geometry_type(self) -> int:
        """Return the geometry-type index read from the record."""
        return int(self.source["geometry_type"])

    def section(self, geometry: type[GeomData]) -> GeomData:
        """Build the section from the record's flat attribute fields."""
        try:
            data = {attr: self.source[attr] for attr in geometry.attrs}
        except KeyError as error:
            raise KeyError(
                f"tabular geometry record missing attribute {error.args[0]!r} "
                f"required by section {geometry.name!r}"
            ) from None
        return geometry(None, data)


@dataclass
class CrossSection:
    """Manage poloidal cross-sections."""

    ids: ImasIds = field(repr=False)
    data: GeomData = field(init=False)
    reader: ClassVar[type[MachineGeometryReader]] = ImasGeometryReader
    transform: ClassVar[dict[int, object]] = {
        1: Outline,
        2: Rectangle,
        3: Oblique,
        4: Arcs,
        5: Annulus,
        6: ThickLine,
    }

    def __post_init__(self):
        """Build geometry instance through the reader seam."""
        reader = self.reader(self.ids)
        geometry_type = reader.geometry_type
        try:
            self.data = reader.section(self.transform[geometry_type])
        except KeyError:
            if geometry_type == -999999999:
                default_geometry_type = 1  # remove once WEST pf_passive is fixed
                warn(f"geometry type unset, fixing value to {default_geometry_type}")
                self.data = reader.section(self.transform[default_geometry_type])
            else:
                raise
        if self.data.name == "outline" and len(self.data.data["r"]) == 1:  # WEST data
            for attr in "rz":
                self.data.data[attr] = self.data.data[attr][0]
            for attr in ["width", "height"]:
                self.data.data[attr] = 0.05
            self.data = Rectangle(None, self.data.data)

        for attr in self.data.data:
            if isinstance(self.data.data[attr], float) and not np.isfinite(
                self.data.data[attr]
            ):
                self.data.data[attr] = 0.1  # TODO remove once MastU IDS is fixed

    def __getattr__(self, attr):
        """Return data attributes."""
        return getattr(self.data, attr)


@dataclass(frozen=True)
class MachineContour:
    """Static poloidal contour carried by a machine description."""

    kind: str
    r: tuple[float, ...]
    z: tuple[float, ...]

    @classmethod
    def from_record(cls, record: dict) -> MachineContour:
        """Build a contour without changing the coordinates read from its IDS."""
        r = tuple(float(value) for value in record["r"])
        z = tuple(float(value) for value in record["z"])
        if len(r) != len(z):
            raise ValueError("machine contour r and z arrays have different lengths")
        return cls(kind=str(record["kind"]), r=r, z=z)


@dataclass(frozen=True)
class MachineIronMaterial:
    """Material law stored as paired B [T] and relative-permeability samples.

    The iron_core IDS encodes the B-H relation by these two arrays. It carries
    no separate material identifier; none is inferred from the segment name.
    """

    b_field: tuple[float, ...]
    permeability_relative: tuple[float, ...]

    def __post_init__(self):
        """Reject an unpaired material table without changing its samples."""
        if len(self.b_field) != len(self.permeability_relative):
            raise ValueError(
                "iron B and relative-permeability arrays have different lengths"
            )


@dataclass(frozen=True)
class MachineIronSegment:
    """One ordered iron region and its own material law."""

    name: str
    description: str
    outline: MachineContour
    material: MachineIronMaterial


WEST_IRON_SEGMENT_COUNT = 6
"""WEST stores its iron as column, two hubs, two arms and one return limb."""

WEST_IRON_SECTION_TOLERANCE = 0.01
"""A stored outline is refused when thickness times radius deviates by this."""

WEST_IRON_LIMB_WIDTH = 0.800
"""Toroidal width of one of the six WEST arms and return limbs, in metres."""

WEST_IRON_LIMB_DEPTH = 1.200
"""Radial depth of the return limb and poloidal thickness of an arm, in metres."""


def contour_moment(contour: tuple[tuple[float, float], ...]) -> float:
    """Return the first moment about the toroidal axis of a closed contour.

    This is the iron section per radian, ``integral of R dA`` over the poloidal
    polygon, which is the quantity a toroidal fill fraction has to preserve.
    """
    radius = np.asarray([point[0] for point in contour], dtype=float)
    height = np.asarray([point[1] for point in contour], dtype=float)
    next_radius = np.roll(radius, -1)
    next_height = np.roll(height, -1)
    cross = radius * next_height - next_radius * height
    return float(((radius + next_radius) * cross).sum() / 6)


def _box(radius: tuple[float, float], height: tuple[float, float]) -> tuple:
    """Return the closed rectangular contour spanning the two ranges."""
    (r0, r1), (z0, z1) = radius, height
    return ((r0, z0), (r1, z0), (r1, z1), (r0, z1), (r0, z0))


@dataclass(frozen=True)
class IronRegionSection:
    """Stored and filled section per radian of one iron region."""

    region: str
    stored: float
    filled: float

    @property
    def relative(self) -> float:
        """Return the filled minus stored section relative to the stored one."""
        return (self.filled - self.stored) / self.stored


@dataclass(frozen=True)
class IronFillRegion:
    """One region of the true flat iron outline and its toroidal fill rule.

    ``fill_knots`` holds the toroidal iron fraction at knot radii; it is
    interpolated linearly between them and held constant outside. ``None``
    marks a contiguous block of stored iron, whose fill fraction is one. The
    knots are read from the stored outline's own section per radian, so the
    region reproduces that section by construction.
    """

    name: str
    kind: str
    outline: tuple[tuple[float, float], ...]
    fill_knots: tuple[tuple[float, float], ...] | None
    stored_section_per_radian: float

    @property
    def radius(self) -> np.ndarray:
        """Return the knot radii of the fill profile."""
        if self.fill_knots is None:
            r0, _, r1, _ = self.bounds
            return np.array([r0, r1], dtype=float)
        return np.asarray([knot[0] for knot in self.fill_knots], dtype=float)

    @property
    def profile(self) -> np.ndarray:
        """Return the fill fraction at each knot radius."""
        if self.fill_knots is None:
            return np.ones(2)
        return np.asarray([knot[1] for knot in self.fill_knots], dtype=float)

    @property
    def contiguity_radius(self) -> float:
        """Return the outermost radius at which the region is fully iron."""
        radius = self.radius[self.profile >= 1 - 1e-12]
        return float(radius.max()) if len(radius) else float(self.radius[0])

    def fill(self, radius) -> np.ndarray:
        """Return the toroidal iron fraction at each radius."""
        radius = np.asarray(radius, dtype=float)
        if self.fill_knots is None:
            return np.ones_like(radius)
        return np.clip(np.interp(radius, self.radius, self.profile), 0.0, 1.0)

    def covered(self, centroids) -> np.ndarray:
        """Return the mask of poloidal centroids inside this region."""
        centroids = np.asarray(centroids, dtype=float)
        radius = np.asarray([point[0] for point in self.outline], dtype=float)
        height = np.asarray([point[1] for point in self.outline], dtype=float)
        return inside_polygon(centroids[:, 0], centroids[:, 1], radius, height)

    @property
    def is_rectangular(self) -> bool:
        """Return whether the region is an axis-aligned box."""
        radius = sorted({point[0] for point in self.outline})
        height = sorted({point[1] for point in self.outline})
        return len(radius) == 2 and len(height) == 2

    @property
    def bounds(self) -> tuple[float, float, float, float]:
        """Return the (R min, Z min, R max, Z max) extents of the region."""
        radius = [point[0] for point in self.outline]
        height = [point[1] for point in self.outline]
        return min(radius), min(height), max(radius), max(height)

    def radial_integral(self) -> float:
        """Return the exact ``integral of fill fraction times R dR``.

        The profile is linear between knots, so each interval integrates in
        closed form; this is the quantity the stored section per radian is
        reproduced against.
        """
        radius, fill = self.radius, self.profile
        total = 0.0
        for index in range(len(radius) - 1):
            x0, x1 = radius[index], radius[index + 1]
            f0, f1 = fill[index], fill[index + 1]
            span = x1 - x0
            if span == 0:
                continue
            slope = (f1 - f0) / span
            total += f0 * (x1**2 - x0**2) / 2 + slope * (
                (x1**3 - x0**3) / 3 - x0 * (x1**2 - x0**2) / 2
            )
        return total

    def section_per_radian(self) -> float:
        """Return ``integral of fill fraction times R dA`` over the region."""
        if self.fill_knots is None:
            return contour_moment(self.outline)
        r0, z0, r1, z1 = self.bounds
        if not np.isclose(self.radius[0], r0) or not np.isclose(self.radius[-1], r1):
            raise ValueError(
                f"fill profile of region {self.name!r} spans "
                f"{self.radius[0]:.6f} to {self.radius[-1]:.6f} m, its outline "
                f"{r0:.6f} to {r1:.6f} m"
            )
        return (z1 - z0) * self.radial_integral()


@dataclass(frozen=True)
class IronCoreOutline:
    """True flat iron outline, toroidal fill fraction and section accounting.

    One read of the machine description serves every consumer: the stored
    segments are retained, each arm and limb region carries the fill fraction
    its section per radian requires, and ``sections`` reports the stored
    section each region reproduces.
    """

    description: StaticMachineDescription
    regions: tuple[IronFillRegion, ...]
    sections: tuple[IronRegionSection, ...]
    limb_width: float
    limb_depth: float
    limb_arm_gap: float
    gap_input: str

    @property
    def cross_field_factor(self) -> float:
        """Return the demagnetising factor of the limb section, N = w / (w + d)."""
        return self.limb_width / (self.limb_width + self.limb_depth)

    @property
    def stored(self) -> tuple[MachineContour, ...]:
        """Return the stored contours unchanged."""
        return tuple(segment.outline for segment in self.description.iron_segments)

    @property
    def outlines(self) -> tuple[tuple[tuple[float, float], ...], ...]:
        """Return every region outline on the true flat iron outline."""
        return tuple(region.outline for region in self.regions)

    def region(self, name: str) -> IronFillRegion:
        """Return the named region."""
        return next(region for region in self.regions if region.name == name)

    def fill(self, centroids) -> np.ndarray:
        """Return the toroidal iron fraction of each poloidal cell centroid."""
        centroids = np.asarray(centroids, dtype=float)
        fill = np.zeros(len(centroids), dtype=float)
        for region in self.regions:
            covered = region.covered(centroids)
            fill[covered] = region.fill(centroids[covered, 0])
        return fill

    def cross_field(self, centroids) -> np.ndarray:
        """Return the per-cell demagnetising factor (N_R, N_Z) of each cell.

        A limb shares H with the gaps beside it along its own axis, so N is
        zero there and ``cross_field_factor`` across it; an arm is crossed in
        the vertical direction and a limb in the radial one.
        """
        centroids = np.asarray(centroids, dtype=float)
        factor = np.zeros((len(centroids), 2))
        for region in self.regions:
            if region.fill_knots is None:
                continue
            component = 1 if region.kind == "arm" else 0
            factor[region.covered(centroids), component] = self.cross_field_factor
        return factor

    def maximum_relative_section_error(self, kinds: tuple[str, ...]) -> float:
        """Return the largest section mismatch over the named region kinds."""
        selected = {
            section.region: abs(section.relative)
            for section, region in zip(self.sections, self.regions, strict=True)
            if region.kind in kinds
        }
        if not selected:
            raise ValueError(f"no {kinds} region carries a section")
        return max(selected.values())


@dataclass(frozen=True)
class MachineSection:
    """Named active-coil section routed through the geometry reader seam."""

    name: str
    section: GeomData
    turns: float = 0.0

    @property
    def outline(self) -> tuple[tuple[float, float], ...]:
        """Return the canonical section outline without its closing vertex."""
        radius, height = self.section.poly.exterior.xy
        return tuple(
            (float(r_value), float(z_value))
            for r_value, z_value in zip(radius[:-1], height[:-1], strict=True)
        )

    @property
    def centre(self) -> tuple[float, float]:
        """Return the area centroid of the routed section."""
        centroid = self.section.poly.centroid
        return float(centroid.x), float(centroid.y)

    @property
    def width(self) -> float:
        """Return the radial extent of the routed section."""
        r_min, _, r_max, _ = self.section.poly.bounds
        return float(r_max - r_min)

    @property
    def height(self) -> float:
        """Return the vertical extent of the routed section."""
        _, z_min, _, z_max = self.section.poly.bounds
        return float(z_max - z_min)

    @property
    def skew(self) -> float:
        """Return the directed angle of the first non-zero outline edge."""
        for start, end in itertools.pairwise(self.outline + self.outline[:1]):
            delta_r = end[0] - start[0]
            delta_z = end[1] - start[1]
            if not np.isclose(delta_r, 0.0) or not np.isclose(delta_z, 0.0):
                return float(np.arctan2(delta_z, delta_r))
        raise ValueError(f"machine section {self.name!r} has no non-zero edge")

    @classmethod
    def from_record(cls, record: dict) -> MachineSection:
        """Route a flat IDS receipt through the canonical section classes."""
        geometry_type = int(record["geometry_type"])
        try:
            geometry = CrossSection.transform[geometry_type]
        except KeyError:
            raise ValueError(f"unsupported geometry type {geometry_type}") from None
        return cls(
            name=str(record["name"]),
            section=TabularGeometryReader(record).section(geometry),
            turns=float(record.get("turns_with_sign", record.get("turns", 0.0))),
        )


@dataclass(frozen=True)
class MachineCoil:
    """Named active coil retaining its ordered poloidal elements."""

    name: str
    identifier: str
    elements: tuple[MachineSection, ...]

    def winding_sections(self, count: int) -> tuple[MachineSection, ...]:
        """Divide a rectangular winding pack into ordered series sections.

        The section geometry occupies the original pack without overlap.  The
        final turn count takes the floating-point remainder, so the signed
        turns of the series sections sum exactly to the source winding.
        """
        if count < 1:
            raise ValueError("winding section count must be positive")
        if len(self.elements) == count:
            return self.elements
        if len(self.elements) != 1:
            raise ValueError("winding layout does not match source elements")
        source = self.elements[0]
        if source.section.name != "rectangle" or source.turns == 0:
            raise ValueError("a winding pack needs a rectangle and signed turns")
        geometry = source.section.data
        height = float(geometry["height"]) / count
        centre = float(geometry["z"])
        turns = source.turns / count
        sections = []
        for index in range(count):
            record = {
                "name": f"{self.name} pancake {index + 1}",
                "geometry_type": 2,
                "r": float(geometry["r"]),
                "z": centre + (index + 0.5 - count / 2) * height,
                "width": float(geometry["width"]),
                "height": height,
                "turns_with_sign": (
                    turns if index + 1 < count else source.turns - turns * (count - 1)
                ),
            }
            sections.append(MachineSection.from_record(record))
        return tuple(sections)

    @classmethod
    def from_record(cls, record: dict) -> MachineCoil:
        """Build one coil and route every element through ``MachineSection``."""
        elements = tuple(
            MachineSection.from_record(element)
            for element in record.get("elements", ())
        )
        return cls(
            name=str(record["name"]),
            identifier=str(record.get("identifier", "")),
            elements=elements,
        )


@dataclass(frozen=True)
class DiagnosticSightline:
    """Static diagnostic location and optional line-of-sight endpoints."""

    name: str
    position: tuple[float, float, float]
    start: tuple[float, float, float] | None
    end: tuple[float, float, float] | None

    @classmethod
    def from_record(cls, record: dict) -> DiagnosticSightline:
        """Build a diagnostic record, retaining absent endpoints as absent."""

        def point(key: str) -> tuple[float, float, float] | None:
            value = record.get(key)
            if value is None:
                return None
            return tuple(float(component) for component in value)

        position = point("position")
        if position is None:
            raise ValueError("diagnostic record has no position")
        return cls(
            name=str(record["name"]),
            position=position,
            start=point("start"),
            end=point("end"),
        )


@dataclass(frozen=True)
class StaticMachineDescription:
    """Static IDS quantities used to construct a Nova machine."""

    contour: MachineContour | None
    active_sections: tuple[MachineSection, ...]
    passive_loop_count: int
    toroidal_coil_count: int
    sightlines: tuple[DiagnosticSightline, ...]
    active_coils: tuple[MachineCoil, ...] = ()
    wall_units: tuple[WallUnit, ...] = ()
    iron_segments: tuple[MachineIronSegment, ...] = ()

    @classmethod
    def from_iron_core_ids(cls, iron: object) -> StaticMachineDescription:
        """Read every segment through the IMAS geometry owner, in IDS order."""
        return cls(
            contour=None,
            active_sections=(),
            passive_loop_count=0,
            toroidal_coil_count=0,
            sightlines=(),
            iron_segments=tuple(
                ImasGeometryReader(segment.geometry).iron_segment(segment)
                for segment in iron.segment
            ),
        )

    @classmethod
    def read_iron_core(
        cls, source: str | os.PathLike, *, dd_version: str
    ) -> StaticMachineDescription:
        """Open static iron at the source manifest's exact dictionary pin.

        ``source`` is an IMAS entry URI or netCDF path. An absent occurrence is
        a refusal about this entry, never a fabricated empty iron frame.
        """
        if not dd_version:
            raise ValueError("iron read requires the source manifest DD version")
        with open_db_entry(source, "r", dd_version=dd_version) as entry:
            occurrences = entry.list_all_occurrences("iron_core")
            if 0 not in occurrences:
                raise ValueError(
                    f"iron_core occurrence 0 is absent from {source}: {occurrences}"
                )
            iron = entry.get("iron_core", 0, lazy=False, autoconvert=False)
            written = str(iron.ids_properties.version_put.data_dictionary)
            if written != dd_version:
                raise ValueError(
                    f"iron_core DD version {written!r} disagrees with "
                    f"manifest {dd_version!r}"
                )
            return cls.from_iron_core_ids(iron)

    @classmethod
    def read_west_iron_outline(
        cls,
        source: str | os.PathLike,
        *,
        dd_version: str,
        limb_arm_gap: str = "stored",
        limb_width: float = WEST_IRON_LIMB_WIDTH,
        limb_depth: float = WEST_IRON_LIMB_DEPTH,
    ) -> IronCoreOutline:
        """Read the WEST iron once on its true flat outline.

        The stored outline thins as ``thickness * radius = constant``, so the
        true flat regions are rebuilt with a toroidal fill fraction that
        preserves the section per radian of each arm and of the return limb.

        * ``limb_arm_gap`` names how the stored 2 mm face gap between the
          return limb and the arm underside is treated. Only ``'stored'`` is
          supported: the gap decision keeps the description's own faces, and
          any other value is refused rather than silently converted.
        * ``limb_width`` and ``limb_depth`` are the section of the
          limb-geometry decision, 0.800 m wide by 1.200 m deep.
        """
        if limb_arm_gap != "stored":
            raise ValueError(
                f"limb_arm_gap {limb_arm_gap!r} is not supported: the stored "
                "2 mm limb to arm gap is kept as stored, so this read accepts "
                "only 'stored'"
            )
        description = cls.read_iron_core(source, dd_version=dd_version)
        return iron_core_outline(
            description,
            limb_width=limb_width,
            limb_depth=limb_depth,
            gap_input=limb_arm_gap,
        )

    @classmethod
    def from_record(cls, record: dict) -> StaticMachineDescription:
        """Route an IMAS extraction record through Nova's machine dataclasses."""
        contour_record = record.get("contour")
        active_records = tuple(record.get("pf_active", ()))
        if active_records and "elements" in active_records[0]:
            active_coils = tuple(
                MachineCoil.from_record(coil) for coil in active_records
            )
            active_sections = tuple(
                element for coil in active_coils for element in coil.elements
            )
        else:
            active_sections = tuple(
                MachineSection.from_record(section) for section in active_records
            )
            active_coils = tuple(
                MachineCoil(
                    name=section.name,
                    identifier=section.name,
                    elements=(section,),
                )
                for section in active_sections
            )
        return cls(
            contour=(
                MachineContour.from_record(contour_record)
                if contour_record is not None
                else None
            ),
            active_sections=active_sections,
            passive_loop_count=int(record.get("pf_passive_loop_count", 0)),
            toroidal_coil_count=int(record.get("tf_coil_count", 0)),
            sightlines=tuple(
                DiagnosticSightline.from_record(sightline)
                for sightline in record.get("thomson_scattering", ())
            ),
            active_coils=active_coils,
            wall_units=tuple(record.get("wall_units", ())),
        )


def iron_core_outline(
    description: StaticMachineDescription,
    *,
    limb_width: float = WEST_IRON_LIMB_WIDTH,
    limb_depth: float = WEST_IRON_LIMB_DEPTH,
    gap_input: str = "stored",
) -> IronCoreOutline:
    """Build the true flat iron outline from one stored description.

    The stored segments are classified by their own geometry — the return limb
    is the only segment taller than two limb depths, the arms are the two whose
    poloidal thickness is one limb depth, the column is the remaining segment
    that straddles the midplane and the hubs are the rest. A stored arm or limb
    that does not obey thickness times radius constant to
    :data:`WEST_IRON_SECTION_TOLERANCE` is refused by name; it is never
    converted.
    """
    segments = description.iron_segments
    if len(segments) != WEST_IRON_SEGMENT_COUNT:
        raise ValueError(
            f"WEST iron outline expects {WEST_IRON_SEGMENT_COUNT} stored "
            f"segments, read {len(segments)}"
        )
    extents = [_segment_extents(segment) for segment in segments]
    tolerance = WEST_IRON_SECTION_TOLERANCE

    arms = [
        index
        for index, (r0, z0, r1, z1) in enumerate(extents)
        if abs((z1 - z0) - limb_depth) <= tolerance * limb_depth
    ]
    limbs = [
        index
        for index, (r0, z0, r1, z1) in enumerate(extents)
        if (z1 - z0) > 2 * limb_depth
    ]
    if len(arms) != 2 or len(limbs) != 1:
        raise ValueError(
            "stored WEST iron is not column, two hubs, two arms and one "
            f"return limb: arms {arms}, limbs {limbs}"
        )
    (limb_index,) = limbs
    remaining = [
        index for index in range(len(segments)) if index not in (*arms, *limbs)
    ]
    columns = [
        index for index in remaining if extents[index][1] < 0 < extents[index][3]
    ]
    if len(columns) != 1:
        raise ValueError(f"stored WEST iron has no single midplane column: {columns}")
    (column_index,) = columns
    hubs = [index for index in remaining if index != column_index]

    upper = next((index for index in arms if extents[index][1] > 0), None)
    lower = next((index for index in arms if extents[index][3] < 0), None)
    if upper is None or lower is None:
        raise ValueError(
            "stored WEST iron arms are not one above and one below the "
            f"midplane: {[extents[index][1::2] for index in arms]}"
        )
    thickness = {index: _arm_taper(segments[index], extents[index]) for index in arms}
    constant = float(np.median(thickness[upper][:, 1]))
    for index in arms:
        _refuse_taper(segments[index], index, thickness[index], constant)
    limb_r0, limb_z0, limb_r1, limb_z1 = extents[limb_index]
    _refuse_limb(segments[limb_index], limb_index, extents[limb_index], constant)

    upper_face = extents[upper][1]
    lower_face = extents[lower][3]
    gaps = (upper_face - limb_z1, limb_z0 - lower_face)
    if gaps[0] < 0 or not np.isclose(gaps[0], gaps[1], rtol=0, atol=1e-9):
        raise ValueError(
            "stored WEST iron limb to arm gaps are not a stored face pair: "
            f"{gaps[0]:.6f} m above the limb, {gaps[1]:.6f} m below it"
        )
    gap = float(gaps[0])

    regions = []
    for index, face, sign in ((upper, upper_face, 1), (lower, lower_face, -1)):
        panel = (face, face + sign * limb_depth)
        regions.append(
            IronFillRegion(
                name="upper arm" if sign > 0 else "lower arm",
                kind="arm",
                outline=_box((extents[index][0], extents[index][2]), panel),
                fill_knots=_arm_fill_knots(segments[index], face, limb_depth),
                stored_section_per_radian=contour_moment(_contour(segments[index])),
            )
        )
    limb_section = contour_moment(_contour(segments[limb_index]))
    regions.append(
        IronFillRegion(
            name="return limb",
            kind="limb",
            outline=_box((limb_r0, limb_r0 + limb_depth), (limb_z0, limb_z1)),
            fill_knots=_limb_fill_knots(extents[limb_index], limb_depth, limb_section),
            stored_section_per_radian=limb_section,
        )
    )
    for index, kind, name in (
        (column_index, "column", "central column"),
        *((hub, "hub", f"hub {position}") for position, hub in enumerate(hubs)),
    ):
        regions.append(
            IronFillRegion(
                name=name,
                kind=kind,
                outline=_contour(segments[index]),
                fill_knots=None,
                stored_section_per_radian=contour_moment(_contour(segments[index])),
            )
        )
    sections = tuple(
        IronRegionSection(
            region=region.name,
            stored=region.stored_section_per_radian,
            filled=region.section_per_radian(),
        )
        for region in regions
    )
    return IronCoreOutline(
        description=description,
        regions=tuple(regions),
        sections=sections,
        limb_width=limb_width,
        limb_depth=limb_depth,
        limb_arm_gap=float(gap),
        gap_input=gap_input,
    )


def _contour(segment: MachineIronSegment) -> tuple[tuple[float, float], ...]:
    """Return a stored segment as a closed coordinate contour."""
    return tuple(zip(segment.outline.r, segment.outline.z, strict=True))


def _segment_extents(segment: MachineIronSegment) -> tuple[float, float, float, float]:
    """Return the (R min, Z min, R max, Z max) extents of a stored segment."""
    radius = np.asarray(segment.outline.r, dtype=float)
    height = np.asarray(segment.outline.z, dtype=float)
    return (
        float(radius.min()),
        float(height.min()),
        float(radius.max()),
        float(height.max()),
    )


def _arm_fill_knots(
    segment: MachineIronSegment, face: float, depth: float
) -> tuple[tuple[float, float], ...]:
    """Return the arm fill profile read from the stored arm's own thickness.

    A knot sits at every stored vertex radius, so the linear interpolant
    reproduces the stored piecewise-linear taper exactly and the region keeps
    the stored section per radian.
    """
    knots = {}
    for radius, height in zip(segment.outline.r, segment.outline.z, strict=True):
        panel = abs(float(height) - face)
        if panel > 1e-12:
            knots[float(radius)] = panel / depth
    return tuple(sorted(knots.items()))


def _limb_fill_knots(
    extents, depth: float, stored_section: float, nodes: int = 17
) -> tuple[tuple[float, float], ...]:
    """Return the limb fill profile that keeps ``fill times radius`` stored.

    The stored limb is a full-depth block over a thin radial span, so the
    filled flat limb carries a constant ``fill times radius`` across its own
    depth. The knots sample that rule; seventeen nodes bind the linear
    interpolation error below 1e-4 of the section.
    """
    r0, z0, _, z1 = extents
    moment = stored_section / ((z1 - z0) * depth)
    return tuple(
        (float(radius), float(min(1.0, moment / radius)))
        for radius in np.linspace(r0, r0 + depth, nodes)
    )


def _arm_taper(segment: MachineIronSegment, extents) -> np.ndarray:
    """Return (radius, thickness times radius) at every tapered vertex.

    Thickness is measured from the flat face nearest the midplane. Vertices
    inboard of the contiguity radius carry the full limb depth; they are kept
    with their radius so the refusal can check them separately.
    """
    _, z0, _, z1 = extents
    face = z0 if z0 >= 0 else z1
    rows = []
    for radius, height in zip(segment.outline.r, segment.outline.z, strict=True):
        panel = abs(float(height) - face)
        if panel > 1e-12:
            rows.append((float(radius), panel * float(radius)))
    return np.asarray(rows, dtype=float)


def _refuse_taper(
    segment: MachineIronSegment, index: int, taper: np.ndarray, constant: float
) -> None:
    """Refuse a stored arm whose thickness times radius is not constant."""
    tolerance = WEST_IRON_SECTION_TOLERANCE
    depth = WEST_IRON_LIMB_DEPTH
    contiguity = WEST_IRON_SEGMENT_COUNT * WEST_IRON_LIMB_WIDTH / (2 * np.pi)
    for radius, moment in taper:
        if moment / radius > depth * (1 + tolerance):
            raise ValueError(
                f"stored iron segment {index} {segment.name!r} does not obey "
                "the limb section: thickness "
                f"{moment / radius:.6f} m at radius {radius:.4f} m exceeds the "
                f"limb depth {depth:.6f} m"
            )
        if radius < contiguity * (1 - 1e-9):
            if abs(moment / radius - depth) > tolerance * depth:
                raise ValueError(
                    f"stored iron segment {index} {segment.name!r} does not obey "
                    "thickness times radius constant: thickness "
                    f"{moment / radius:.6f} m at radius {radius:.4f} m is not "
                    f"the limb depth {depth:.6f} m"
                )
            continue
        relative = moment / constant - 1
        if abs(relative) > tolerance:
            raise ValueError(
                f"stored iron segment {index} {segment.name!r} does not obey "
                f"thickness times radius constant: {moment:.6f} m2 at radius "
                f"{radius:.4f} m against {constant:.6f} m2, {relative:+.2%} "
                f"beyond the {tolerance:.0%} tolerance"
            )


def _refuse_limb(
    segment: MachineIronSegment, index: int, extents, constant: float
) -> None:
    """Refuse a stored return limb whose depth times radius is not constant."""
    r0, _, r1, _ = extents
    moment = (r1 - r0) * 0.5 * (r0 + r1)
    relative = moment / constant - 1
    if abs(relative) > WEST_IRON_SECTION_TOLERANCE:
        raise ValueError(
            f"stored iron segment {index} {segment.name!r} does not obey "
            f"thickness times radius constant: limb depth {r1 - r0:.6f} m times "
            f"mean radius {0.5 * (r0 + r1):.6f} m reads {moment:.6f} m2 against "
            f"{constant:.6f} m2, {relative:+.2%} beyond the "
            f"{WEST_IRON_SECTION_TOLERANCE:.0%} tolerance"
        )


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
    index: int
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
        """Return True if geometry.name == 'oblique' or 'annulus' or 'outline'."""
        return self.section in ["oblique", "annulus", "outline"]

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
        if digit == "":
            return name
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
        if "VES" in label or "VV" in label or "Vacuum" in label:
            return "vv"
        if "passive" in label:
            return "passive"
        if "TRI" in label or "Tri" in label:
            return "trs"
        if label == "INB_RAIL" or "Div" in label:
            return "dir"
        if "TR" in label:
            return "oh"
        if "VF" in label:
            return "vf"
        if "CS" in label or "PF" in label:
            return label[:2].lower()
        if "SS" in label:
            return "vs3j"
        if "VS" in label:
            return "vs3"
        if "cryo" in label.lower():
            return "cryo"
        if "FPPC" in label:
            return "ivc"
        return "pf"

    @staticmethod
    def update_resistivity(index, frame, subframe, resistance):
        """Update frame and subframe resistivity."""
        # a label selection returns a bare column vector ordered by index, so
        # each coil's resistivity is read back positionally, not by label
        rho = resistance * frame.loc[index, "area"] / frame.loc[index, "dy"]
        frame.loc[index, "rho"] = rho
        for name, coil_rho in zip(index, rho, strict=True):
            subframe.loc[subframe.frame == name, "rho"] = coil_rho
        subframe.update_frame()

    @staticmethod
    def update_passive_turns(index, frame, subframe):
        """Update turns to match cross-sectional area."""
        frame.loc[index, "nturn"] = frame.loc[index, "area"]
        for name in index:
            subindex = subframe.frame == name
            subframe.loc[subindex, "nturn"] = subframe.loc[subindex, "area"]
        subframe.update_frame()


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

    def insert(self, shell: Shell, **kwargs):
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
                **kwargs,
            )
            self.update_resistivity(
                index, shell.frame, shell.subframe, self.data["resistance"][i]
            )
            # self.update_passive_turns(index, *shell.frames)
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
            "name": self.data["name"],
            "section": self.data["section"],
        } | kwargs
        index = super().insert(constructor, **kwargs)
        # self.update_passive_turns(index, *constructor.frames)
        return index


@dataclass
class PassivePolyCoilData(PassiveCoilData):
    """Extract coildata from passive ids."""

    geometry_attrs: ClassVar[list[str]] = ["poly"]


@dataclass
class CoilDatabase(CoilSet, CoilData):
    """Manage coilset construction from ids structures."""

    machine: str = "iter_md"
    ids_node: str = ""

    def load_build(self):
        """Load the cached coilset, locating its source before a rebuild.

        A cached coilset is authoritative on its own, so an unreachable
        source is an error only once the geometry has to be read back from
        its IDS. Construction otherwise probes the datastore tolerantly and
        carries on with no IDS attached.
        """
        try:
            self.load()
            return
        except FileNotFoundError, OSError:
            self.locate_datastore()
        super().load_build()

    def locate_datastore(self):
        """Raise a located error when the machine description path is absent.

        Two parameters steer the filesystem lookup without describing any of
        the machine: ``user`` picks the imasdb root (the shared tree, or a
        named account's public tree) and the major of ``dd_version`` picks the
        version subtree beneath it. Neither is visible in the geometry the
        caller asked for, so a default that does not match the installation
        resolves to a path that simply is not there. Report the level that is
        missing beside the parameter that chose it, rather than letting the
        backend fail on an assembled path.
        """
        if self._ids is not None or self.mode != "r":
            return
        if f"user={self.user};" not in self.uri:  # not a pulse/run layout
            return
        if not os.path.isdir(self.home):
            root = (
                "the shared tree under $IMAS_HOME"
                if self.user == "public"
                else f"the public tree of account {self.user!r}"
            )
            raise FileNotFoundError(
                f"no imasdb root at {self.home!r} ({root}); "
                f"the root is selected by user={self.user!r}"
            )
        machine_path = os.path.dirname(self.database_path)
        if not os.path.isdir(machine_path):
            raise FileNotFoundError(
                f"no {self.machine!r} database at {machine_path!r}; "
                f"the database is selected by machine={self.machine!r} "
                f"under user={self.user!r}"
            )
        if not os.path.isdir(self.database_path):
            majors = sorted(
                entry for entry in os.listdir(machine_path) if entry.isdigit()
            )
            raise FileNotFoundError(
                f"no data dictionary major {self.dd_version.major} at "
                f"{self.database_path!r}; the subtree is selected by "
                f"dd_version={str(self.dd_version)!r} and {machine_path!r} "
                f"holds major {majors}"
            )
        if self.is_empty:
            raise FileNotFoundError(
                f"no {self.name!r} entry at {self.ids_path!r}; the entry is "
                f"selected by pulse={self.pulse} and run={self.run}"
            )

    @cached_property
    def ids_index(self):
        """Return cached ids_index instance."""
        return IdsIndex(self.ids, self.ids_node)

    @property
    def group_attrs(self) -> dict:
        """
        Return group attrs.

        Extends :func:`~nova.imas.database.CoilData`. The data dictionary
        version is part of the compiled coil geometry identity, preventing a
        request made under one schema from loading an entry built under
        another.
        """
        return (
            self.coilset_attrs | self.ids_attrs | {"dd_version": str(self.dd_version)}
        )


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
                if element.is_thickline():
                    shelldata.append(loop, element)
                    continue
                if element.is_rectangular():
                    coildata.append(loop, element)
                    continue
                if element.is_poly():
                    polydata.append(loop, element)
                    continue
                raise NotImplementedError(
                    f"geometory {element.section} not implemented"
                )
            coildata.insert(self.coil, delta=self.dcoil)
            polydata.insert(self.coil, delta=-10)
            shelldata.insert(self.shell, delta=self.dshell)


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
    """Manage active poloidal loop ids, pf_active."""

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
                raise NotImplementedError(f"geometry {element.name} not implemented")
            if len(ids_loop.element) == 1:
                constructor = self.coil
            else:
                constructor = self.turn
            coildata.insert(constructor)
            polydata.insert(constructor)
            current_limit_max = ids_loop.current_limit_max
            if len(current_limit_max) > 0:
                maximum_current[ids_loop.identifier.value] = current_limit_max[-1, 0]
        self.update_current_limits(maximum_current)

    def update_current_limits(self, maximum_current: dict[str, float]):
        """Set symmetric per-coil current limits from an identifier mapping.

        Imin and Imax are subspace attributes, so each limit is addressed to a
        single independent row through the subspace accessor. An identifier
        that names no inserted coil is a mismatch between the coil array and
        the geometry built from it, and raises rather than writing nothing.
        """
        labels = set(np.asarray(self.frame.subspace.index).tolist())
        if unknown := sorted(set(maximum_current) - labels):
            raise KeyError(
                f"current limits carry identifiers {unknown} that name no "
                f"inserted coil; frame holds {sorted(labels)}"
            )
        for coil, limit in maximum_current.items():
            self.sLoc[coil, "Imax"] = float(limit)
            self.sLoc[coil, "Imin"] = -float(limit)
        # circuit linking rebuilds the subspace from the frame columns, so the
        # limits are projected back onto the frame before that happens
        self.frame.update_frame()

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
    count: itertools.count = field(init=False, default_factory=itertools.count)

    def append(self, unit):
        """Append contour data."""
        if (name := unit.name.value) == "":
            name = f"vessel_{next(self.count)}"
        self.data[name] = np.c_[unit.annular.centreline.r, unit.annular.centreline.z]
        if unit.annular.centreline.closed == 1 and not np.allclose(
            self.data[name][0], self.data[name][-1]
        ):
            self.data[name] = np.append(self.data[name], self.data[name][:1], axis=0)

    def plot(self, axes=None, legend=False):
        """Plot contours."""
        self.set_axes("2d", axes=axes)
        for component, contour in self.data.items():
            self.axes.plot(*contour.T, label=component)
        if legend:
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

    def plot(self, axes=None, color="k", **kwargs):
        """Plot closed contour."""
        self.get_axes("2d", axes=axes)
        self.axes.plot(*self.loop.T, color=color, **kwargs)


@dataclass
class Wall(CoilDatabase):
    """Manage plasma boundary, wall ids."""

    pulse: int = 116000
    run: int = 2
    name: str = "wall"
    occurrence: int = 0

    @cached_property
    def vessel(self):
        """Return vessel.

        The cached coilset holds meshed geometry, not the vessel outline, so a
        boundary read needs the wall ids itself even when the coilset loaded
        from cache. Locate the source instead of dereferencing an absent ids.
        """
        if self.ids is None:
            self.locate_datastore()
            raise ValueError(
                f"no {self.name!r} ids attached; the vessel outline is read "
                f"from {self.ids_path!r}"
            )
        if len(self.ids.description_2d) != 1:
            raise ValueError(
                f"wall ids carries {len(self.ids.description_2d)} descriptions; "
                "the boundary read selects exactly one"
            )
        return self.ids.description_2d[0].vessel  # DDv4

    @cached_property
    def contour(self):
        """Return closed firstwall contour instance."""
        firstwall = ContourData()
        for unit in self.vessel.unit:
            firstwall.append(unit)
        return Contour(firstwall.data)

    @cached_property
    def units(self) -> tuple[WallUnit, ...]:
        """Return every limiter unit without concatenating their outlines."""

        if self.ids is None:
            self.locate_datastore()
            raise ValueError(
                f"no {self.name!r} ids attached; limiter units are read "
                f"from {self.ids_path!r}"
            )
        return wall_units_from_ids(self.ids)

    @cached_property
    def boundary(self):
        """Return the sole occupiable vessel boundary for legacy consumers."""

        vessels = tuple(unit for unit in self.units if unit.kind == "vessel")
        if len(vessels) != 1:
            raise ValueError(
                f"wall carries {len(vessels)} vessel units; use units or segments"
            )
        return vessels[0].vertices

    def segment(self, index=0):
        """Return indexed firstwall segment."""
        return np.array(
            [
                self.vessel.unit[index].annular.centreline.r,
                self.vessel.unit[index].annular.centreline.z,
            ]
        ).T  # DDv4

    @cached_property
    def segments(self):
        """Return first wall segments."""
        return [unit.vertices for unit in self.units]

    @cached_property
    def outline(self):
        """Return first wall xz outline."""
        return {
            "x": [segment[:, 0] for segment in self.segments],
            "z": [segment[:, 1] for segment in self.segments],
        }

    def build(self):
        """Build plasma bound by firstwall contour."""
        self.firstwall.retain_units(self.units)
        for unit in self.units:
            if unit.kind == "vessel":
                self.firstwall.insert(unit.vertices)

    def insert(self, data: xarray.Dataset):
        """Insert wall and divertor geometory into dataset structure."""
        for i, (attr, segment) in enumerate(zip(["wall", "divertor"], self.segments)):
            data[attr] = (f"{attr}_index", "point"), segment
            index = np.arange(data[attr].shape[0])
            data.coords[f"{attr}_index"] = index
        data.attrs["wall_md"] = ",".join(
            [str(value) for _, value in self.ids_attrs.items()]
        )


@dataclass
class Magnetics(IdsData):
    """Manage magnetics diagnostic ids, magnetics."""

    pulse: int = 0
    run: int = 0
    occurrence: int = 0
    name: str = "magnetics"
    machine: str = "iter_md"
    ids_node: str = ""

    flux_loop_type: ClassVar[dict] = {"poloidal_flux_loop": 1}

    @cached_property
    def ids_index(self):
        """Return cached ids_index instance."""
        return IdsIndex(self.ids, self.ids_node)

    def extract(self):
        """Retun magnetics diagnostic geometry."""
        # extract data from IDS
        with self.ids_index.node("flux_loop"):
            name = self.ids_index.array("name")
            radius = self.ids_index.array("position.r")
            height = self.ids_index.array("position.z")
            type_index = self.ids_index.array("type.index")
        # build dataset
        diagnostic_data = {}
        for diagnostic_name, diagnostic_type in self.flux_loop_type.items():
            index = diagnostic_type == type_index
            if sum(index) == 0:
                continue
            diagnostic_data[diagnostic_name] = {
                "x": radius[index],
                "z": height[index],
                "name": name[index],
                "segment": "circle",
            }
        return diagnostic_data

    def insert(self, coilset: CoilSet):
        """Insert diagnostics into coilset."""
        for diagnostic, data in self.extract().items():
            getattr(coilset, diagnostic).insert(**data)


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
class Geometry(IdsBase):
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

    >>> geometry = Geometry(wall=False, pf_active='iter_md', pf_passive='iter_md')
    >>> geometry.wall
    False
    >>> geometry.pf_active == PoloidalFieldActive.default_ids_attrs()
    True
    >>> geometry.pf_passive == PoloidalFieldPassive.default_ids_attrs()
    True

    Modify pf_active attrs via dict input:

    >>> pf_active = Geometry(pf_active={'run': 101}).pf_active
    >>> pf_active == geometry.ids_attrs | {'run': 101, 'name': 'pf_active'}
    True

    Specify pf_active as an ids with pulse, run, and name:

    >>> database = Database(111001, 202, 'pf_active', machine='iter_md')
    >>> pf_active = Geometry(ids=database.ids, pf_active='iter_md').pf_active
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
    source_attrs: ClassVar[tuple[str, ...]] = (
        *IdsBase.database_attrs,
        "dd_version",
    )

    def __post_init__(self):
        """Map geometry parameters to dict attributes."""
        self.set_filename()
        for attr, geometry in self.geometry.items():
            source = getattr(self, attr)
            ids_attrs = self.get_ids_attrs(source, geometry)
            ids_attrs = self._carry_source_dd_version(source, ids_attrs)
            setattr(self, attr, ids_attrs)
        if hasattr(super(), "__post_init__"):
            super().__post_init__()

    def _carry_source_dd_version(self, source, ids_attrs):
        """Attach the source-specific data dictionary version to its descriptor."""
        if not isinstance(ids_attrs, dict):
            return ids_attrs
        if "dd_version" in ids_attrs:
            return ids_attrs | {"dd_version": str(ids_attrs["dd_version"])}
        if isinstance(source, IdsBase):
            dd_version = source.dd_version
        elif "ids" in ids_attrs:
            dd_version = Database(ids=ids_attrs["ids"]).ids_dd_version
        else:
            return ids_attrs
        return ids_attrs | {"dd_version": str(dd_version)}

    def set_filename(self):
        """Set filename when all geometry attrs are str or False."""
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

    @property
    def _dataset_attrs(self) -> list[str]:
        """Return list of dataset attributes names."""
        return list(self.geometry)

    @property
    def dataset_attrs(self) -> dict:
        """Return dataset attributes."""
        return {
            attr: (
                value
                if isinstance(value := getattr(self, attr), bool)
                or np.issubdtype(type(value), np.integer)
                or "ids" not in value
                else Database(ids=value["ids"]).ids_hash
            )
            for attr in self._dataset_attrs
        }


@dataclass
class Diagnostics(IdsBase):
    """Manage diagnostic mesurments."""

    magnetics: Ids | bool | str = False
    diagnostic: ClassVar[dict] = {"magnetics": Magnetics}

    def __post_init__(self):
        """Map geometry parameters to dict attributes."""
        for attr, diagnostic in self.diagnostic.items():
            ids_attrs = self.get_ids_attrs(getattr(self, attr), diagnostic)
            setattr(self, attr, ids_attrs)
        if hasattr(super(), "__post_init__"):
            super().__post_init__()


@dataclass
class Machine(CoilSet, Geometry, CoilData):  # Diagnostics,
    """Manage ITER machine geometry."""

    @property
    def metadata(self):
        """Manage machine metadata.

        Raises
        ------
            AssertionError:
                A change in the metadata group hash has been detected.
        """
        metadata = self.coilset_attrs
        for geometry in self.dataset_attrs:
            if (attrs := self.dataset_attrs[geometry]) is False:
                continue
            if isinstance(attrs, int | bytes | str):  # ids input
                metadata[geometry] = attrs
                continue
            metadata[geometry] = ",".join(
                str(attrs[attr]) for attr in self.source_attrs if attr in attrs
            )
        return metadata

    @metadata.setter
    def metadata(self, metadata: dict):
        """Set instance metadata, assert consistent attr_hash."""
        for attr in self.frameset_attrs:
            setattr(self, attr, metadata[attr])
        self._restore_route_attrs(
            {attr: metadata[attr] for attr in self._route_fields if attr in metadata}
        )
        for attr in metadata.keys() & self._biot_attrs.keys():
            setattr(self, attr, metadata[attr])
        for attr in self.dataset_attrs:
            if attr not in metadata:
                setattr(self, attr, False)
                continue
            if isinstance(metadata[attr], np.integer):
                setattr(self, attr, metadata[attr])
                continue
            source_attrs = self.source_attrs[: len(metadata[attr].split(","))]
            values = [
                self._format_dataset_attrs(name, value)
                for name, value in zip(
                    source_attrs, metadata[attr].split(","), strict=True
                )
            ]
            setattr(self, attr, dict(zip(source_attrs, values, strict=True)))
        assert self.group == self.hash_attrs(self.group_attrs)

    @staticmethod
    def _format_dataset_attrs(name: str, value: str) -> str | int | float:
        """Restore numeric source attributes while preserving DD versions."""
        if name == "dd_version":
            return value
        if "." in value:
            return float(value)
        try:
            return int(value)
        except ValueError:
            return value

    @property
    def group_attrs(self) -> dict:
        """
        Return group attrs.

        Extends :func:`~nova.imas.database.CoilData.group_attrs`. The data
        dictionary version is folded into the composite identity so a machine
        description assembled under a differing DD version cannot reuse a stale
        cache entry, mirroring the per-source fold in
        :func:`~nova.imas.database.Database.group_attrs`.
        """
        return (
            self.coilset_attrs
            | self.dataset_attrs
            | {"dd_version": str(self.dd_version)}
        )

    def solve_biot(self):
        """Solve biot instances."""
        if self.sloc["plasma"].sum() > 0:
            boundary = self.geometry["wall"](**self.wall).boundary
            self.plasma.solve(boundary=boundary)
        # self.poloidal_flux_loop.solve()
        self.inductance.solve()
        self.grid.solve(limit=self.limit)
        self.field.solve()
        self.force.solve()

    def build(self, **kwargs):
        """Build dataset, frameset and, biotset and save to file."""
        self.frameset_attrs = kwargs
        self.clear_frameset()
        for attr, geometry in self.geometry.items():
            geometry_attrs = getattr(self, attr)
            if isinstance(geometry_attrs, dict):
                geometry_attrs = {"dd_version": str(self.dd_version)} | geometry_attrs
                coilset = geometry(
                    **geometry_attrs, **self.frameset_attrs, **self.route_attrs
                )
                self += coilset
        """
        for attr, diagnostic in self.diagnostic.items():
            diagnostic_attrs = getattr(self, attr)
            if isinstance(diagnostic_attrs, dict):
                diagnostic(**diagnostic_attrs).insert(self)
        """

        if hasattr(super(), "build"):
            super().build()
        self.solve_biot()
        self.store()

    def load(self):
        """Load machine geometry and data."""
        super().load()
        self.metadata = self.data.attrs
        return self

    def store(self):
        """Store frameset, biot attributes and metadata."""
        expected_group = self.hash_attrs(self.group_attrs)
        if self.group != expected_group:
            raise ValueError(
                "machine cache identity changed after its group was selected"
            )
        self.data.attrs |= self.metadata
        super().store()
        return self


if __name__ == "__main__":
    import doctest

    doctest.testmod(verbose=False)

    kwargs = {"pulse": 105028, "run": 1, "machine": "iter_md"}  # DINA
    kwargs = {"pulse": 105028, "run": 1, "machine": "iter"}  # DINA
    # kwargs = {"pulse": 45272, "run": 1, "machine": "mast_u"}  # MastU
    # kwargs = {"pulse": 57410, "run": 0, "machine": "west"}  # WEST
    # kwargs = {"pulse": 17151, "run": 3, "machine": "aug", "pf_passive": False}

    machine = Machine(
        **kwargs,
        pf_active=True,
        pf_passive=False,
        wall=False,
        tplasma="h",
        ninductance=10,
        dshell=1.5,
        ngrid=2e3,
    )

    machine.plot()
    """
    machine = Machine(
        **kwargs,
        pf_active="iter_md",
        pf_passive="iter_md",
        elm=False,
        wall=True,
        tplasma="h",
        nwall=10,
        ninductance=10,
        dplasma=-2000,
        ngrid=2000,
    )
    """

    # from nova.imas.coils_non_axisymmetric import CoilsNonAxisymmetric

    # machine += CoilsNonAxisymmetric(111003, 2)  # CC
    # machine += CoilsNonAxisymmetric(115001, 1)  # ELM

    # machine.ferritic.insert("Fi")

    # machine.frame.vtkplot()
