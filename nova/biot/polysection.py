"""Exact Biot-Savart coupling for toroidal polygonal cross-sections.

Every production pair is integrated over the authored section. The closed-form
Part V reduction is the default; the boundary-quadrature implementation remains
an exact reference and compiled-device route. Approximate distance-banded and
point-filament comparators live in their dedicated study modules and cannot be
selected through this production element.

Quantities are per ampere of total conductor current, in raw SI: total poloidal
flux :math:`\\Phi = 2 \\pi R A_\\phi` [Wb] and field components [T].
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from functools import cached_property
import json
from typing import ClassVar, Literal

import numpy as np
from shapely.geometry import MultiPolygon, Polygon

from nova.biot.greens import section_centroid
from nova.biot.matrix import Matrix
from nova.biot.polygon import _N_NODES, _N_PANELS, polygon_greens
from nova.biot.polygonanalytic import (
    _horizontal_reflection,
    _section_centroid,
    polygon_analytic_field_moments,
    polygon_analytic_flux_moments,
    polygon_analytic_greens,
)
from nova.biot.sectionaverage import section_triangles


@dataclass(frozen=True)
class PolySectionPolicy:
    """Immutable exact-kernel policy for one polygon-section instance."""

    exact_kernel: Literal["closed_form", "quadrature"] = "closed_form"
    backend: Literal["numpy", "jax"] = "numpy"
    precision: Literal["float64"] = "float64"
    device_eligibility: Literal["host", "axisymmetric_ring"] = "host"
    quadrature: tuple[int, int] | None = None

    def __post_init__(self):
        """Validate and resolve every setting that changes kernel values."""
        if self.exact_kernel not in {"closed_form", "quadrature"}:
            raise ValueError(f"unknown polygon-section kernel {self.exact_kernel!r}")
        if self.backend not in {"numpy", "jax"}:
            raise ValueError(f"unsupported polygon-section backend {self.backend!r}")
        if self.precision != "float64":
            raise ValueError(
                f"unsupported polygon-section precision {self.precision!r}"
            )
        expected_device = "host" if self.backend == "numpy" else "axisymmetric_ring"
        if self.device_eligibility != expected_device:
            raise ValueError(
                f"the {self.backend!r} backend requires {expected_device!r} "
                "device eligibility"
            )
        if self.exact_kernel == "closed_form" and self.quadrature is not None:
            raise ValueError("closed-form routing does not accept a quadrature rule")
        if self.exact_kernel == "quadrature":
            rule = (_N_PANELS, _N_NODES) if self.quadrature is None else self.quadrature
            if len(rule) != 2 or any(
                isinstance(value, bool | np.bool_)
                or not isinstance(value, int | np.integer)
                or value <= 0
                for value in rule
            ):
                raise ValueError("quadrature must contain two positive integers")
            object.__setattr__(self, "quadrature", tuple(int(value) for value in rule))

    @property
    def key(self) -> str:
        """Return the canonical cache and source-batch identity."""
        return json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))

    @classmethod
    def resolve(
        cls, value: PolySectionPolicy | Mapping | str | None
    ) -> PolySectionPolicy:
        """Return a validated policy from an instance or persisted identity."""
        if value is None or value == "":
            return cls()
        if isinstance(value, cls):
            return value
        if isinstance(value, str):
            try:
                value = json.loads(value)
            except json.JSONDecodeError as error:
                raise ValueError("invalid polygon-section policy identity") from error
        if isinstance(value, Mapping):
            values = dict(value)
            if values.get("quadrature") is not None:
                values["quadrature"] = tuple(values["quadrature"])
            return cls(**values)
        raise TypeError(f"cannot resolve polygon-section policy from {type(value)}")


DEFAULT_POLYSECTION_POLICY = PolySectionPolicy().key


@dataclass
class PolySection(Matrix):
    """Couple complete toroidal conductors of arbitrary polygonal section.

    The section vertices come from each source element's own polygon, so a
    regular hexagonal plasma cell and a cell clipped by the first wall are
    handled by the same code path with no shape assumption.
    """

    axisymmetric: ClassVar[bool] = True
    name: ClassVar[str] = "polysection"
    policy: PolySectionPolicy | Mapping | str = field(default_factory=PolySectionPolicy)

    def __post_init__(self):
        """Resolve this element's private routing policy before evaluation."""
        self.policy = PolySectionPolicy.resolve(self.policy)
        super().__post_init__()

    @staticmethod
    def section_greens(
        target_r: np.ndarray,
        target_z: np.ndarray,
        vertices: np.ndarray,
        policy: PolySectionPolicy | Mapping | str | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return exact ``(psi, Br, Bz)`` per ampere over the authored section."""
        policy = PolySectionPolicy.resolve(policy)
        target_r = np.asarray(target_r, dtype=np.float64)
        target_z = np.asarray(target_z, dtype=np.float64)
        return PolySection.exact_greens(target_r, target_z, vertices, policy)

    @staticmethod
    def exact_greens(
        target_r: np.ndarray,
        target_z: np.ndarray,
        vertices: np.ndarray,
        policy: PolySectionPolicy | Mapping | str | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return ``(psi, Br, Bz)`` from the configured exact kernel.

        This is the single production selection point for the two exact
        implementations.
        """
        policy = PolySectionPolicy.resolve(policy)
        if policy.exact_kernel == "closed_form":
            return polygon_analytic_greens(target_r, target_z, vertices)
        rule = dict(zip(("n_panels", "n_nodes"), policy.quadrature))
        return polygon_greens(target_r, target_z, vertices, **rule)

    @staticmethod
    def _material_geometry(value) -> Polygon | MultiPolygon:
        """Return validated Shapely material from a Nova section wrapper."""
        geometry = value
        visited: set[int] = set()
        while not isinstance(geometry, Polygon | MultiPolygon):
            if id(geometry) in visited or not hasattr(geometry, "poly"):
                raise ValueError("polygon-section source requires polygonal material")
            visited.add(id(geometry))
            geometry = geometry.poly
        if geometry.is_empty or not geometry.is_valid or geometry.area <= 0.0:
            raise ValueError(
                "polygon-section source material must be valid and positive"
            )
        return geometry

    @staticmethod
    def _material_area_centroid(material: Polygon | MultiPolygon) -> np.ndarray:
        """Return the authored material's area centroid as an expansion point."""
        if isinstance(material, Polygon) and len(material.interiors) == 0:
            vertices = np.asarray(material.exterior.coords, dtype=np.float64)[:-1, :2]
            return section_centroid(vertices)
        return np.asarray(material.centroid.coords[0], dtype=np.float64)

    @cached_property
    def _section_components(self) -> list[tuple[tuple[np.ndarray, float], ...]]:
        """Return positive simple components and normalized source-current weights."""
        sections = []
        for value in np.asarray(self.source["poly"]):
            geometry = self._material_geometry(value)
            if isinstance(geometry, Polygon) and len(geometry.interiors) == 0:
                points = np.asarray(geometry.exterior.coords, dtype=np.float64)[:-1, :2]
                sections.append(((points, 1.0),))
                continue
            triangles, area = section_triangles(geometry)
            total = float(area.sum())
            sections.append(
                tuple(
                    (vertices, float(weight / total))
                    for vertices, weight in zip(triangles, area)
                )
            )
        return sections

    @cached_property
    def _coupling(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return the ``(psi, Br, Bz)`` matrices, shape ``(target, source)``."""
        target_r = np.asarray(self.target("r"))
        target_z = np.asarray(self.target("z"))
        psi = np.empty(target_r.shape)
        br = np.empty(target_r.shape)
        bz = np.empty(target_r.shape)
        for column, components in enumerate(self._section_components):
            psi[:, column] = 0.0
            br[:, column] = 0.0
            bz[:, column] = 0.0
            for vertices, weight in components:
                component = self.section_greens(
                    target_r[:, column], target_z[:, column], vertices, self.policy
                )
                psi[:, column] += weight * component[0]
                br[:, column] += weight * component[1]
                bz[:, column] += weight * component[2]
        return psi, br, bz

    @cached_property
    def _moment_coupling(self) -> tuple[np.ndarray, ...]:
        """Return radial and vertical companion matrices for all three kernels."""
        target_r = np.asarray(self.target("r"))
        target_z = np.asarray(self.target("z"))
        companions = tuple(np.zeros(target_r.shape) for _ in range(6))
        for column, (source_value, components) in enumerate(
            zip(np.asarray(self.source["poly"]), self._section_components, strict=True)
        ):
            material = self._material_geometry(source_value)
            expansion_point = self._material_area_centroid(material)
            for vertices, weight in components:
                flux = polygon_analytic_flux_moments(
                    target_r[:, column],
                    target_z[:, column],
                    vertices,
                    expansion_point=expansion_point,
                )
                radial, vertical = polygon_analytic_field_moments(
                    target_r[:, column],
                    target_z[:, column],
                    vertices,
                    expansion_point=expansion_point,
                )
                rows = (
                    flux[1],
                    flux[2],
                    radial[1],
                    radial[2],
                    vertical[1],
                    vertical[2],
                )
                for output, row in zip(companions, rows, strict=True):
                    output[:, column] += weight * row
        return companions

    @cached_property
    def Psi(self):
        """Return the total poloidal flux array [Wb/A]."""
        return self._coupling[0]

    @cached_property
    def PsiR(self):
        """Return the radial flux-moment companion [m Wb/A]."""
        return self._moment_coupling[0]

    @cached_property
    def PsiZ(self):
        """Return the vertical flux-moment companion [m Wb/A]."""
        return self._moment_coupling[1]

    @cached_property
    def Aphi(self):
        """Return the toroidal vector potential array [Wb/(m.A)]."""
        radius = np.asarray(self.target("r"))
        potential = np.zeros_like(self.Psi)
        np.divide(
            self.Psi,
            2 * np.pi * self.mu_0 * radius,
            out=potential,
            where=radius != 0,
        )
        return potential

    @cached_property
    def Br(self):
        """Return the radial field array [T/A]."""
        return self._coupling[1]

    @cached_property
    def BrR(self):
        """Return the radial moment companion of radial field [m T/A]."""
        return self._moment_coupling[2]

    @cached_property
    def BrZ(self):
        """Return the vertical moment companion of radial field [m T/A]."""
        return self._moment_coupling[3]

    @cached_property
    def Bz(self):
        """Return the vertical field array [T/A]."""
        return self._coupling[2]

    @cached_property
    def BzR(self):
        """Return the radial moment companion of vertical field [m T/A]."""
        return self._moment_coupling[4]

    @cached_property
    def BzZ(self):
        """Return the vertical moment companion of vertical field [m T/A]."""
        return self._moment_coupling[5]


@dataclass
class TiledPolySection(PolySection):
    """Evaluate complete-ring sections through the compiled quadrature tile kernel.

    This is an explicit product adapter around the existing ring kernel. It keeps
    the ordinary :class:`Matrix` turn, target-quadrature and row-reduction contract;
    :class:`~nova.biot.solve.Solve` performs the complete source electrical
    reduction after each bounded route batch returns. Historical finite arcs use
    their geometry-specific host elements and never enter this adapter.

    Complex material is expanded into positive simple triangles before packing.
    Their area fractions are accumulated back onto the authored source column, so
    holes carry no current and disconnected components retain one electrical
    identity.
    """

    _tile_side: ClassVar[int] = 40

    def __post_init__(self):
        """Require the explicit accelerator route before allocating an evaluator."""
        super().__post_init__()
        if (
            self.policy.backend != "jax"
            or self.policy.device_eligibility != "axisymmetric_ring"
        ):
            raise ValueError("tiled polygon sections require the JAX ring policy")

    def build_transform(self):
        """Avoid the unused target-by-source Cartesian transform allocation."""
        self.coordinate_axes = np.empty((0, 3, 3), dtype=np.float64)
        self.coordinate_origin = np.empty((0, 3), dtype=np.float64)

    @cached_property
    def _packed_sections(self) -> tuple[list[np.ndarray], np.ndarray, np.ndarray]:
        """Return simple sections, authored owners and positive current fractions."""
        sections = []
        owner = []
        fraction = []
        for column, components in enumerate(self._section_components):
            for vertices, weight in components:
                sections.append(vertices)
                owner.append(column)
                fraction.append(weight)
        return (
            sections,
            np.asarray(owner, dtype=np.intp),
            np.asarray(fraction, dtype=np.float64),
        )

    @cached_property
    def _closed_coupling(self) -> tuple[np.ndarray, ...]:
        """Return all nine exact rows from one fixed-shape traced evaluator."""
        from nova.biot.polygon import pad_batch
        from nova.biot.tiledassembly import TilePlan, tile_evaluator

        target_r = np.hypot(
            np.asarray(self.target.x, dtype=np.float64),
            np.asarray(self.target.y, dtype=np.float64),
        )
        target_z = np.asarray(self.target.z, dtype=np.float64)
        sections, owner, fraction = self._packed_sections
        outputs = np.zeros((9, target_r.size, len(self.source)), dtype=np.float64)
        if target_r.size == 0 or not sections:
            return tuple(outputs)

        edge, weight, norm = pad_batch(sections)
        section_centre = np.column_stack(
            [_section_centroid(vertices) for vertices in sections]
        )
        authored_centre = np.column_stack(
            [
                self._material_area_centroid(
                    self._material_geometry(np.asarray(self.source["poly"])[column])
                )
                for column in owner
            ]
        )
        reflection_axis = np.full(len(sections), np.nan, dtype=np.float64)
        reflection_partner = np.repeat(
            np.arange(edge.shape[0], dtype=np.intp)[:, None], len(sections), axis=1
        )
        for column, vertices in enumerate(sections):
            reflection = _horizontal_reflection(vertices)
            if reflection is None:
                continue
            axis, vertex_partner = reflection
            reflection_axis[column] = axis
            count = len(vertices)
            for index in range(count):
                reflection_partner[index, column] = vertex_partner[(index + 1) % count]

        target_tile = min(self._tile_side, target_r.size)
        source_tile = min(self._tile_side, len(sections))
        plan = TilePlan(
            target_tile=target_tile,
            source_tile=source_tile,
            block=target_tile * source_tile,
            n_panels=16,
            n_nodes=48,
        )
        evaluate = tile_evaluator(
            plan,
            batched=True,
            kernel="moments",
            precision=self.policy.precision,
            edge_count=edge.shape[0],
        )
        for rows, columns in plan.tiles(target_r.size, len(sections)):
            tile = evaluate(
                target_r[rows],
                target_z[rows],
                edge[:, :, columns],
                weight[:, columns],
                norm[columns],
                section_centre[:, columns],
                authored_centre[:, columns],
                reflection_axis[columns],
                reflection_partner[:, columns],
            )
            tile_owner = owner[columns]
            tile_fraction = fraction[columns]
            for result, values in zip(outputs, tile, strict=True):
                reduced = np.zeros(
                    (rows.stop - rows.start, len(self.source)), dtype=np.float64
                )
                np.add.at(
                    reduced.T,
                    tile_owner,
                    (values * tile_fraction[np.newaxis, :]).T,
                )
                result[rows] += reduced
        return tuple(outputs)

    @cached_property
    def _coupling(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return packed tile results accumulated onto authored source columns."""
        if self.policy.exact_kernel == "closed_form":
            return tuple(self._closed_coupling[index] for index in (0, 3, 6))
        from nova.biot.polygon import pad_batch
        from nova.biot.polygonanalytic import polygon_analytic_greens
        from nova.biot.tiledassembly import TilePlan, tile_evaluator

        target_r = np.hypot(
            np.asarray(self.target.x, dtype=np.float64),
            np.asarray(self.target.y, dtype=np.float64),
        )
        target_z = np.asarray(self.target.z, dtype=np.float64)
        sections, owner, fraction = self._packed_sections
        if target_r.size == 0 or not sections:
            empty = np.zeros((target_r.size, len(self.source)), dtype=np.float64)
            return empty.copy(), empty.copy(), empty.copy()

        components = np.zeros((3, target_r.size, len(self.source)), dtype=np.float64)
        axis_positions = np.flatnonzero(target_r == 0.0)
        if len(axis_positions) > 0:
            # The quadrature field is a flux gradient divided by radius.  Keep
            # symmetry-axis rows out of that traced graph and use the same finite
            # closed reduction as the host quadrature route before accumulating
            # material pieces back to their authored source columns.
            for vertices, column, current_fraction in zip(sections, owner, fraction):
                values = polygon_analytic_greens(
                    target_r[axis_positions], target_z[axis_positions], vertices
                )
                for result, value in zip(components, values):
                    result[axis_positions, column] += current_fraction * value

        off_axis_positions = np.flatnonzero(target_r != 0.0)
        if len(off_axis_positions) == 0:
            return tuple(components)

        n_panels, n_nodes = (
            self.policy.quadrature
            if self.policy.exact_kernel == "quadrature"
            else (_N_PANELS, _N_NODES)
        )
        edge, weight, norm = pad_batch(sections)
        plan = TilePlan(
            target_tile=min(self._tile_side, len(off_axis_positions)),
            source_tile=min(self._tile_side, len(sections)),
            block=16,
            n_panels=n_panels,
            n_nodes=n_nodes,
        )
        evaluate = tile_evaluator(
            plan,
            batched=True,
            kernel="quadrature",
            precision=self.policy.precision,
            edge_count=edge.shape[0],
        )
        for rows, columns in plan.tiles(len(off_axis_positions), len(sections)):
            authored_rows = off_axis_positions[rows]
            tile = evaluate(
                target_r[authored_rows],
                target_z[authored_rows],
                edge[:, :, columns],
                weight[:, columns],
                norm[columns],
            )
            tile_owner = owner[columns]
            tile_fraction = fraction[columns]
            for result, values in zip(components, tile):
                reduced = np.zeros(
                    (len(authored_rows), len(self.source)), dtype=np.float64
                )
                np.add.at(
                    reduced.T,
                    tile_owner,
                    (values * tile_fraction[np.newaxis, :]).T,
                )
                result[authored_rows] += reduced
        return tuple(components)

    @cached_property
    def _moment_coupling(self) -> tuple[np.ndarray, ...]:
        """Return traced companions, retaining the host quadrature reference."""
        if self.policy.exact_kernel == "closed_form":
            return tuple(self._closed_coupling[index] for index in (1, 2, 4, 5, 7, 8))
        return super()._moment_coupling

    @cached_property
    def Aphi(self):
        """Return finite zero vector potential on the magnetic axis."""
        radius = np.hypot(
            np.asarray(self.target.x, dtype=np.float64),
            np.asarray(self.target.y, dtype=np.float64),
        )[:, np.newaxis]
        potential = np.zeros_like(self.Psi)
        np.divide(
            self.Psi,
            2 * np.pi * self.mu_0 * radius,
            out=potential,
            where=radius != 0.0,
        )
        return potential
