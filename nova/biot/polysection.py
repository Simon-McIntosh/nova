"""Biot-Savart coupling for toroidal conductors of polygonal cross-section.

A point-filament ring is log-singular at its own location, so any target that
approaches a conductor — a neighbouring plasma cell, or the cell itself on the
diagonal of the plasma-plasma matrix — inherits a spurious near-field spike.
Spreading the current uniformly over the true cross-section removes the
singularity: the flux and field stay finite and smooth through the conductor.
:func:`nova.biot.polygon.polygon_greens` does that for an arbitrary polygon,
which is what a hexagonal (or wall-clipped) plasma cell needs.

Near and far
------------
The exact polygon kernel integrates over the section boundary at every target,
so it costs roughly four orders of magnitude more per target-source pair than
the point-filament form. Beyond a few section sizes it also buys nothing: the
finite-area correction there is the constant second-moment term, a few tenths of
a percent in flux and far below any measurement it is compared against. This
element therefore evaluates the exact kernel only inside a standoff band of
``standoff`` section radii and the point-filament form outside it — the same
near/far contract :func:`nova.biot.greens.hybrid_greens` already applies to
rectangular sections, generalised to a polygon. On a plasma grid the band holds
a few percent of the target-source pairs, which is what makes the exact
treatment affordable where it is physically real.

Three bands
-----------
The two-way near/far split above is a study knob, not the shipped default,
because a bare point filament does not converge to a finite section for a full
ring at any standoff. The measured alternative is the three-band scheme in
:mod:`nova.biot.bandedcoupling` — converged rule, reduced rule, moment-corrected
filament, binned by distance to the section contour — which holds every component
to one part in a million of its peak. It is available here through ``banded`` and
is off by default: exact everywhere remains the shipped lane and the reference
the banded one is measured against.

Two exact kernels
-----------------
Independently of how pairs are binned, the *exact* treatment itself comes two
ways. :func:`nova.biot.polygon.polygon_greens` reduces the section integral to a
contour sum and does the remaining angular integral by quadrature;
:func:`nova.biot.polygonanalytic.polygon_analytic_greens` does that integral in
closed form as well, leaving only two smooth ``arsinh`` residuals per corner.
``closed_form`` selects the second, and it composes with either binning — exact
everywhere becomes closed-form everywhere, and the three-band scheme's near band
takes the closed form where the quadrature's own singularity is.

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

from nova.biot.bandedcoupling import banded_greens
from nova.biot.greens import greens_bz_br, greens_psi, section_centroid
from nova.biot.matrix import Matrix
from nova.biot.polygon import _N_NODES, _N_PANELS, polygon_greens
from nova.biot.polygonanalytic import polygon_analytic_greens
from nova.biot.sectionaverage import section_triangles


@dataclass(frozen=True)
class PolySectionPolicy:
    """Immutable routing policy for one polygon-section kernel instance."""

    arrangement: Literal["exact", "standoff", "banded", "filament"] = "exact"
    exact_kernel: Literal["closed_form", "quadrature"] = "closed_form"
    backend: Literal["numpy", "jax"] = "numpy"
    precision: Literal["float64"] = "float64"
    device_eligibility: Literal["host", "axisymmetric_ring"] = "host"
    standoff: float | None = None
    quadrature: tuple[int, int] | None = None

    def __post_init__(self):
        """Validate and resolve every setting that changes kernel values."""
        if self.arrangement not in {"exact", "standoff", "banded", "filament"}:
            raise ValueError(
                f"unknown polygon-section arrangement {self.arrangement!r}"
            )
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
        if self.backend == "jax" and self.arrangement != "exact":
            raise ValueError("the tiled ring backend evaluates only exact routing")
        if self.backend == "jax" and self.exact_kernel != "quadrature":
            raise ValueError(
                "the tiled ring backend requires the compiled quadrature kernel"
            )
        if self.arrangement == "filament" and self.exact_kernel != "closed_form":
            raise ValueError("filament routing does not accept an exact kernel")

        if self.arrangement == "standoff":
            if isinstance(self.standoff, bool | np.bool_) or not isinstance(
                self.standoff, int | float | np.integer | np.floating
            ):
                raise ValueError("standoff routing requires a finite distance")
            standoff = float(self.standoff)
            if not np.isfinite(standoff):
                raise ValueError("standoff routing requires a finite distance")
            if standoff <= 0:
                raise ValueError("standoff distance must be positive")
            object.__setattr__(self, "standoff", standoff)
        elif self.standoff is not None:
            raise ValueError(
                f"standoff has no meaning for {self.arrangement!r} routing"
            )

        if self.exact_kernel == "closed_form" and self.quadrature is not None:
            raise ValueError("closed-form routing does not accept a quadrature rule")
        if self.arrangement == "banded" and self.quadrature is not None:
            raise ValueError("banded routing owns its fixed near and middle rules")
        if self.exact_kernel == "quadrature" and self.arrangement in {
            "exact",
            "standoff",
        }:
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
    def section_radius(vertices: np.ndarray) -> float:
        """Return the section's bounding radius about its area centroid [m]."""
        vertices = np.asarray(vertices, dtype=np.float64)
        centre = section_centroid(vertices)
        return float(np.max(np.hypot(*(vertices - centre).T)))

    @staticmethod
    def near_band(
        target_r: np.ndarray,
        target_z: np.ndarray,
        vertices: np.ndarray,
        policy: PolySectionPolicy | Mapping | str | None = None,
    ) -> np.ndarray:
        """Return the mask of targets inside the section's standoff band.

        Every target is inside the band when the standoff is ``None`` or
        infinite, which is how *exact everywhere* is expressed.
        """
        target_r = np.asarray(target_r, dtype=np.float64)
        policy = PolySectionPolicy.resolve(policy)
        if policy.arrangement == "filament":
            return np.zeros(target_r.shape, dtype=bool)
        if policy.arrangement != "standoff":
            return np.ones(target_r.shape, dtype=bool)
        vertices = np.asarray(vertices, dtype=np.float64)
        centre = section_centroid(vertices)
        distance = np.hypot(
            target_r - centre[0],
            np.asarray(target_z, dtype=np.float64) - centre[1],
        )
        return distance < policy.standoff * PolySection.section_radius(vertices)

    @staticmethod
    def section_greens(
        target_r: np.ndarray,
        target_z: np.ndarray,
        vertices: np.ndarray,
        policy: PolySectionPolicy | Mapping | str | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return ``(psi, Br, Bz)`` per ampere: exact near the section, point far.

        The returned arrays are shaped like ``target_r``. With ``banded`` set, the
        three-band scheme handles every pair instead and ``standoff`` does not
        apply — the bands carry their own measured rules. ``closed_form`` selects
        which exact kernel serves either arrangement, and ``quadrature`` applies
        only to the boundary-quadrature one.
        """
        policy = PolySectionPolicy.resolve(policy)
        target_r = np.asarray(target_r, dtype=np.float64)
        target_z = np.asarray(target_z, dtype=np.float64)
        if policy.arrangement == "banded":
            return banded_greens(
                target_r,
                target_z,
                vertices,
                closed_form=policy.exact_kernel == "closed_form",
            )
        psi = np.empty(target_r.shape)
        br = np.empty(target_r.shape)
        bz = np.empty(target_r.shape)
        near = PolySection.near_band(target_r, target_z, vertices, policy)
        if near.any():
            psi[near], br[near], bz[near] = PolySection.exact_greens(
                target_r[near], target_z[near], vertices, policy
            )
        far = ~near
        if far.any():
            centre = section_centroid(vertices)
            psi[far] = greens_psi(target_r[far], target_z[far], centre[0], centre[1])
            bz[far], br[far] = greens_bz_br(
                target_r[far], target_z[far], centre[0], centre[1]
            )
        return psi, br, bz

    @staticmethod
    def exact_greens(
        target_r: np.ndarray,
        target_z: np.ndarray,
        vertices: np.ndarray,
        policy: PolySectionPolicy | Mapping | str | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return ``(psi, Br, Bz)`` from the configured exact kernel.

        The one place the two evaluations are chosen between, so the standoff
        band, the banded scheme's near band and a direct call cannot disagree
        about which one is in force.
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
            if self.policy.arrangement == "filament":
                centre = sum(
                    weight * section_centroid(vertices)
                    for vertices, weight in components
                )
                psi[:, column] = greens_psi(
                    target_r[:, column], target_z[:, column], centre[0], centre[1]
                )
                bz[:, column], br[:, column] = greens_bz_br(
                    target_r[:, column], target_z[:, column], centre[0], centre[1]
                )
                continue
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
    def Psi(self):
        """Return the total poloidal flux array [Wb/A]."""
        return self._coupling[0]

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
    def Bz(self):
        """Return the vertical field array [T/A]."""
        return self._coupling[2]


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
    def _coupling(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return packed tile results accumulated onto authored source columns."""
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
