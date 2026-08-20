"""Demonstrate free-boundary forward solves against labelled DIII-D maps.

The labelled flux and q95 cross the measured corpus convention boundary once,
before any Nova kernel sees them.  Each selected diverted frame supplies its
own deterministically extracted p-prime and FF-prime functions and the shipped
nineteen-conductor state.  ``ForwardProfile`` then solves the unmodified
free-boundary problem.  There is no response fit; the only alignment applied
when scoring flux is its physically arbitrary additive gauge.

The competition data do not ship a machine wall.  The rectangular enclosing
surface used by the topology read is therefore labelled a pseudo-wall.  It is
derived only from the released EFIT-grid extent, and the first frame is solved
again with one outward displacement to expose the resulting sensitivity.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from contourpy import contour_generator
from matplotlib.path import Path as PolygonPath
from scipy.constants import mu_0
from scipy.interpolate import RectBivariateSpline
from scipy.spatial import cKDTree

from benchmarks.diiid_corpus_conventions import (
    CORPUS_COCOS,
    NOVA_COCOS,
    corpus_f_to_nova,
    corpus_flux_to_nova_total,
    corpus_q_to_nova,
)
from nova.biot.greens import hybrid_greens
from nova.biot.polygon import polygon_greens
from nova.equilibrium.conservation import FluxLattice
from nova.equilibrium.flux_surface_geometry import (
    FluxSurfaceGeometry,
    SurfaceGeometryError,
)
from nova.equilibrium.forward import ForwardProfile
from nova.equilibrium.map_extraction import extract_flux_functions
from nova.equilibrium.source import DomainProfile, ForwardSource
from nova.imas.diiid_description import (
    POLOIDAL_CONDUCTORS,
    dataset_machine_description,
    geometry_digest,
    vacuum_response,
)
from nova.jax.config import configure_dtypes

DEFAULT_DATA = Path("/work/projects/imas_gpu/sophelio/raw/data/diii_d_train")
DEFAULT_OUTPUT = Path("docs/figures/diiid-forward-onboarding/forward-gs")
PREREGISTRATION_NAME = "forward_gs_preregistration.json"
RECEIPT_NAME = "forward_gs_receipt.json"
FRAME_FIGURE_NAME = "frame_flux_comparison.png"
COHORT_FIGURE_NAME = "cohort_match_summary.png"

LABEL_REPRESENTABILITY_MEDIAN_R2 = 0.949
IRREDUCIBLE_LABEL_RESIDUAL_FRACTION = 0.9968
RETAINED_CEILING_FRACTION = 0.95
REGISTERED_MEDIAN_INTERIOR_R2_BAR = (
    RETAINED_CEILING_FRACTION * LABEL_REPRESENTABILITY_MEDIAN_R2
)
REGISTERED_FRAME_COUNT = 3
REGISTERED_GRID_STRIDE = 2
REGISTERED_RESIDUAL_TOLERANCE = 1.0e-5
REGISTERED_SOLVER_ROUTE = "host"
REGISTERED_HOST_TOLERANCE = 1.0e-8
REGISTERED_HOST_MAXIMUM_EVALUATIONS = 1701
REGISTERED_HOST_INITIAL_RELAXATION = 0.2
REGISTERED_HOST_MINIMUM_RELAXATION = 1.0e-6
REGISTERED_HOST_RELAXATION_REDUCTION_INTERVAL = 100
REGISTERED_BASELINE_PSEUDO_WALL_EXPANSION = 0.02
PSEUDO_WALL_EXPANSIONS = (0.02, 0.05)
TOROIDAL_FIELD_TURNS = 144
TOROIDAL_FIELD_TURNS_SOURCE = "https://fusion.gat.com/pubs-ext/SOFT02/A24059.pdf"

_GEOMETRY_COLUMNS = (
    "coil_name",
    "coil_input_column",
    "coil_R",
    "coil_Z",
    "coil_width",
    "coil_height",
    "coil_angle1",
    "coil_angle2",
    "thomson_chord_name",
    "thomson_chord_R",
    "thomson_chord_Z",
)
_LABEL_COLUMNS = (
    "efit_times",
    "efit_psirz",
    "efit_q95",
    "efit_r_axis",
    "efit_z_axis",
    "efit_lcfs_n",
    "efit_lcfs_r",
    "efit_lcfs_z",
    "efit_grid_R",
    "efit_grid_Z",
    "magnetics_dsep",
    "magnetics_dsep_times",
)
_CURRENT_COLUMNS = ("magnetics_time",) + tuple(
    f"magnetics_{name}" for name in (*POLOIDAL_CONDUCTORS, "bcoil")
)
_COUPLING_CACHE: dict[
    tuple[str, float],
    tuple[tuple[str, ...], np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray],
] = {}


@dataclass(frozen=True)
class SelectedFrame:
    """One corpus frame selected without reference to its eventual score."""

    path: Path
    frame: int
    time_ms: float


@dataclass(frozen=True)
class MatchMetrics:
    """The predeclared comparison quantities for one forward solve."""

    interior_r_squared: float
    interior_fractional_rms: float
    additive_gauge_wb: float
    separatrix_mean_radial_separation_mm: float
    separatrix_maximum_radial_separation_mm: float
    magnetic_axis_displacement_mm: float
    predicted_q95_nova: float
    labelled_q95_nova: float
    signed_relative_q95_error: float


@dataclass(frozen=True)
class FrameResult:
    """One scored map and its convergence qualification."""

    shot: str
    frame: int
    time_ms: float
    geometry_digest: str
    reliable_flux_surfaces: int
    pseudo_wall_expansion: float
    pseudo_wall_statement: str
    fixed_point_relative_residual: float
    residual_tolerance: float
    finite: bool
    diverted: bool
    converged: bool
    convergence_criterion: str
    solver_termination: str
    metrics: MatchMetrics


def preregistration() -> dict[str, Any]:
    """Return the immutable cohort, solver, and pass declaration."""

    return {
        "measurement": "DIII-D labelled-map free-boundary forward GS match",
        "corpus_cocos": CORPUS_COCOS,
        "nova_cocos": NOVA_COCOS,
        "selection": {
            "paths": "lexicographic corpus order",
            "frame": "eligible time nearest the median eligible labelled time",
            "eligibility": (
                "finite map and q95, at least eight LCFS points, positive recorded "
                "DSEP; no score-dependent filtering"
            ),
            "frames": REGISTERED_FRAME_COUNT,
            "cohort_reduction": (
                "reduced from eight to three before scoring because the host "
                "root route is substantially more expensive; the convergence "
                "criterion and 0.90155 score bar are unchanged"
            ),
        },
        "source": (
            "extract_flux_functions(label) p-prime and FF-prime plus the shipped "
            "nineteen poloidal-conductor currents; zero fitted coefficients"
        ),
        "solver": {
            "entry_point": "nova.equilibrium.forward.ForwardProfile",
            "route": REGISTERED_SOLVER_ROUTE,
            "grid_stride": REGISTERED_GRID_STRIDE,
            "solver_tolerance": REGISTERED_HOST_TOLERANCE,
            "maximum_evaluations": REGISTERED_HOST_MAXIMUM_EVALUATIONS,
            "initial_relaxation": REGISTERED_HOST_INITIAL_RELAXATION,
            "minimum_relaxation": REGISTERED_HOST_MINIMUM_RELAXATION,
            "relaxation_reduction_interval": (
                REGISTERED_HOST_RELAXATION_REDUCTION_INTERVAL
            ),
            "relative_residual_tolerance": REGISTERED_RESIDUAL_TOLERANCE,
            "seed": "the convention-clean labelled map, used only as a branch seed",
            "policy_history": [
                {
                    "route": "newton_krylov",
                    "newton_steps": 4,
                    "gmres_iterations": 8,
                    "warmup_sweeps": 0,
                    "outcome": (
                        "quarantined: unchanged label seed at relative residual "
                        "1.057120826; no match score claimed"
                    ),
                },
                {
                    "route": "host_krylov",
                    "function_tolerance": 1.0e-8,
                    "maximum_iterations": 20,
                    "outcome": (
                        "quarantined: iteration budget exhausted on a diverted "
                        "finite state at relative residual 1.857399452; no match "
                        "score claimed"
                    ),
                },
                {
                    "route": REGISTERED_SOLVER_ROUTE,
                    "solver_tolerance": REGISTERED_HOST_TOLERANCE,
                    "maximum_evaluations": REGISTERED_HOST_MAXIMUM_EVALUATIONS,
                    "initial_relaxation": REGISTERED_HOST_INITIAL_RELAXATION,
                    "minimum_relaxation": REGISTERED_HOST_MINIMUM_RELAXATION,
                    "relaxation_reduction_interval": (
                        REGISTERED_HOST_RELAXATION_REDUCTION_INTERVAL
                    ),
                    "status": "active scoring policy",
                },
            ],
        },
        "score": {
            "region": "label-LCFS interior sampled on the forward lattice",
            "gauge": "one additive constant over that interior",
            "coefficients_fitted": 0,
            "registered_median_interior_r_squared_bar": (
                REGISTERED_MEDIAN_INTERIOR_R2_BAR
            ),
            "bar_basis": {
                "measured_label_representability_median_r_squared": (
                    LABEL_REPRESENTABILITY_MEDIAN_R2
                ),
                "strict_gs_residual_attributed_to_irreducible_non_gs_content": (
                    IRREDUCIBLE_LABEL_RESIDUAL_FRACTION
                ),
                "fraction_of_measured_ceiling_retained": RETAINED_CEILING_FRACTION,
                "discretisation_floor_used": False,
            },
        },
        "pseudo_wall": {
            "statement": (
                "rectangular enclosing control surface derived from efit_grid; "
                "a pseudo-wall standing in for the absent machine wall"
            ),
            "baseline_expansion": REGISTERED_BASELINE_PSEUDO_WALL_EXPANSION,
            "sensitivity_expansions": list(PSEUDO_WALL_EXPANSIONS),
            "sensitivity_frame": "first preregistered frame only",
        },
        "q95": {
            "definition": "FluxSurfaceGeometry safety factor at psi_norm 0.95",
            "toroidal_field_model": (
                "ideal 144-turn DIII-D toroidal coil, F=mu0*N*I/(2*pi)"
            ),
            "turns_source": TOROIDAL_FIELD_TURNS_SOURCE,
            "coefficient_fitted": False,
            "assumption_scope": (
                "q95 scales linearly with F and therefore with the external "
                "144-turn assumption; flux, separatrix and magnetic-axis metrics "
                "do not depend on that turn count"
            ),
        },
        "convention_crossings": {
            "efit_psirz": "corpus_flux_to_nova_total exactly once",
            "efit_q95": "corpus_q_to_nova exactly once",
            "toroidal_field_function": "corpus_f_to_nova exactly once",
            "p_prime_and_ff_prime": (
                "extracted after the flux crossing; already native Nova quantities"
            ),
        },
    }


def write_preregistration(output: Path) -> Path:
    """Write the scoring declaration before any frame may be solved."""

    output.mkdir(parents=True, exist_ok=True)
    path = output / PREREGISTRATION_NAME
    path.write_text(json.dumps(preregistration(), indent=2, sort_keys=True) + "\n")
    return path


def require_preregistration(path: Path) -> str:
    """Fail closed unless the on-disk declaration matches this runner."""

    if not path.is_file():
        raise RuntimeError("the forward-GS bar must be preregistered before scoring")
    encoded = path.read_bytes()
    if json.loads(encoded) != preregistration():
        raise RuntimeError("the on-disk forward-GS preregistration does not match")
    return hashlib.sha256(encoded).hexdigest()


def _read(path: Path, columns: tuple[str, ...]) -> dict[str, Any]:
    try:
        import pyarrow.parquet as parquet
    except ImportError as error:
        raise RuntimeError(
            "the corpus benchmark requires a pyarrow-enabled runner"
        ) from error
    table = parquet.read_table(path, columns=list(dict.fromkeys(columns)))
    return {name: table[name][0].as_py() for name in table.column_names}


def _eligible_frame(row: dict[str, Any]) -> int | None:
    times = np.asarray(row["efit_times"], dtype=float)
    q95 = np.asarray(row["efit_q95"], dtype=float)
    counts = np.asarray(row["efit_lcfs_n"], dtype=int)
    dsep = np.interp(
        times,
        np.asarray(row["magnetics_dsep_times"], dtype=float),
        np.asarray(row["magnetics_dsep"], dtype=float),
    )
    finite_maps = np.asarray(
        [np.all(np.isfinite(frame)) for frame in row["efit_psirz"]], dtype=bool
    )
    eligible = np.flatnonzero(
        finite_maps & np.isfinite(q95 + dsep) & (counts >= 8) & (dsep > 0.0)
    )
    if eligible.size == 0:
        return None
    middle_time = np.median(times[eligible])
    return int(eligible[np.argmin(np.abs(times[eligible] - middle_time))])


def select_frames(paths: list[Path], count: int) -> list[SelectedFrame]:
    """Select the declared cohort without looking at a match metric."""

    selected: list[SelectedFrame] = []
    for path in paths:
        row = _read(path, _LABEL_COLUMNS)
        frame = _eligible_frame(row)
        if frame is None:
            continue
        selected.append(SelectedFrame(path, frame, float(row["efit_times"][frame])))
        if len(selected) == count:
            break
    if len(selected) < count:
        raise RuntimeError(f"only {len(selected)} eligible diverted frames were found")
    return selected


def canonical_axes(row: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    """Return uniform axes spanning the rounded released endpoints."""

    stored_r = np.asarray(row["efit_grid_R"], dtype=float)
    stored_z = np.asarray(row["efit_grid_Z"], dtype=float)
    return (
        np.linspace(stored_r[0], stored_r[-1], stored_r.size),
        np.linspace(stored_z[0], stored_z[-1], stored_z.size),
    )


def pseudo_wall(
    radius: np.ndarray, height: np.ndarray, expansion: float, points_per_side: int = 33
) -> np.ndarray:
    """Return a clockwise rectangle enclosing the EFIT-grid extent."""

    if expansion < 0.0:
        raise ValueError("pseudo-wall expansion cannot be negative")
    radial_pad = expansion * float(np.ptp(radius))
    vertical_pad = expansion * float(np.ptp(height))
    low_r, high_r = radius[0] - radial_pad, radius[-1] + radial_pad
    low_z, high_z = height[0] - vertical_pad, height[-1] + vertical_pad
    if low_r <= 0.0:
        raise ValueError("pseudo-wall expansion reaches non-positive radius")
    lower_r = np.linspace(low_r, high_r, points_per_side, endpoint=False)
    right_z = np.linspace(low_z, high_z, points_per_side, endpoint=False)
    upper_r = np.linspace(high_r, low_r, points_per_side, endpoint=False)
    left_z = np.linspace(high_z, low_z, points_per_side, endpoint=False)
    lower = np.c_[lower_r, np.full_like(lower_r, low_z)]
    right = np.c_[np.full_like(right_z, high_r), right_z]
    upper = np.c_[upper_r, np.full_like(upper_r, high_z)]
    left = np.c_[np.full_like(left_z, low_r), left_z]
    return np.vstack([lower, right, upper, left])


def _plasma_mask(
    row: dict[str, Any], frame: int, radius: np.ndarray, height: np.ndarray
) -> np.ndarray:
    count = int(row["efit_lcfs_n"][frame])
    contour = np.c_[
        np.asarray(row["efit_lcfs_r"][frame][:count], dtype=float),
        np.asarray(row["efit_lcfs_z"][frame][:count], dtype=float),
    ]
    mesh_r, mesh_z = np.meshgrid(radius, height, indexing="ij")
    return (
        PolygonPath(contour)
        .contains_points(np.c_[mesh_r.ravel(), mesh_z.ravel()])
        .reshape(mesh_r.shape)
    )


def _label_state(
    row: dict[str, Any], frame: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Cross the corpus flux boundary once and extract native Nova profiles."""

    radius, height = canonical_axes(row)
    label = corpus_flux_to_nova_total(
        np.asarray(row["efit_psirz"][frame], dtype=float)
    ).T
    interpolant = RectBivariateSpline(radius, height, label, kx=3, ky=3, s=0)
    axis_flux = float(
        interpolant.ev(row["efit_r_axis"][frame], row["efit_z_axis"][frame])
    )
    count = int(row["efit_lcfs_n"][frame])
    boundary_flux = float(
        np.median(
            interpolant.ev(
                np.asarray(row["efit_lcfs_r"][frame][:count], dtype=float),
                np.asarray(row["efit_lcfs_z"][frame][:count], dtype=float),
            )
        )
    )
    normalised = (label - axis_flux) / (boundary_flux - axis_flux)
    extraction = extract_flux_functions(
        radius,
        height,
        label,
        normalised,
        plasma_mask=_plasma_mask(row, frame, radius, height),
        min_samples=6,
    )
    reliable = extraction.reliable & np.isfinite(
        extraction.p_prime + extraction.ff_prime
    )
    if np.count_nonzero(reliable) < 2:
        raise RuntimeError("fewer than two reliable extracted flux-function surfaces")
    return (
        label,
        extraction.psi_norm[reliable],
        extraction.p_prime[reliable],
        extraction.ff_prime[reliable],
    )


def _profile_function(nodes: np.ndarray, values: np.ndarray):
    grid = jnp.asarray(nodes)
    samples = jnp.asarray(values)

    def function(psi_norm):
        """Interpolate one extracted absolute flux function."""

        return jnp.interp(jnp.asarray(psi_norm), grid, samples)

    return function


def _green_block(
    target: np.ndarray, source: np.ndarray, width: float, height: float
) -> np.ndarray:
    """Return finite-cell total-flux response in Wb/A."""

    return np.stack(
        [
            hybrid_greens(
                target[:, 0], target[:, 1], source_r, source_z, width, height
            )[0]
            for source_r, source_z in source
        ],
        axis=1,
    )


def _source_current(
    row: dict[str, Any], description, names: tuple[str, ...], time_ms: float
) -> np.ndarray:
    source_time = np.asarray(row["magnetics_time"], dtype=float)
    by_name = {item.name: item for item in description.conductors}
    currents = []
    for name in names:
        conductor = by_name[name]
        values = np.asarray(row[conductor.input_column], dtype=float)
        valid = np.isfinite(source_time + values)
        current_ka = np.interp(time_ms, source_time[valid], values[valid])
        currents.append(1000.0 * current_ka * conductor.turns.applied_multiplier)
    return np.asarray(currents)


def _boundary_field_function(row: dict[str, Any], time_ms: float) -> float:
    time = np.asarray(row["magnetics_time"], dtype=float)
    current = np.asarray(row["magnetics_bcoil"], dtype=float)
    valid = np.isfinite(time + current)
    current_a = 1000.0 * np.interp(time_ms, time[valid], current[valid])
    corpus_field_function = mu_0 * TOROIDAL_FIELD_TURNS * current_a / (2.0 * np.pi)
    return float(corpus_f_to_nova(corpus_field_function))


def _wall_source_response(description, names, wall: np.ndarray) -> np.ndarray:
    by_name = {item.name: item for item in description.conductors}
    return np.stack(
        [
            polygon_greens(wall[:, 0], wall[:, 1], by_name[name].vertices)[0]
            for name in names
        ],
        axis=1,
    )


def _couplings(row, machine, radius, height, expansion):
    """Return geometry-only Green operators, cached across the cohort."""

    key = (machine.physical.physical_digest, float(expansion))
    cached = _COUPLING_CACHE.get(key)
    if cached is not None:
        return cached
    lattice = FluxLattice(radius, height)
    coordinate = lattice.coordinate
    wall = pseudo_wall(*canonical_axes(row), expansion)
    names, response = vacuum_response(machine.physical, radius, height)
    source_to_grid = np.stack([plane.T.ravel() for plane in response], axis=1)
    source_to_wall = _wall_source_response(machine.physical, names, wall)
    width = float(np.diff(radius).mean())
    vertical_extent = float(np.diff(height).mean())
    plasma_to_grid = _green_block(coordinate, coordinate, width, vertical_extent)
    plasma_to_wall = _green_block(wall, coordinate, width, vertical_extent)
    result = (
        names,
        wall,
        source_to_grid,
        source_to_wall,
        plasma_to_grid,
        plasma_to_wall,
    )
    _COUPLING_CACHE[key] = result
    return result


def build_profile(
    row: dict[str, Any],
    frame: int,
    expansion: float,
) -> tuple[ForwardProfile, np.ndarray, np.ndarray, np.ndarray, int, str]:
    """Build one prescribed-source production solve from a labelled frame."""

    label_full, surfaces, p_prime, ff_prime = _label_state(row, frame)
    radius_full, height_full = canonical_axes(row)
    radius = radius_full[::REGISTERED_GRID_STRIDE]
    height = height_full[::REGISTERED_GRID_STRIDE]
    label = label_full[::REGISTERED_GRID_STRIDE, ::REGISTERED_GRID_STRIDE]
    lattice = FluxLattice(radius, height)
    source_row = str(row.get("_source_path", "corpus row"))
    machine = dataset_machine_description(row, source_row=source_row)
    (
        names,
        wall,
        source_to_grid,
        source_to_wall,
        plasma_to_grid,
        plasma_to_wall,
    ) = _couplings(row, machine, radius, height, expansion)
    source = ForwardSource(
        core=DomainProfile(
            p_prime=_profile_function(surfaces, p_prime),
            ff_prime=_profile_function(surfaces, ff_prime),
        ),
        boundary_pressure=0.0,
        boundary_field_function=_boundary_field_function(
            row, float(row["efit_times"][frame])
        ),
    )
    profile = ForwardProfile.from_lattice(
        lattice,
        source,
        external_current=_source_current(
            row, machine.physical, names, float(row["efit_times"][frame])
        ),
        source_to_grid=source_to_grid,
        plasma_to_grid=plasma_to_grid,
        source_to_wall=source_to_wall,
        plasma_to_wall=plasma_to_wall,
        wall_coordinate=wall,
        polarity=1,
        inside_material=np.ones(lattice.node_count, dtype=bool),
    )
    spline = RectBivariateSpline(radius, height, label, kx=3, ky=3, s=0)
    seed = np.r_[label.ravel(), spline.ev(wall[:, 0], wall[:, 1])]
    statement = (
        "rectangular enclosing control surface derived from efit_grid extent; "
        "pseudo-wall standing in for the absent machine wall"
    )
    return profile, seed, label, wall, int(np.size(surfaces)), statement


def _separatrix(
    radius: np.ndarray,
    height: np.ndarray,
    flux: np.ndarray,
    axis: float,
    boundary: float,
) -> np.ndarray:
    normalised = (flux - axis) / (boundary - axis)
    lines = contour_generator(x=radius, y=height, z=normalised.T).lines(1.0)
    finite = [line[np.all(np.isfinite(line), axis=1)] for line in lines]
    finite = [line for line in finite if len(line) >= 4]
    if not finite:
        return np.empty((0, 2))
    return max(
        finite,
        key=lambda line: np.sum(np.linalg.norm(np.diff(line, axis=0), axis=1)),
    )


def gauge_metrics(
    labelled: np.ndarray, predicted: np.ndarray, interior: np.ndarray
) -> tuple[float, float, float, np.ndarray]:
    """Return gauge-aligned R-squared and fractional RMS on the label core."""

    selected = interior & np.isfinite(labelled + predicted)
    actual = labelled[selected]
    estimate = predicted[selected]
    gauge = float(np.mean(actual - estimate))
    residual = estimate + gauge - actual
    total = float(np.sum((actual - np.mean(actual)) ** 2))
    r_squared = 1.0 - float(np.sum(residual**2)) / total
    reference_rms = float(np.sqrt(np.mean((actual - np.mean(actual)) ** 2)))
    fractional_rms = float(np.sqrt(np.mean(residual**2)) / reference_rms)
    return r_squared, fractional_rms, gauge, predicted + gauge


def contour_separation(
    predicted: np.ndarray, labelled: np.ndarray
) -> tuple[float, float]:
    """Return symmetric nearest-contour radial separations in millimetres."""

    if len(predicted) < 2 or len(labelled) < 2:
        return float("nan"), float("nan")
    distances = np.r_[
        cKDTree(labelled).query(predicted)[0],
        cKDTree(predicted).query(labelled)[0],
    ]
    return 1000.0 * float(np.mean(distances)), 1000.0 * float(np.max(distances))


def _solve_registered(profile: ForwardProfile, seed: np.ndarray):
    """Run the preregistered relaxed host schedule and report its termination."""

    state = np.asarray(seed, dtype=float)
    relaxation = REGISTERED_HOST_INITIAL_RELAXATION
    evaluations = 0
    equilibrium = None
    while evaluations < REGISTERED_HOST_MAXIMUM_EVALUATIONS:
        budget = min(
            REGISTERED_HOST_RELAXATION_REDUCTION_INTERVAL,
            REGISTERED_HOST_MAXIMUM_EVALUATIONS - evaluations,
        )
        equilibrium = profile.solve(
            state,
            route=REGISTERED_SOLVER_ROUTE,
            evaluations=budget,
            relaxation=relaxation,
            tolerance=REGISTERED_HOST_TOLERANCE,
        )
        used = int(
            np.count_nonzero(np.isfinite(np.asarray(equilibrium.fixed_point.trace)))
        )
        evaluations += used
        residual = float(equilibrium.fixed_point.residual)
        if residual <= REGISTERED_HOST_TOLERANCE:
            return (
                equilibrium,
                f"converged after {evaluations} relaxed host evaluations; "
                f"final relaxation={relaxation:.9g}",
            )
        state = np.asarray(equilibrium.flux, dtype=float)
        if used < budget:
            break
        relaxation = max(REGISTERED_HOST_MINIMUM_RELAXATION, relaxation / 2.0)
    if equilibrium is None:
        raise RuntimeError("the registered host schedule performed no evaluations")
    return (
        equilibrium,
        f"exhausted {evaluations} relaxed host evaluations; "
        f"final relaxation={relaxation:.9g}",
    )


def solve_frame(
    row: dict[str, Any], frame: int, expansion: float
) -> tuple[FrameResult, dict[str, np.ndarray]]:
    """Solve and score one frame without hiding a failed criterion."""

    profile, seed, label, wall, reliable, wall_statement = build_profile(
        row, frame, expansion
    )
    equilibrium, solver_termination = _solve_registered(profile, seed)
    radius = profile.lattice.radius
    height = profile.lattice.height
    predicted = np.asarray(equilibrium.flux[: profile.lattice.node_count]).reshape(
        profile.lattice.shape
    )
    interior = _plasma_mask(row, frame, radius, height)
    r_squared, fractional_rms, gauge, aligned = gauge_metrics(
        label, predicted, interior
    )
    topology = equilibrium.topology
    predicted_contour = _separatrix(
        radius,
        height,
        predicted,
        float(topology.axis_flux),
        float(topology.boundary_flux),
    )
    count = int(row["efit_lcfs_n"][frame])
    labelled_contour = np.c_[
        np.asarray(row["efit_lcfs_r"][frame][:count], dtype=float),
        np.asarray(row["efit_lcfs_z"][frame][:count], dtype=float),
    ]
    separation_mean, separation_maximum = contour_separation(
        predicted_contour, labelled_contour
    )
    labelled_axis = np.asarray(
        [row["efit_r_axis"][frame], row["efit_z_axis"][frame]], dtype=float
    )
    axis_displacement = 1000.0 * float(
        np.linalg.norm(np.asarray(topology.axis, dtype=float) - labelled_axis)
    )
    labelled_q95 = float(corpus_q_to_nova(row["efit_q95"][frame]))
    try:
        geometry = FluxSurfaceGeometry.from_equilibrium(
            profile.lattice,
            profile.source,
            equilibrium,
            rho_tor_norm=np.linspace(0.0, 1.0, 33),
            edge_psi_norm=0.95,
        )
        predicted_q95 = float(geometry.safety_factor[-1])
    except SurfaceGeometryError, ValueError, FloatingPointError:
        predicted_q95 = float("nan")
    q95_error = (predicted_q95 - labelled_q95) / abs(labelled_q95)
    residual = float(equilibrium.fixed_point.residual)
    finite = bool(equilibrium.finite.passed)
    diverted = bool(topology.diverted)
    converged = bool(
        finite
        and diverted
        and np.isfinite(residual)
        and residual <= REGISTERED_RESIDUAL_TOLERANCE
    )
    result = FrameResult(
        shot=Path(row["_source_path"]).name,
        frame=frame,
        time_ms=float(row["efit_times"][frame]),
        geometry_digest=geometry_digest(row),
        reliable_flux_surfaces=reliable,
        pseudo_wall_expansion=expansion,
        pseudo_wall_statement=wall_statement,
        fixed_point_relative_residual=residual,
        residual_tolerance=REGISTERED_RESIDUAL_TOLERANCE,
        finite=finite,
        diverted=diverted,
        converged=converged,
        convergence_criterion=(
            "finite receipt AND diverted topology AND fixed-point relative residual "
            f"<= {REGISTERED_RESIDUAL_TOLERANCE:g}"
        ),
        solver_termination=solver_termination,
        metrics=MatchMetrics(
            interior_r_squared=r_squared,
            interior_fractional_rms=fractional_rms,
            additive_gauge_wb=gauge,
            separatrix_mean_radial_separation_mm=separation_mean,
            separatrix_maximum_radial_separation_mm=separation_maximum,
            magnetic_axis_displacement_mm=axis_displacement,
            predicted_q95_nova=predicted_q95,
            labelled_q95_nova=labelled_q95,
            signed_relative_q95_error=q95_error,
        ),
    )
    fields = {
        "radius": np.asarray(radius),
        "height": np.asarray(height),
        "labelled": label,
        "predicted": aligned,
        "difference": aligned - label,
        "labelled_contour": labelled_contour,
        "predicted_contour": predicted_contour,
        "pseudo_wall": wall,
    }
    return result, fields


def _distribution(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=float)
    return {
        "minimum": float(np.nanmin(array)),
        "median": float(np.nanmedian(array)),
        "maximum": float(np.nanmax(array)),
        "mean": float(np.nanmean(array)),
    }


def summarize(
    results: list[FrameResult],
    sensitivity: list[FrameResult],
    preregistration_hash: str,
) -> dict[str, Any]:
    """Return the cohort verdict while retaining every frame record."""

    r_squared = [item.metrics.interior_r_squared for item in results]
    baseline = next(
        item.metrics.interior_r_squared
        for item in sensitivity
        if item.pseudo_wall_expansion == REGISTERED_BASELINE_PSEUDO_WALL_EXPANSION
    )
    sensitivity_rows = [
        {
            "pseudo_wall_expansion": item.pseudo_wall_expansion,
            "interior_r_squared": item.metrics.interior_r_squared,
            "r_squared_change_from_baseline": (
                item.metrics.interior_r_squared - baseline
            ),
            "fixed_point_relative_residual": item.fixed_point_relative_residual,
            "converged": item.converged,
        }
        for item in sensitivity
    ]
    all_converged = all(item.converged for item in results)
    median_r_squared = float(np.median(r_squared))
    return {
        "preregistration_sha256": preregistration_hash,
        "frames": len(results),
        "coefficients_fitted": 0,
        "metric_assumptions": {
            "q95": ("depends linearly on F and the disclosed 144-turn external input"),
            "flux_separatrix_axis": (
                "independent of the toroidal-field turn-count assumption"
            ),
        },
        "pseudo_wall": {
            "statement": results[0].pseudo_wall_statement,
            "sensitivity": sensitivity_rows,
            "maximum_absolute_r_squared_move": float(
                max(
                    abs(item["r_squared_change_from_baseline"])
                    for item in sensitivity_rows
                )
            ),
        },
        "convergence": {
            "criterion": results[0].convergence_criterion,
            "converged_frames": sum(item.converged for item in results),
            "nonconverged_frames": [
                {"shot": item.shot, "frame": item.frame}
                for item in results
                if not item.converged
            ],
            "all_converged": all_converged,
        },
        "metrics": {
            "interior_r_squared": _distribution(r_squared),
            "interior_fractional_rms": _distribution(
                [item.metrics.interior_fractional_rms for item in results]
            ),
            "separatrix_mean_radial_separation_mm": _distribution(
                [item.metrics.separatrix_mean_radial_separation_mm for item in results]
            ),
            "separatrix_maximum_radial_separation_mm": _distribution(
                [
                    item.metrics.separatrix_maximum_radial_separation_mm
                    for item in results
                ]
            ),
            "magnetic_axis_displacement_mm": _distribution(
                [item.metrics.magnetic_axis_displacement_mm for item in results]
            ),
            "signed_relative_q95_error": _distribution(
                [item.metrics.signed_relative_q95_error for item in results]
            ),
        },
        "registered_median_interior_r_squared_bar": (REGISTERED_MEDIAN_INTERIOR_R2_BAR),
        "passed": bool(
            len(results) >= REGISTERED_FRAME_COUNT
            and all_converged
            and median_r_squared >= REGISTERED_MEDIAN_INTERIOR_R2_BAR
        ),
        "frame_records": [asdict(item) for item in results],
    }


def frame_figure(
    results: list[FrameResult], fields: list[dict[str, np.ndarray]], path: Path
) -> None:
    """Plot labelled, predicted, and difference maps for every scored frame."""

    figure, axes = plt.subplots(
        len(results), 3, figsize=(11, 2.5 * len(results)), constrained_layout=True
    )
    for row_axes, result, frame in zip(axes, results, fields, strict=True):
        radius, height = frame["radius"], frame["height"]
        columns = (
            ("labelled", "Labelled psi", "viridis"),
            ("predicted", "Forward psi", "viridis"),
            ("difference", "Forward - label", "coolwarm"),
        )
        for axis, (name, title, colour) in zip(row_axes, columns, strict=True):
            image = axis.pcolormesh(
                radius, height, frame[name].T, shading="auto", cmap=colour
            )
            axis.plot(
                frame["labelled_contour"][:, 0],
                frame["labelled_contour"][:, 1],
                color="black",
                linewidth=0.8,
                label="label LCFS",
            )
            if len(frame["predicted_contour"]):
                axis.plot(
                    frame["predicted_contour"][:, 0],
                    frame["predicted_contour"][:, 1],
                    color="tab:red",
                    linestyle="--",
                    linewidth=0.9,
                    label="forward separatrix",
                )
            axis.set_aspect("equal")
            axis.set_xlabel("R [m]")
            axis.set_ylabel("Z [m]")
            axis.set_title(title)
            figure.colorbar(image, ax=axis, label="total poloidal flux [Wb]")
        row_axes[0].text(
            0.01,
            0.99,
            f"{result.shot} frame {result.frame}\n"
            f"R2={result.metrics.interior_r_squared:.3f}; "
            f"converged={result.converged}",
            transform=row_axes[0].transAxes,
            va="top",
            fontsize=7,
            color="white",
        )
    axes[0, 0].legend(loc="lower left", fontsize=7)
    figure.savefig(path, dpi=180)
    plt.close(figure)


def cohort_figure(results: list[FrameResult], path: Path) -> None:
    """Plot the four declared match categories over the complete cohort."""

    index = np.arange(len(results))
    labels = [f"{Path(item.shot).stem[-6:]}:{item.frame}" for item in results]
    figure, axes = plt.subplots(2, 2, figsize=(11, 7), constrained_layout=True)
    axes[0, 0].plot(
        index, [item.metrics.interior_r_squared for item in results], "o-", label="R2"
    )
    axes[0, 0].axhline(
        REGISTERED_MEDIAN_INTERIOR_R2_BAR,
        color="black",
        linestyle="--",
        label="registered median bar",
    )
    twin = axes[0, 0].twinx()
    twin.plot(
        index,
        [item.metrics.interior_fractional_rms for item in results],
        "s:",
        color="tab:orange",
        label="fractional RMS",
    )
    axes[0, 0].set_title("Interior flux match")
    axes[0, 0].legend(fontsize=8)
    twin.set_ylabel("fractional RMS")
    axes[0, 1].plot(
        index,
        [item.metrics.separatrix_mean_radial_separation_mm for item in results],
        "o-",
        label="mean",
    )
    axes[0, 1].plot(
        index,
        [item.metrics.separatrix_maximum_radial_separation_mm for item in results],
        "s-",
        label="maximum",
    )
    axes[0, 1].set_title("Separatrix separation")
    axes[0, 1].set_ylabel("mm")
    axes[0, 1].legend(fontsize=8)
    axes[1, 0].plot(
        index,
        [item.metrics.magnetic_axis_displacement_mm for item in results],
        "o-",
    )
    axes[1, 0].set_title("Magnetic-axis displacement")
    axes[1, 0].set_ylabel("mm")
    axes[1, 1].axhline(0.0, color="black", linewidth=0.7)
    axes[1, 1].plot(
        index,
        [item.metrics.signed_relative_q95_error for item in results],
        "o-",
    )
    axes[1, 1].set_title("Signed relative q95 error")
    for axis in axes.ravel():
        axis.set_xticks(index, labels, rotation=45, ha="right", fontsize=7)
        axis.grid(alpha=0.25)
    figure.savefig(path, dpi=180)
    plt.close(figure)


def run(
    data: Path, output: Path, frames: int = REGISTERED_FRAME_COUNT
) -> dict[str, Any]:
    """Write the bar, execute the declared cohort, and publish its evidence."""

    if frames != REGISTERED_FRAME_COUNT:
        raise ValueError(
            f"scoring requires exactly {REGISTERED_FRAME_COUNT} preregistered frames"
        )
    configure_dtypes()
    preregistration_path = write_preregistration(output)
    preregistration_hash = require_preregistration(preregistration_path)
    paths = sorted(data.glob("*.parquet"))
    selected = select_frames(paths, frames)
    results: list[FrameResult] = []
    fields: list[dict[str, np.ndarray]] = []
    baseline_expansion = REGISTERED_BASELINE_PSEUDO_WALL_EXPANSION
    for number, selected_frame in enumerate(selected, start=1):
        row = _read(
            selected_frame.path,
            _LABEL_COLUMNS + _GEOMETRY_COLUMNS + _CURRENT_COLUMNS,
        )
        row["_source_path"] = str(selected_frame.path)
        result, frame_fields = solve_frame(
            row, selected_frame.frame, baseline_expansion
        )
        if not result.converged:
            raise RuntimeError(
                f"non-converged frame {result.shot}:{result.frame}: "
                f"residual={result.fixed_point_relative_residual:.9e}, "
                f"diverted={result.diverted}, finite={result.finite}, "
                f"termination={result.solver_termination}"
            )
        results.append(result)
        fields.append(frame_fields)
        print(
            f"SOLVED {number}/{len(selected)} {result.shot}:{result.frame} "
            f"residual={result.fixed_point_relative_residual:.6e} "
            f"converged={result.converged}",
            flush=True,
        )
    first = selected[0]
    first_row = _read(first.path, _LABEL_COLUMNS + _GEOMETRY_COLUMNS + _CURRENT_COLUMNS)
    first_row["_source_path"] = str(first.path)
    sensitivity = []
    for expansion in PSEUDO_WALL_EXPANSIONS:
        if expansion == baseline_expansion:
            sensitivity.append(results[0])
        else:
            expanded = solve_frame(first_row, first.frame, expansion)[0]
            if not expanded.converged:
                raise RuntimeError(
                    f"non-converged pseudo-wall sensitivity for "
                    f"{expanded.shot}:{expanded.frame}: "
                    f"residual={expanded.fixed_point_relative_residual:.9e}, "
                    f"diverted={expanded.diverted}, finite={expanded.finite}, "
                    f"termination={expanded.solver_termination}"
                )
            sensitivity.append(expanded)
    receipt = {
        "preregistration": preregistration(),
        "preregistration_path": str(preregistration_path),
        "result": summarize(results, sensitivity, preregistration_hash),
    }
    receipt_path = output / RECEIPT_NAME
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    frame_figure(results, fields, output / FRAME_FIGURE_NAME)
    cohort_figure(results, output / COHORT_FIGURE_NAME)
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--frames", type=int, default=REGISTERED_FRAME_COUNT)
    parser.add_argument("--preregister-only", action="store_true")
    arguments = parser.parse_args()
    path = write_preregistration(arguments.output)
    print(f"PREREGISTERED {path}", flush=True)
    if arguments.preregister_only:
        return
    receipt = run(arguments.data, arguments.output, arguments.frames)
    headline = dict(receipt["result"])
    headline.pop("frame_records", None)
    print(json.dumps(headline, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
