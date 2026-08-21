# ruff: noqa: E501
"""Bank one reference-seeded MAST free-boundary forward solve.

The slice is selected only from the committed native-grid decomposition bank.
Its declared pressure and diamagnetic gradients drive ``ForwardProfile.solve``
on the 65-point normalized-flux base after conversion to Nova's total-flux
COCOS 17 convention. The fitted conductor state is primary and the archived
experimental state is a side arm.

The current image is evaluated in the reference's declared flux coordinate.
Both its normalized-flux argument and its support use the declared axis and
boundary constants in the unchanged total-flux gauge. Nova's extracted
topology remains an output used for geometry scoring; it does not replace the
declared coordinate that the stored profiles belong to.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import jax.numpy as jnp
import matplotlib
import numpy as np
import zarr
from contourpy import contour_generator
from matplotlib import pyplot as plt
from scipy.interpolate import RectBivariateSpline
from scipy.spatial import cKDTree

from benchmarks.efit_native_grid_decomposition import _circuit_drives
from benchmarks.efit_topology_boundary_score import (
    _live_flux_map,
    _stored_lcfs,
    _stored_x_points,
)
from nova.biot.greens import hybrid_greens
from nova.biot.null import Null1D, Null2D
from nova.biot.target import FluxTarget
from nova.catalog.mast_geometry import MachineGeometryRegistry
from nova.equilibrium.conservation import FluxLattice
from nova.equilibrium.convention import TOTAL_FLUX_FACTOR
from nova.equilibrium.forward import ForwardProfile
from nova.equilibrium.forward_operator import ForwardFluxOperator
from nova.equilibrium.source import DomainProfile, ForwardSource
from nova.equilibrium.stencil_mesh import CellCurrentMoments
from nova.equilibrium.wall_mask import inside_polygon
from nova.imas.mast_solve_inputs import SHOT_STORE
from nova.imas.mast_vacuum_response import loop_response_matrix
from nova.imas.parity_tolerances import ScorecardField, registered_tolerances
from nova.jax.config import configure_dtypes
from nova.geometry.hexstencil import hex_stencil

matplotlib.use("Agg")

DECOMPOSITION_BANK = Path(
    "docs/figures/efit-flux-decomposition/native-grid-decomposition.json"
)
DEFAULT_OUTPUT = Path("docs/figures/efit-forward-parity")
RECEIPT_NAME = "reference-seeded-forward-slice.json"
FIGURE_NAME = "reference-seeded-forward-slice.png"
GRID_STRIDE = 2
FIXED_POINT_CRITERION = 1.0e-8
NEWTON_STEPS = 12
GMRES_ITERATIONS = 12
WARMUP_SWEEPS = 0
RELAXATION = 0.5
STEP_CAP = 10.0
SADDLE_OFFSET_LIMIT = 0.01
LCFS_CONTOUR_LIMIT = 0.006


@dataclass
class DeclaredAnchorOperator(ForwardFluxOperator):
    """Evaluate the source on the reference's own declared coordinate."""

    declared_axis_flux: float = 0.0
    declared_boundary_flux: float = 1.0
    declared_support: np.ndarray | None = None

    def __post_init__(self):
        """Validate the fixed source-coordinate declaration."""
        super().__post_init__()
        if self.declared_axis_flux == self.declared_boundary_flux:
            raise ValueError("declared reference anchors have zero span")
        if self.declared_support is None:
            raise ValueError("declared reference support is required")
        self.declared_support = jnp.asarray(self.declared_support, dtype=bool)
        if self.declared_support.shape != (self.grid.node_number,):
            raise ValueError("declared support must carry one flag per grid node")
        self.use_linear_moments = False

    def cell_current_moments(self, psi) -> CellCurrentMoments:
        """Return source current bounded in the declared reference coordinate."""
        grid_flux = jnp.asarray(psi)[: self.grid.node_number]
        psi_norm = (grid_flux - self.declared_axis_flux) / (
            self.declared_boundary_flux - self.declared_axis_flux
        )
        bounded = self.declared_support & (psi_norm >= 0.0) & (psi_norm <= 1.0)
        density = self.source.core.current_density(
            self.radius, jnp.where(bounded, psi_norm, 1.0)
        )
        current = jnp.where(bounded, density * self.area, 0.0)
        zero = jnp.zeros_like(current)
        return CellCurrentMoments(current, zero, zero)


def _absolute(value: float) -> float:
    """Return a JSON-stable absolute scalar."""
    return float(abs(value))


def _qualification(row: dict[str, Any]) -> dict[str, Any] | None:
    """Return measurable pre-solve qualification for one banked slice."""
    reference = row["reference_consistency_checked_before_localisation"]
    saddle = reference["stored_boundary_anchor_against_map_saddle"]
    contour = reference["stored_lcfs_against_map_contour"]
    typed = saddle["selected"] is not None
    offset = (
        None
        if saddle["offset_fraction_of_declared_span"] is None
        else _absolute(saddle["offset_fraction_of_declared_span"])
    )
    lcfs_sup = float(contour["declared_coordinate_sup_error_from_one"])
    if not typed or offset is None:
        return None
    return {
        "typed_saddle_resolved": True,
        "typed_saddle_candidate_count": int(saddle["candidate_count"]),
        "declared_boundary_to_map_saddle_offset_fraction_of_declared_span": offset,
        "declared_boundary_to_map_saddle_limit": SADDLE_OFFSET_LIMIT,
        "stored_lcfs_contour_sup_discrepancy_fraction_of_declared_span": lcfs_sup,
        "stored_lcfs_contour_limit": LCFS_CONTOUR_LIMIT,
        "passes": bool(offset < SADDLE_OFFSET_LIMIT and lcfs_sup < LCFS_CONTOUR_LIMIT),
    }


def select_slice(bank: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    """Select the lowest worst-fraction qualified banked slice."""
    report = json.loads(bank.read_text())
    candidates = []
    for row in report["slices"]:
        qualification = _qualification(row)
        if qualification is None or not qualification["passes"]:
            continue
        score = max(
            qualification[
                "declared_boundary_to_map_saddle_offset_fraction_of_declared_span"
            ],
            qualification[
                "stored_lcfs_contour_sup_discrepancy_fraction_of_declared_span"
            ],
        )
        candidates.append(
            (
                score,
                int(row["shot"]),
                int(row["slice_index"]),
                row,
                qualification,
            )
        )
    if not candidates:
        raise RuntimeError("decomposition bank contains no qualified forward slice")
    _, _, _, row, qualification = min(candidates, key=lambda item: item[:3])
    return row, qualification


def _profile_function(nodes: np.ndarray, values: np.ndarray):
    """Return a traced interpolation of one stored 65-point profile."""
    grid = jnp.asarray(nodes, dtype=jnp.float64)
    samples = jnp.asarray(values, dtype=jnp.float64)

    def function(psi_norm):
        """Evaluate the stored profile on its declared normalized coordinate."""
        return jnp.interp(jnp.asarray(psi_norm), grid, samples)

    return function


def _plasma_response(target: np.ndarray, source: np.ndarray, dr: float, dz: float):
    """Return rectangular-cell total-flux response in Wb/A."""
    return np.stack(
        [
            hybrid_greens(target[:, 0], target[:, 1], source_r, source_z, dr, dz)[0]
            for source_r, source_z in source
        ],
        axis=1,
    )


def _inside_limiter(radius: np.ndarray, height: np.ndarray, limiter: np.ndarray):
    """Return material-domain flags on a radius-major lattice."""
    rr, zz = np.meshgrid(radius, height, indexing="ij")
    return np.asarray(
        inside_polygon(rr.ravel(), zz.ravel(), limiter[:, 0], limiter[:, 1]),
        dtype=bool,
    )


def _source_response(
    geometry: dict[str, Any], coordinates: np.ndarray, families: tuple[str, ...]
) -> np.ndarray:
    """Return active-winding total-flux response in Wb/(A turn)."""
    return loop_response_matrix(geometry, coordinates, families=families)


def _stored_map(
    group: zarr.Group, row: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return canonical axes and the live total-flux reference map."""
    stored_r = np.asarray(group["gridr"], dtype=np.float64)
    stored_z = np.asarray(group["gridz"], dtype=np.float64)
    radius = np.linspace(stored_r[0], stored_r[-1], stored_r.size)
    height = np.linspace(stored_z[0], stored_z[-1], stored_z.size)
    if not np.allclose(stored_r, radius, rtol=2.0e-7, atol=1.0e-8):
        raise ValueError("efm/gridr is not a uniform 65-point axis")
    if not np.allclose(stored_z, height, rtol=2.0e-7, atol=1.0e-8):
        raise ValueError("efm/gridz is not a uniform 65-point axis")
    return radius, height, TOTAL_FLUX_FACTOR * _live_flux_map(group, row, len(radius)).T


def build_profile(
    group: zarr.Group,
    shot: int,
    row: int,
    current_field: str,
) -> tuple[ForwardProfile, np.ndarray, np.ndarray, dict[str, Any]]:
    """Build one prescribed-anchor forward profile and reference seed."""
    full_r, full_z, reference_full = _stored_map(group, row)
    radius = full_r[::GRID_STRIDE]
    height = full_z[::GRID_STRIDE]
    reference = reference_full[::GRID_STRIDE, ::GRID_STRIDE]
    lattice = FluxLattice(radius, height)
    limiter = np.column_stack(
        [
            np.asarray(group["limiterr"], dtype=float),
            np.asarray(group["limiterz"], dtype=float),
        ]
    )
    material = _inside_limiter(radius, height, limiter)
    lcfs = _stored_lcfs(group, row)
    declared_support = (
        np.asarray(
            inside_polygon(
                lattice.coordinate[:, 0],
                lattice.coordinate[:, 1],
                lcfs[:, 0],
                lcfs[:, 1],
            ),
            dtype=bool,
        )
        & material
    )

    selection = MachineGeometryRegistry.default().select(shot)
    geometry = selection.configuration.geometry
    families, drive, mapping = _circuit_drives(group, row, geometry, current_field)
    source_to_grid = _source_response(geometry, lattice.coordinate, families)
    source_to_wall = _source_response(geometry, limiter, families)
    dr = float(np.diff(radius).mean())
    dz = float(np.diff(height).mean())
    plasma_to_grid = _plasma_response(lattice.coordinate, lattice.coordinate, dr, dz)
    plasma_to_wall = _plasma_response(limiter, lattice.coordinate, dr, dz)

    psi_norm = np.asarray(group["psi_norm"], dtype=np.float64)
    if psi_norm.shape != (65,) or not np.allclose(psi_norm, np.linspace(0.0, 1.0, 65)):
        raise ValueError("efm/psi_norm is not the declared uniform 65-point base")
    p_prime = np.asarray(group["pprime"][row], dtype=np.float64) / TOTAL_FLUX_FACTOR
    ff_prime = np.asarray(group["ffprime"][row], dtype=np.float64) / TOTAL_FLUX_FACTOR
    source = ForwardSource(
        core=DomainProfile(
            p_prime=_profile_function(psi_norm, p_prime),
            ff_prime=_profile_function(psi_norm, ff_prime),
        ),
        boundary_pressure=float(group["ppsi_c"][row, -1]),
        boundary_field_function=float(group["fpsi_c"][row, -1]),
    )
    axis_flux = TOTAL_FLUX_FACTOR * float(group["psi_axis"][row])
    boundary_flux = TOTAL_FLUX_FACTOR * float(group["psi_boundary"][row])
    operator = DeclaredAnchorOperator(
        grid=FluxTarget(
            source_target=jnp.asarray(source_to_grid),
            plasma_target=jnp.asarray(plasma_to_grid),
            null=Null2D.from_coordinates(
                lattice.coordinate,
                hex_stencil(lattice.shape),
                maxsize=5,
            ),
        ),
        wall=FluxTarget(
            source_target=jnp.asarray(source_to_wall),
            plasma_target=jnp.asarray(plasma_to_wall),
            null=Null1D(jnp.asarray(limiter, dtype=jnp.float64)),
        ),
        source=source,
        external_current=jnp.asarray(drive),
        area=jnp.asarray(lattice.cell_area),
        polarity=1,
        inside_material=jnp.asarray(material),
        use_linear_moments=False,
        declared_axis_flux=axis_flux,
        declared_boundary_flux=boundary_flux,
        declared_support=declared_support,
    )
    profile = ForwardProfile(
        operator=operator,
        lattice=lattice,
        newton_steps=NEWTON_STEPS,
    )
    spline = RectBivariateSpline(full_r, full_z, reference_full, kx=3, ky=3, s=0.0)
    seed = np.r_[reference.ravel(), spline.ev(limiter[:, 0], limiter[:, 1])]
    provenance = {
        "current_field": f"efm/{current_field}",
        "current_mapping": mapping,
        "target_cocos": 17,
        "profile_base": "efm/psi_norm, 65 uniform points from 0 to 1",
        "pprime_source": "efm/pprime divided by 2*pi into Pa/Wb",
        "ffprime_source": "efm/ffprime divided by 2*pi into T*m/Wb convention",
        "seed_source": "efm/psirz multiplied by 2*pi into total Wb",
        "declared_axis_flux_wb": axis_flux,
        "declared_boundary_flux_wb": boundary_flux,
        "declared_support_nodes": int(np.count_nonzero(declared_support)),
        "gauge": "unchanged reference total-flux gauge; no re-zeroing or mixed-gauge constants",
    }
    return profile, seed, reference, provenance


def _contour(radius, height, flux, axis_flux, boundary_flux):
    """Return the longest solved unit-normalized contour."""
    normalised = (flux - axis_flux) / (boundary_flux - axis_flux)
    lines = contour_generator(x=radius, y=height, z=normalised.T).lines(1.0)
    finite = [line[np.all(np.isfinite(line), axis=1)] for line in lines]
    finite = [line for line in finite if len(line) >= 4]
    if not finite:
        return np.empty((0, 2))
    selected = max(
        finite, key=lambda line: np.linalg.norm(np.diff(line, axis=0), axis=1).sum()
    )
    return selected


def _symmetric_mean_distance(left: np.ndarray, right: np.ndarray) -> float:
    """Return symmetric mean nearest-neighbour contour distance."""
    if len(left) < 2 or len(right) < 2:
        return float("nan")
    return float(
        0.5
        * (cKDTree(right).query(left)[0].mean() + cKDTree(left).query(right)[0].mean())
    )


def _iteration_count(trace: np.ndarray, criterion: float) -> int:
    """Return the first measured evaluation meeting the criterion, or the budget."""
    finite = np.flatnonzero(np.isfinite(trace))
    passing = finite[trace[finite] <= criterion]
    return int(passing[0] + 1 if passing.size else finite[-1] + 1)


def _newton_iteration_count(map_evaluations: int) -> int:
    """Return completed Newton promotions at one mapped-evaluation count."""
    stride = GMRES_ITERATIONS + 2
    return int(np.ceil(max(map_evaluations - WARMUP_SWEEPS, 0) / stride))


def solve_arm(
    group: zarr.Group, shot: int, row: int, current_field: str
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Solve and score one conductor-current arm."""
    profile, seed, reference, provenance = build_profile(
        group, shot, row, current_field
    )
    equilibrium = profile.solve(
        seed,
        route="newton_krylov",
        gmres_iterations=GMRES_ITERATIONS,
        warmup=WARMUP_SWEEPS,
        relaxation=RELAXATION,
        step_cap=STEP_CAP,
    )
    mapped = np.asarray(profile.flux_map()(equilibrium.flux), dtype=np.float64)
    solved = np.asarray(
        equilibrium.flux[: profile.lattice.node_count], dtype=np.float64
    ).reshape(profile.lattice.shape)
    span = float(np.ptp(reference))
    difference = solved - reference
    trace = np.asarray(equilibrium.fixed_point.trace, dtype=np.float64)
    grid_count = profile.lattice.node_count
    grid_scale = max(float(np.max(np.abs(mapped[:grid_count]))), 1.0e-30)
    wall_scale = max(float(np.max(np.abs(mapped[grid_count:]))), 1.0e-30)
    separated_defect = {
        "grid_relative_sup": float(
            np.max(
                np.abs(mapped[:grid_count] - np.asarray(equilibrium.flux)[:grid_count])
            )
            / grid_scale
        ),
        "wall_relative_sup": float(
            np.max(
                np.abs(mapped[grid_count:] - np.asarray(equilibrium.flux)[grid_count:])
            )
            / wall_scale
        ),
    }
    topology = equilibrium.topology
    solved_contour = _contour(
        profile.lattice.radius,
        profile.lattice.height,
        solved,
        float(topology.axis_flux),
        float(topology.boundary_flux),
    )
    stored_contour = _stored_lcfs(group, row)
    stored_x = _stored_x_points(group, row)
    solved_x = (
        np.asarray(topology.x_point, dtype=float).reshape(1, 2)
        if bool(topology.diverted)
        else np.empty((0, 2))
    )
    x_distance = (
        float(
            np.min(np.linalg.norm(solved_x[:, None, :] - stored_x[None, :, :], axis=2))
        )
        if len(solved_x) and len(stored_x)
        else None
    )
    stored_diverted = bool(len(stored_x))
    solved_diverted = bool(topology.diverted)
    tolerances = registered_tolerances()
    axis_distance = float(
        np.linalg.norm(
            np.asarray(topology.axis, dtype=float)
            - np.asarray(
                [group["magnetic_axis_r"][row], group["magnetic_axis_z"][row]],
                dtype=float,
            )
        )
    )
    lcfs_distance = _symmetric_mean_distance(solved_contour, stored_contour)
    plasma_current = float(np.sum(np.asarray(equilibrium.cell_current)))
    reference_current = float(group["plasma_current_c"][row])
    current_fraction = abs(plasma_current / reference_current)
    vacuum_branch = bool(current_fraction < 0.01)
    moments = equilibrium.moments
    beta_reference = float(group["betap"][row])
    li_reference = float(group["li"][row])
    metrics = {
        "flux_map": {
            "comparison_target": "efm/psirz multiplied by 2*pi into total Wb",
            "sup_fraction_of_reference_span": float(np.max(np.abs(difference)) / span),
            "rms_fraction_of_reference_span": float(
                np.sqrt(np.mean(difference**2)) / span
            ),
            "reference_span_wb": span,
            "raw_same_gauge_comparison": True,
        },
        "magnetic_axis": {
            "distance_m": axis_distance,
            "registered_bound_m": float(
                tolerances[ScorecardField.MAGNETIC_AXIS_DISTANCE_M].bound
            ),
            "passes": bool(
                tolerances[ScorecardField.MAGNETIC_AXIS_DISTANCE_M].passes(
                    axis_distance
                )
            ),
        },
        "lcfs": {
            "symmetric_mean_distance_m": lcfs_distance,
            "registered_bound_m": float(
                tolerances[ScorecardField.LCFS_DISTANCE_M].bound
            ),
            "passes": bool(
                tolerances[ScorecardField.LCFS_DISTANCE_M].passes(lcfs_distance)
            ),
        },
        "x_point": {
            "distance_m": x_distance,
            "registered_bound_m": float(
                tolerances[ScorecardField.X_POINT_DISTANCE_M].bound
            ),
            "passes": bool(
                x_distance is not None
                and tolerances[ScorecardField.X_POINT_DISTANCE_M].passes(x_distance)
            ),
        },
        "topology": {
            "solved_class": "diverted" if solved_diverted else "limited",
            "reference_class": "diverted" if stored_diverted else "limited",
            "agreement": float(solved_diverted == stored_diverted),
            "registered_bound": float(
                tolerances[ScorecardField.TOPOLOGY_CLASS_AGREEMENT_FRACTION].bound
            ),
            "passes": bool(solved_diverted == stored_diverted),
        },
        "plasma_current": {
            "solved_a": plasma_current,
            "reference_a": reference_current,
            "signed_relative_deviation": plasma_current / reference_current - 1.0,
            "absolute_fraction_of_reference": current_fraction,
            "dina_calibration_signed_relative_deviation": -0.0112,
        },
        "poloidal_beta": {
            "solved": float(moments.poloidal_beta),
            "reference": beta_reference,
            "signed_relative_deviation": float(moments.poloidal_beta) / beta_reference
            - 1.0,
        },
        "internal_inductance": {
            "solved": float(moments.internal_inductance),
            "reference": li_reference,
            "signed_relative_deviation": float(moments.internal_inductance)
            / li_reference
            - 1.0,
        },
        "dina_axis_calibration_m": 0.0412,
    }
    map_evaluations = _iteration_count(trace, FIXED_POINT_CRITERION)
    fixed_point_pass = bool(max(separated_defect.values()) <= FIXED_POINT_CRITERION)
    record = {
        "current_arm": provenance,
        "solver": {
            "entry_point": "ForwardProfile.solve",
            "route": "newton_krylov",
            "reference_seeded": True,
            "newton_iteration_count": _newton_iteration_count(map_evaluations),
            "map_evaluations_to_criterion": map_evaluations,
            "fixed_point_defect": float(equilibrium.fixed_point.residual),
            "separated_field_defect": separated_defect,
            "registered_criterion": FIXED_POINT_CRITERION,
            "passes_registered_criterion": fixed_point_pass,
            "residual_trace": [
                float(value) if np.isfinite(value) else None for value in trace
            ],
            "finite_receipt": bool(equilibrium.finite.passed),
        },
        "branch": {
            "classification": "vacuum" if vacuum_branch else "plasma",
            "vacuum_branch": vacuum_branch,
            "criterion": (
                "absolute solved current below 1 percent of efm/plasma_current_c"
            ),
            "fixed_point_convergence_is_not_parity": bool(
                vacuum_branch and fixed_point_pass
            ),
        },
        "metrics": metrics,
        "parity_verdict": (
            "FAIL_VACUUM_BRANCH"
            if vacuum_branch
            else "PASS_FIXED_POINT_ONLY"
            if fixed_point_pass
            else "FAIL_FIXED_POINT"
        ),
    }
    fields = {
        "radius": np.asarray(profile.lattice.radius),
        "height": np.asarray(profile.lattice.height),
        "reference": reference,
        "solved": solved,
        "difference": difference,
        "stored_contour": stored_contour,
        "solved_contour": solved_contour,
    }
    return record, fields


def _figure(fields: dict[str, np.ndarray], path: Path) -> None:
    """Plot the primary reference, solution and span-normalized difference."""
    radius = fields["radius"]
    height = fields["height"]
    span = float(np.ptp(fields["reference"]))
    figure, axes = plt.subplots(1, 3, figsize=(10.5, 3.3), constrained_layout=True)
    panels = (
        (fields["reference"], "EFM reference", "viridis"),
        (fields["solved"], "Forward solution", "viridis"),
        (fields["difference"] / span, "Difference / span", "coolwarm"),
    )
    for axis, (values, title, cmap) in zip(axes, panels, strict=True):
        image = axis.pcolormesh(radius, height, values.T, shading="auto", cmap=cmap)
        axis.plot(
            fields["stored_contour"][:, 0],
            fields["stored_contour"][:, 1],
            color="white",
            lw=0.8,
        )
        if len(fields["solved_contour"]):
            axis.plot(
                fields["solved_contour"][:, 0],
                fields["solved_contour"][:, 1],
                color="black",
                lw=0.8,
                ls="--",
            )
        axis.set_title(title)
        axis.set_xlabel("R [m]")
        axis.set_aspect("equal")
        figure.colorbar(image, ax=axis, shrink=0.78)
    axes[0].set_ylabel("Z [m]")
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180)
    plt.close(figure)


def run(store: Path, bank: Path, output: Path) -> dict[str, Any]:
    """Run both current arms and bank the primary field-to-field comparison."""
    configure_dtypes()
    selected, qualification = select_slice(bank)
    shot = int(selected["shot"])
    row = int(selected["slice_index"])
    group = zarr.open_group(str(store / f"{shot}.zarr"), mode="r")["efm"]
    primary, fields = solve_arm(group, shot, row, "fcoil_c")
    comparison, _ = solve_arm(group, shot, row, "fcoil_x")
    figure_path = output / FIGURE_NAME
    _figure(fields, figure_path)
    receipt = {
        "receipt": "reference-seeded MAST forward parity slice",
        "backend": "JAX_PLATFORMS=cpu required by invocation",
        "selection": {
            "source": str(bank),
            "rule": "minimum worst qualification fraction; tie-break by shot then slice index",
            "shot": shot,
            "slice_index": row,
            "time_s": float(group["time"][row]),
            "qualification_before_attribution": qualification,
        },
        "path_audit": {
            "sensor_reads": 0,
            "whitening_matrices": 0,
            "least_squares_updates": 0,
            "inverse_fit_coefficients": 0,
            "passes": True,
        },
        "primary": primary,
        "comparison_arm": comparison,
        "dina_calibration": {
            "axis_distance_m": 0.0412,
            "plasma_current_signed_relative_deviation": -0.0112,
            "poloidal_beta_signed_relative_deviation": 0.0224,
            "internal_inductance_signed_relative_deviation": -0.00135,
            "role": "calibration beside this MAST result, not an acceptance bound",
        },
        "figure_src": "/nova/figures/efit-forward-parity/reference-seeded-forward-slice.png",
    }
    output.mkdir(parents=True, exist_ok=True)
    receipt_path = output / RECEIPT_NAME
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n")
    return receipt


def main() -> None:
    """Parse paths, run the banked slice and print headline metrics."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--store", type=Path, default=SHOT_STORE)
    parser.add_argument("--bank", type=Path, default=DECOMPOSITION_BANK)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    arguments = parser.parse_args()
    receipt = run(arguments.store, arguments.bank, arguments.output)
    selection = receipt["selection"]
    primary = receipt["primary"]
    flux = primary["metrics"]["flux_map"]
    solver = primary["solver"]
    print(
        "FORWARD_SLICE "
        f"shot={selection['shot']} row={selection['slice_index']} "
        f"qualification={selection['qualification_before_attribution']['passes']} "
        f"newton_iterations={solver['newton_iteration_count']} "
        f"map_evaluations={solver['map_evaluations_to_criterion']} "
        f"defect={solver['fixed_point_defect']:.9g} "
        f"sup_span={flux['sup_fraction_of_reference_span']:.9g} "
        f"rms_span={flux['rms_fraction_of_reference_span']:.9g}"
    )


if __name__ == "__main__":
    main()
