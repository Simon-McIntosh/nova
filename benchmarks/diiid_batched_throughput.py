"""Measure fixed-shape batched forward-solve throughput on an accelerator.

The workload is Nova's production ``ForwardProfile.solve_batch`` route on a
bootstrapped free-boundary equilibrium.  Each timed slice executes a fixed
Newton--Krylov policy, takes the preceding slice as its warm start, verifies
that every ensemble member remains on the same plasma branch, and emits the
vacuum decay-index conditioning at its magnetic-axis radius.

The benchmark measures compute throughput.  It does not claim that the
bootstrapped source is a fitted DIII-D discharge or that a fixed evaluation
budget establishes convergence.  The output records the residual and branch
qualification beside the timing so downstream sizing cannot erase that
distinction.
"""

from __future__ import annotations

import argparse
import csv
import json
import platform
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from scipy.constants import mu_0

from nova.biot.greens import hybrid_greens
from nova.equilibrium.conservation import FluxLattice
from nova.equilibrium.diagnostics import vertical_conditioning_receipt
from nova.equilibrium.forward import ForwardProfile
from nova.equilibrium.source import DomainProfile, ForwardSource
from nova.jax.config import configure_dtypes

DEFAULT_OUTPUT = Path("docs/figures/diiid-forward-onboarding/throughput")
BUDGET_JSON = "batched_throughput_budget.json"
BUDGET_CSV = "batched_throughput_budget.csv"
BUDGET_HTML = "batched_throughput_budget.html"

P_PRIME = -3.0e5
FF_PRIME = -0.25
AXIS_RADIUS = 1.0
SEED_SPAN = 0.35
DRIVE = 1.4
BOUNDARY_FIELD_FUNCTION = 5.0
CONDUCTOR_COUNT = 16
GRID_POINTS = 25


@dataclass(frozen=True)
class BranchReceipt:
    """Identity and location of one solved plasma branch."""

    identity: str
    diverted: bool
    plasma_current_a: float
    axis_r_m: float
    axis_z_m: float


def validate_warm_start_continuity(
    shot: str,
    previous: list[BranchReceipt],
    current: list[BranchReceipt],
    *,
    maximum_axis_step_m: float,
) -> None:
    """Refuse a warm-start batch that changes branch along one shot."""

    if len(previous) != len(current):
        raise RuntimeError(f"{shot}: warm-start ensemble size changed")
    for member, (before, after) in enumerate(zip(previous, current, strict=True)):
        displacement = float(
            np.hypot(after.axis_r_m - before.axis_r_m, after.axis_z_m - before.axis_z_m)
        )
        if not np.isfinite(displacement):
            raise RuntimeError(
                f"{shot}: ensemble member {member} has a non-finite branch axis"
            )
        if before.identity != after.identity:
            raise RuntimeError(
                f"{shot}: ensemble member {member} changed branch "
                f"from {before.identity} to {after.identity}"
            )
        if displacement > maximum_axis_step_m:
            raise RuntimeError(
                f"{shot}: ensemble member {member} axis moved {displacement:.6g} m "
                f"across a {maximum_axis_step_m:.6g} m continuity bound"
            )


def _terms() -> tuple[float, float, float]:
    alpha = np.pi**2 * mu_0 * P_PRIME / 2.0
    return alpha, -2.0 * alpha * AXIS_RADIUS**2, 2.0 * np.pi**2 * FF_PRIME


def _solovev(radius, height):
    alpha, offset, beta = _terms()
    return alpha * radius**4 + offset * radius**2 + beta * height**2


def _wall_loop(points: int = 61) -> tuple[np.ndarray, float]:
    alpha, offset, beta = _terms()
    wall_flux = _solovev(AXIS_RADIUS, 0.0) - SEED_SPAN
    inner, outer = np.sqrt(np.sort(np.roots([alpha, offset, -wall_flux])))
    centre, half = 0.5 * (inner + outer), 0.5 * (outer - inner)
    angle = 2.0 * np.pi * np.arange(points) / points
    radius = centre + half * np.cos(angle)
    argument = np.clip((wall_flux - _solovev(radius, 0.0)) / beta, 0.0, None)
    wall = np.c_[radius, np.sign(np.sin(angle)) * np.sqrt(argument)]
    return wall, float(wall_flux)


def _green_block(target: np.ndarray, source: np.ndarray) -> np.ndarray:
    return np.stack(
        [
            hybrid_greens(target[:, 0], target[:, 1], r, z, 0.05, 0.05)[0]
            for r, z in source
        ],
        axis=1,
    )


def _gradient(amplitude: float):
    def tapered(psi_norm):
        return amplitude * (1.0 - jnp.clip(jnp.asarray(psi_norm), 0.0, 1.0))

    return tapered


def build_workload() -> tuple[ForwardProfile, jax.Array]:
    """Build the receipt-bearing free-boundary workload used for every row."""

    lattice = FluxLattice(
        np.linspace(0.6, 1.42, GRID_POINTS),
        np.linspace(-0.42, 0.42, GRID_POINTS),
    )
    coordinate = lattice.coordinate
    wall, wall_flux = _wall_loop()
    seed_grid = _solovev(coordinate[:, 0], coordinate[:, 1])
    seed_wall = _solovev(wall[:, 0], wall[:, 1])
    seed = jnp.asarray(np.r_[seed_grid, seed_wall])
    inside = seed_grid >= wall_flux
    angle = 2.0 * np.pi * np.arange(CONDUCTOR_COUNT) / CONDUCTOR_COUNT
    conductor = np.c_[
        AXIS_RADIUS + 0.62 * np.cos(angle),
        0.62 * np.sin(angle),
    ]
    coupling = {
        "plasma_to_grid": _green_block(coordinate, coordinate),
        "plasma_to_wall": _green_block(wall, coordinate),
        "source_to_grid": _green_block(coordinate, conductor),
        "source_to_wall": _green_block(wall, conductor),
    }

    flat_source = ForwardSource(
        core=DomainProfile(
            p_prime=lambda psi: jnp.full_like(jnp.asarray(psi), P_PRIME),
            ff_prime=lambda psi: jnp.full_like(jnp.asarray(psi), FF_PRIME),
        ),
        boundary_field_function=BOUNDARY_FIELD_FUNCTION,
    )
    flat = ForwardProfile.from_lattice(
        lattice,
        flat_source,
        external_current=np.zeros(CONDUCTOR_COUNT),
        wall_coordinate=wall,
        polarity=1,
        inside_material=inside,
        **coupling,
    )
    cell_current = np.asarray(flat.operator.cell_current(seed))
    target = np.r_[
        seed_grid - coupling["plasma_to_grid"] @ cell_current,
        seed_wall - coupling["plasma_to_wall"] @ cell_current,
    ]
    weight = np.r_[inside.astype(float), np.ones(len(wall))]
    matrix = np.r_[coupling["source_to_grid"], coupling["source_to_wall"]]
    conductor_current = np.linalg.lstsq(
        matrix * weight[:, None], target * weight, rcond=None
    )[0]
    source = ForwardSource(
        core=DomainProfile(
            p_prime=_gradient(2.0 * DRIVE * P_PRIME),
            ff_prime=_gradient(2.0 * DRIVE * FF_PRIME),
        ),
        boundary_field_function=BOUNDARY_FIELD_FUNCTION,
    )
    profile = ForwardProfile.from_lattice(
        lattice,
        source,
        external_current=conductor_current,
        wall_coordinate=wall,
        polarity=1,
        inside_material=inside,
        **coupling,
    )
    return profile, seed


def _branch_receipts(equilibrium) -> list[BranchReceipt]:
    receipts = []
    for member in range(equilibrium.flux.shape[0]):
        plasma_current = float(equilibrium.ledger.total[member])
        diverted = bool(equilibrium.topology.diverted[member])
        identity = (
            "vacuum"
            if not np.isfinite(plasma_current) or abs(plasma_current) <= 1.0e-6
            else ("diverted_plasma" if diverted else "limited_plasma")
        )
        axis = np.asarray(equilibrium.topology.axis[member], dtype=float)
        receipts.append(
            BranchReceipt(
                identity=identity,
                diverted=diverted,
                plasma_current_a=plasma_current,
                axis_r_m=float(axis[0]),
                axis_z_m=float(axis[1]),
            )
        )
    return receipts


def _conditioning(profile, current: np.ndarray, branch: BranchReceipt) -> dict:
    source_flux = np.asarray(profile.operator.grid.source_target) @ current
    source_flux = source_flux.reshape(profile.lattice.shape)
    radius = np.asarray(profile.lattice.radius, dtype=float)
    height = np.asarray(profile.lattice.height, dtype=float)
    vertical_index = int(np.argmin(np.abs(height - branch.axis_z_m)))
    vertical_field = np.gradient(source_flux[:, vertical_index], radius) / (
        2.0 * np.pi * radius
    )
    return asdict(
        vertical_conditioning_receipt(radius, vertical_field, branch.axis_r_m)
    )


def _device_record() -> dict[str, str | int]:
    device = jax.devices()[0]
    return {
        "platform": device.platform,
        "kind": device.device_kind,
        "id": int(device.id),
        "host": platform.node(),
        "jax_version": jax.__version__,
    }


def measure(
    batch_sizes: tuple[int, ...],
    *,
    frames: int,
    repeats: int,
    newton_steps: int,
    gmres_iterations: int,
    warmup: int,
    relaxation: float,
    step_cap: float,
    maximum_axis_step_m: float,
) -> dict:
    """Run the declared budget ladder and retain every solve receipt."""

    configure_dtypes()
    profile, seed = build_workload()
    frame_scale = np.linspace(0.99, 1.01, frames)
    rows = []
    solve_receipts = []
    for batch_size in batch_sizes:
        solve = jax.jit(
            lambda state, current: profile.solve_batch(
                state,
                route="newton_krylov",
                current=current,
                newton_steps=newton_steps,
                gmres_iterations=gmres_iterations,
                warmup=warmup,
                relaxation=relaxation,
                step_cap=step_cap,
            )
        )
        base_current = np.asarray(profile.operator.external_current, dtype=float)
        initial = jnp.repeat(seed[None, :], batch_size, axis=0)
        compile_current = jnp.repeat(base_current[None, :], batch_size, axis=0)
        compiled = solve(initial, compile_current)
        jax.block_until_ready(compiled.flux)

        started = time.perf_counter()
        max_residual = 0.0
        stable_receipts = 0
        total_receipts = 0
        branch_identities: set[str] = set()
        for repeat in range(repeats):
            state = initial
            previous: list[BranchReceipt] | None = None
            for frame, scale in enumerate(frame_scale):
                current = np.repeat((scale * base_current)[None, :], batch_size, axis=0)
                equilibrium = solve(state, jnp.asarray(current))
                jax.block_until_ready(equilibrium.flux)
                branches = _branch_receipts(equilibrium)
                vacuum_members = [
                    member
                    for member, branch in enumerate(branches)
                    if branch.identity == "vacuum"
                ]
                if vacuum_members:
                    raise RuntimeError(
                        "bootstrapped-shot: fixed-budget solve left the plasma "
                        f"branch for ensemble members {vacuum_members}"
                    )
                if previous is not None:
                    validate_warm_start_continuity(
                        "bootstrapped-shot",
                        previous,
                        branches,
                        maximum_axis_step_m=maximum_axis_step_m,
                    )
                previous = branches
                residuals = np.asarray(equilibrium.fixed_point.residual, dtype=float)
                max_residual = max(max_residual, float(np.max(residuals)))
                for member, branch in enumerate(branches):
                    conditioning = _conditioning(profile, current[member], branch)
                    stable_receipts += int(conditioning["stable"])
                    total_receipts += 1
                    branch_identities.add(branch.identity)
                    solve_receipts.append(
                        {
                            "batch_size": batch_size,
                            "repeat": repeat,
                            "frame": frame,
                            "ensemble_member": member,
                            "branch": asdict(branch),
                            "conditioning": conditioning,
                            "relative_residual": float(residuals[member]),
                        }
                    )
                state = equilibrium.flux
        elapsed = time.perf_counter() - started
        slices = batch_size * frames * repeats
        rows.append(
            {
                "device": jax.devices()[0].device_kind,
                "batch_size": batch_size,
                "precision": str(seed.dtype),
                "convergence_policy": (
                    f"fixed {warmup} relaxed warmup + {newton_steps} Newton steps "
                    f"x {gmres_iterations} GMRES iterations; no residual early exit"
                ),
                "frames": frames,
                "repeats": repeats,
                "timed_slices": slices,
                "elapsed_seconds": elapsed,
                "slices_per_second": slices / elapsed,
                "maximum_relative_residual": max_residual,
                "branch_identities": sorted(branch_identities),
                "conditioning_receipts": total_receipts,
                "stable_conditioning_receipts": stable_receipts,
            }
        )
    return {
        "measurement": "batched forward-solve compute budget",
        "device": _device_record(),
        "warm_start": {
            "shot": "bootstrapped-shot",
            "frames_in_order": frames,
            "maximum_axis_step_m": maximum_axis_step_m,
            "policy": "previous solved slice seeds the next slice per ensemble member",
            "branch_switches_accepted": 0,
        },
        "qualification": (
            "throughput is a fixed-budget compute measurement; residual and branch "
            "receipts are reported and are not converted into a convergence claim"
        ),
        "budget_rows": rows,
        "solve_receipts": solve_receipts,
    }


def write_outputs(result: dict, output: Path) -> list[Path]:
    """Publish machine-readable and semantic-table forms of one measurement."""

    output.mkdir(parents=True, exist_ok=True)
    json_path = output / BUDGET_JSON
    csv_path = output / BUDGET_CSV
    html_path = output / BUDGET_HTML
    json_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    columns = (
        "device",
        "batch_size",
        "precision",
        "convergence_policy",
        "timed_slices",
        "elapsed_seconds",
        "slices_per_second",
        "maximum_relative_residual",
        "branch_identities",
        "conditioning_receipts",
        "stable_conditioning_receipts",
    )
    with csv_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in result["budget_rows"]:
            writer.writerow(
                {
                    **row,
                    "branch_identities": ",".join(row["branch_identities"]),
                }
            )
    rows = []
    for row in result["budget_rows"]:
        rows.append(
            "<tr>"
            f"<td>{row['device']}</td><td>{row['batch_size']}</td>"
            f"<td>{row['precision']}</td><td>{row['convergence_policy']}</td>"
            f"<td>{row['slices_per_second']:.6g}</td>"
            f"<td>{row['maximum_relative_residual']:.6g}</td>"
            f"<td>{', '.join(row['branch_identities'])}</td>"
            f"<td>{row['conditioning_receipts']}</td>"
            "</tr>"
        )
    html_path.write_text(
        '<!doctype html><html lang="en"><head><meta charset="utf-8">'
        "<title>Batched forward-solve compute budget</title></head><body>"
        "<h1>Batched forward-solve compute budget</h1>"
        f"<p>{result['qualification']}</p><table><thead><tr>"
        "<th>Device</th><th>Batch</th><th>Precision</th><th>Policy</th>"
        "<th>Slices/s</th><th>Maximum residual</th><th>Branch</th>"
        "<th>conditioning receipts</th></tr></thead><tbody>"
        + "".join(rows)
        + "</tbody></table></body></html>\n"
    )
    return [json_path, csv_path, html_path]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--batch-sizes", default="1,4,8")
    parser.add_argument("--frames", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--newton-steps", type=int, default=1)
    parser.add_argument("--gmres-iterations", type=int, default=4)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--relaxation", type=float, default=0.5)
    parser.add_argument("--step-cap", type=float, default=0.25)
    parser.add_argument("--maximum-axis-step-m", type=float, default=0.08)
    arguments = parser.parse_args()
    batch_sizes = tuple(int(value) for value in arguments.batch_sizes.split(","))
    result = measure(
        batch_sizes,
        frames=arguments.frames,
        repeats=arguments.repeats,
        newton_steps=arguments.newton_steps,
        gmres_iterations=arguments.gmres_iterations,
        warmup=arguments.warmup,
        relaxation=arguments.relaxation,
        step_cap=arguments.step_cap,
        maximum_axis_step_m=arguments.maximum_axis_step_m,
    )
    paths = write_outputs(result, arguments.output)
    for path in paths:
        print(path)


if __name__ == "__main__":
    main()
