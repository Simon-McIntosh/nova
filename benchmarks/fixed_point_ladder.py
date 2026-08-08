"""Budget-ladder race of the fixed-point accelerators on the sweep map.

Measures relaxed Picard, safeguarded Anderson and exact-tangent Newton-Krylov
on the ReconstructProfile least-squares sweep map at a shared map-evaluation
budget, over a family of cold-seed slices spanning benign, shallow-well and
rich-profile configurations.  Each slice's magnetics are generated
self-consistently (bootstrap solve, then re-measure at the converged state),
so every slice has a genuine fixed point on the confined branch.

Also runs the profile-damping comparison on the rich-profile family: the
undamped least-squares path against ``profile_relaxation < 1`` at matched
sweep counts.

Writes ``fixed_point_ladder.json`` and ``fixed-point-ladder.png`` under
``docs/figures/spine-boundary-accelerator/``.
"""

from __future__ import annotations

import dataclasses
import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import jax.numpy as jnp

from nova.equilibrium import ProfileDegrees, ReconstructProfile
from nova.equilibrium.measurement import Magnetics
from nova.jax.fixed_point import anderson, newton_krylov, picard

BUDGET = 40
CHECKPOINTS = (10, 20, 40)
TOLERANCE = 1.0e-3
RELAXATION = 0.6
OUTPUT = Path(__file__).resolve().parents[1] / "docs/figures/spine-boundary-accelerator"

COLORS = {"picard": "#2a78d6", "anderson": "#eb6834", "newton_krylov": "#1baf7a"}
LABELS = {
    "picard": "relaxed Picard",
    "anderson": "safeguarded Anderson",
    "newton_krylov": "exact-tangent NK",
}

NR = NZ = 17


def build_machine(degrees: ProfileDegrees) -> ReconstructProfile:
    """The shared small machine, through the canonical Green kernels."""
    grid_r = np.linspace(0.65, 1.35, NR)
    grid_z = np.linspace(-0.5, 0.5, NZ)
    radius, height = np.meshgrid(grid_r, grid_z)
    inside = ((radius - 1.0) / 0.3) ** 2 + (height / 0.42) ** 2 <= 1.0
    angle = np.linspace(0.0, 2.0 * np.pi, 96, endpoint=False)
    theta = np.linspace(0.0, 2.0 * np.pi, 16, endpoint=False)
    return ReconstructProfile.from_geometry(
        grid_r=grid_r,
        grid_z=grid_z,
        inside_limiter=inside,
        cell_width=np.array(grid_r[1] - grid_r[0]),
        cell_height=np.array(grid_z[1] - grid_z[0]),
        source_r=np.array([1.0, 1.0]),
        source_z=np.array([0.8, -0.8]),
        source_width=np.array([0.12, 0.12]),
        source_height=np.array([0.12, 0.12]),
        source_names=("shaping_upper", "shaping_lower"),
        magnetics=Magnetics(
            r=1.0 + 0.5 * np.cos(theta),
            z=0.6 * np.sin(theta),
            angle=np.zeros(16),
            flux_loop=np.ones(16, dtype=bool),
        ),
        degrees=degrees,
        axis_seed=(1.0, 0.0),
        wall_r=1.0 + 0.31 * np.cos(angle),
        wall_z=0.43 * np.sin(angle),
        relaxation=RELAXATION,
        topology_levels=48,
        topology_bisections=12,
        topology_rays=32,
    )


def self_consistent_measured(solver, source_current, plasma_current):
    """Bootstrap solve from the cold seed, then re-measure at convergence."""
    initial = solver.initial_flux(source_current, plasma_current)
    seed_cell = jnp.linalg.lstsq(
        solver.plasma_to_grid, initial - solver.source_to_grid @ source_current
    )[0]
    bootstrap = solver.source_to_sensor @ source_current + (
        solver.plasma_to_sensor @ seed_cell
    )
    scale = jnp.full(bootstrap.size, 1.0e-3)
    mask = jnp.ones(bootstrap.size, dtype=bool)
    boot_map = solver.least_squares_map(
        source_current, plasma_current, bootstrap, scale, mask
    )
    boot = newton_krylov(
        boot_map,
        initial,
        newton_steps=8,
        gmres_iterations=6,
        warmup=16,
        relaxation=RELAXATION,
    )
    basis, _topology = solver._profile_basis(boot.state)
    coefficients = solver._least_squares_coefficients(
        basis, source_current, plasma_current, bootstrap, scale, mask
    )
    measured = solver.source_to_sensor @ source_current + solver.plasma_to_sensor @ (
        basis @ coefficients
    )
    return (
        initial,
        (source_current, plasma_current, measured, scale, mask),
        float(boot.residual),
    )


def measured_positions(trace):
    """Residuals at their evaluation positions (NaN tangent slots dropped)."""
    trace = np.asarray(trace)
    finite = np.isfinite(trace)
    return np.flatnonzero(finite) + 1, trace[finite]


def converged_at(trace, budget):
    """Whether the last measured residual within ``budget`` is under tolerance."""
    positions, values = measured_positions(trace)
    inside = positions <= budget
    if not inside.any():
        return False, np.nan
    residual = values[inside][-1]
    return bool(residual < TOLERANCE), float(residual)


def race(solver, initial, args):
    """One slice: the three schemes at the shared budget from the cold seed."""
    sweep_map = solver.least_squares_map(*args)
    runs = {
        "picard": picard(sweep_map, initial, evaluations=BUDGET, relaxation=RELAXATION),
        "anderson": anderson(
            sweep_map, initial, evaluations=BUDGET, relaxation=RELAXATION
        ),
        "newton_krylov": newton_krylov(
            sweep_map,
            initial,
            newton_steps=4,
            gmres_iterations=6,
            warmup=8,
            relaxation=RELAXATION,
        ),
    }
    return {name: np.asarray(run.trace) for name, run in runs.items()}


def main() -> int:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    start = time.time()

    slices = []
    # benign wells x plasma current, and the shallow-well family
    for coil in (-6.0e3, -1.0e4, -1.4e4, -2.0e3, -4.0e3):
        for ip in (3.0e4, 5.0e4, 8.0e4):
            slices.append({"coil": coil, "ip": ip, "degrees": (1, 1)})
    # rich-profile family (under-constrained higher-order fits)
    for ip in (3.0e4, 5.0e4, 8.0e4):
        slices.append({"coil": -1.0e4, "ip": ip, "degrees": (2, 2)})

    machines = {
        (1, 1): build_machine(ProfileDegrees(1, 1)),
        (2, 2): build_machine(ProfileDegrees(2, 2)),
    }

    results = []
    for spec in slices:
        solver = machines[spec["degrees"]]
        source_current = solver.pack_source_currents(
            {"shaping_upper": spec["coil"], "shaping_lower": spec["coil"]}
        )
        initial, args, boot_residual = self_consistent_measured(
            solver, source_current, jnp.asarray(spec["ip"])
        )
        traces = race(solver, initial, args)
        row = {
            "coil": spec["coil"],
            "ip": spec["ip"],
            "degrees": list(spec["degrees"]),
            "bootstrap_residual": boot_residual,
            "traces": {k: v.tolist() for k, v in traces.items()},
        }
        for name, trace in traces.items():
            for budget in CHECKPOINTS:
                ok, residual = converged_at(trace, budget)
                row[f"{name}_at_{budget}"] = {"converged": ok, "residual": residual}
        results.append(row)
        print(
            "coil %+.1e ip %.0e deg %s | boot %.1e | "
            % (spec["coil"], spec["ip"], spec["degrees"], boot_residual)
            + " ".join(
                "%s %.1e" % (name[:3], row[f"{name}_at_{BUDGET}"]["residual"])
                for name in traces
            )
        )

    # profile-damping comparison on the rich-profile family
    damping = []
    rich = machines[(2, 2)]
    for spec in [s for s in slices if s["degrees"] == (2, 2)]:
        source_current = rich.pack_source_currents(
            {"shaping_upper": spec["coil"], "shaping_lower": spec["coil"]}
        )
        initial, args, _boot = self_consistent_measured(
            rich, source_current, jnp.asarray(spec["ip"])
        )
        row = {"ip": spec["ip"]}
        for name, profile_relaxation in (("undamped", 1.0), ("damped", 0.5)):
            solver = dataclasses.replace(
                rich, profile_relaxation=profile_relaxation, iterations=24
            )
            result = solver.least_squares(*args, initial)
            row[name] = {
                "residual": float(result.residual),
                "finite": bool(np.isfinite(np.asarray(result.coefficients)).all()),
                "coefficients": np.asarray(result.coefficients).tolist(),
            }
        damping.append(row)
        print(
            "damping ip %.0e: undamped %.1e damped %.1e"
            % (spec["ip"], row["undamped"]["residual"], row["damped"]["residual"])
        )

    summary = {
        "budget": BUDGET,
        "tolerance": TOLERANCE,
        "checkpoints": list(CHECKPOINTS),
        "n_slices": len(results),
        "conversion": {
            name: {
                str(budget): float(
                    np.mean([r[f"{name}_at_{budget}"]["converged"] for r in results])
                )
                for budget in CHECKPOINTS
            }
            for name in COLORS
        },
        "median_residual_at_budget": {
            name: float(
                np.median([r[f"{name}_at_{BUDGET}"]["residual"] for r in results])
            )
            for name in COLORS
        },
        "damping": damping,
        "wall_seconds": time.time() - start,
    }
    print(json.dumps({k: v for k, v in summary.items() if k != "damping"}, indent=2))

    payload = {"summary": summary, "slices": results}
    (OUTPUT / "fixed_point_ladder.json").write_text(json.dumps(payload, indent=1))
    figure(results, summary)
    return 0


def figure(results, summary):
    """Two panels: median residual trace, and conversion at the checkpoints."""
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.2))

    for name in COLORS:
        pooled = {}
        for row in results:
            positions, values = measured_positions(row["traces"][name])
            for position, value in zip(positions, values, strict=True):
                pooled.setdefault(int(position), []).append(value)
        positions = np.asarray(sorted(pooled))
        median = np.asarray([np.median(pooled[p]) for p in positions])
        axes[0].plot(
            positions,
            median,
            color=COLORS[name],
            lw=2,
            marker="o",
            ms=4,
            label=LABELS[name],
        )
    axes[0].axhline(TOLERANCE, color="#8a8a85", lw=1, ls="--")
    axes[0].text(1.5, TOLERANCE * 1.4, "tolerance", color="#8a8a85", fontsize=9)
    axes[0].set_yscale("log")
    axes[0].set_xlabel("map evaluations")
    axes[0].set_ylabel("median relative residual")
    axes[0].set_title("Residual vs shared evaluation budget", fontsize=11)
    axes[0].legend(frameon=False, fontsize=9)
    axes[0].spines[["top", "right"]].set_visible(False)
    axes[0].grid(axis="y", color="#eceae4", lw=0.8)

    width = 0.26
    checkpoints = summary["checkpoints"]
    base = np.arange(len(checkpoints))
    for index, name in enumerate(COLORS):
        fractions = [summary["conversion"][name][str(b)] for b in checkpoints]
        bars = axes[1].bar(
            base + (index - 1) * width,
            fractions,
            width=width * 0.92,
            color=COLORS[name],
            label=LABELS[name],
        )
        for bar, fraction in zip(bars, fractions, strict=True):
            axes[1].text(
                bar.get_x() + bar.get_width() / 2,
                fraction + 0.02,
                "%.0f%%" % (100 * fraction),
                ha="center",
                fontsize=8,
                color="#4a4a45",
            )
    axes[1].set_xticks(base, ["%d evals" % b for b in checkpoints])
    axes[1].set_ylim(0, 1.12)
    axes[1].set_ylabel("slices converged (tol %.0e)" % TOLERANCE)
    axes[1].set_title(
        "Cold-seed conversion, %d slices" % summary["n_slices"], fontsize=11
    )
    axes[1].legend(frameon=False, fontsize=9)
    axes[1].spines[["top", "right"]].set_visible(False)
    axes[1].grid(axis="y", color="#eceae4", lw=0.8)

    fig.tight_layout()
    fig.savefig(OUTPUT / "fixed-point-ladder.png", dpi=160)
    print("wrote", OUTPUT / "fixed-point-ladder.png")


if __name__ == "__main__":
    raise SystemExit(main())
