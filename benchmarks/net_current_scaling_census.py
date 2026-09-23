"""Census the net-current scaling amplitude lambda per forward route.

The declared-current normalisation scales a source profile by one amplitude so
the booked plasma current meets its target.  That amplitude is the net-current
scaling lambda: ``target_current / sum(unscaled cell current)``, produced by
``ForwardFluxOperator.current_normalisation_amplitude`` and admitted only
inside ``SCALAR_CURRENT_AMPLITUDE_BAND`` (``nova/equilibrium/source.py``).

This driver records lambda at the cold seed and at the terminal state for each
route that can carry it, beside the terminal residual, the converged flag, the
terminal core-cell count and the terminal support current.  It reuses the
production certificate seam (``benchmarks.solovev_certificate``) so a row is
the same solve every other certificate driver runs, and redirects the panel and
part roots into this node's output tree so no committed receipt is overwritten.

Two requested routes do not carry a normalisation amplitude at this revision
and are read from their committed receipts rather than re-solved; the report
states for each why a seed-terminal lambda pair is not available there.

Every row is persisted before the next build begins, so a later row failing
converts no completed evidence into an empty aggregate.  The report reconciles
the measured amplitudes against the banked history and states, per route,
whether lambda stays within the declared neighbourhood of unity on converged
rows.
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import sys
from pathlib import Path
from time import perf_counter
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from benchmarks import solovev_certificate as certificate
from benchmarks.exact_clip_seed_amplitude import _problem
from nova.equilibrium.forward_operator import set_support_clip_mode
from nova.equilibrium.solve_request import ExplicitSolveSeed, ForwardSolveRequest
from nova.equilibrium.source import SCALAR_CURRENT_AMPLITUDE_BAND
from nova.jax.config import configure_dtypes
from scripts.oracle_rebaseline import measure as recovery


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_ROOT = ROOT / "docs/figures/forward-solve-api/net-current-scaling"
REPORT_JSON = "report.json"
REPORT_MD = "report.md"

#: Reference resolution for the four certificate rows (the resolution the
#: banked chord amplitude history is quoted at).
CERTIFICATE_CELLS = -110

#: Inclusive neighbourhood of unity a converged row's lambda should hold.
LAMBDA_NEIGHBOURHOOD = (0.9, 1.1)

#: Amplitude band the declared-current normalisation admits, quoted from the
#: production constant so the report names the same bound the solve enforces.
AMPLITUDE_BAND = SCALAR_CURRENT_AMPLITUDE_BAND

#: Banked history this census reconciles against, quoted from the plan that
#: recorded it.  These are reconciliation targets, never inputs to a solve.
BANKED_HISTORY = {
    "chord weak-rotation-reactor-static -110": {
        "seed_lambda": 0.930,
        "terminal_lambda": 1.050,
        "clip_mode": "chord",
        "case": "weak-rotation-reactor-static",
        "requested_cells": -110,
        "note": "discriminator control arm",
    },
    "exact support weak-rotation-reactor-static -110": {
        "seed_lambda": 2.201,
        "terminal_lambda": 5.561,
        "clip_mode": "exact",
        "case": "weak-rotation-reactor-static",
        "requested_cells": -110,
        "note": "exact-support deficit, a symptom of the dropped clip",
    },
    "diverted production seed": {
        "seed_lambda": 0.98,
        "terminal_lambda": None,
        "clip_mode": "chord",
        "case": "diverted-single-null",
        "requested_cells": None,
        "note": "eighteen S18 trip panels, eighty-three-cell diverted production seed",
    },
}

SOL_LEDGER_RECEIPT = (
    ROOT / "docs/figures/forward-solve-api/sol-ledger-census/sol-ledger-census.json"
)
MAST_BANK_RECEIPT = (
    ROOT / "docs/figures/primary-xpoint-evidence/efit-topology-corroboration.json"
)


def _strict(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _strict(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_strict(item) for item in value]
    if isinstance(value, np.ndarray):
        return _strict(value.tolist())
    if isinstance(value, np.generic):
        return _strict(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(
        json.dumps(_strict(payload), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _revision() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()


def _lane() -> dict[str, Any]:
    return {
        "job_id": os.environ.get("SLURM_JOB_ID"),
        "partition": os.environ.get("SLURM_JOB_PARTITION"),
        "node": os.environ.get("SLURM_JOB_NODELIST"),
        "hostname": socket.gethostname(),
        "cpu_count": int(os.environ.get("SLURM_CPUS_PER_TASK", "1")),
        "jax_platform": jax.default_backend(),
        "jax_enable_x64": bool(jax.config.jax_enable_x64),
        "tmpdir": os.environ.get("TMPDIR"),
    }


def _terminal_flags(equilibrium: Any) -> tuple[bool, str]:
    """Read the terminal state's converged flag and termination name directly.

    The certificate telemetry helper walks the per-trip globalisation arrays,
    and on the reduced-Newton route those arrays desynchronise: a
    zero-dimensional cycle-damping array sits beside a nonzero active-set trip
    count, so the helper raises IndexError.  These two fields are all this
    census needs, and both read as scalars from the fixed-point history.
    """

    history = equilibrium.fixed_point
    converged = bool(np.asarray(history.converged).reshape(-1)[0])
    reason = recovery.fixed_point.FixedPointTerminationReason(
        int(np.asarray(history.termination_reason).reshape(-1)[0])
    )
    return converged, reason.name.lower()


def _terminal_fields(equilibrium: Any, operator: Any, target_current: float):
    """Return the terminal lambda, residual, convergence and support census."""

    terminal = equilibrium.fixed_point
    terminal_flux = np.asarray(equilibrium.flux, dtype=np.float64)
    amplitude = float(equilibrium.normalisation.amplitude)
    converged, termination = _terminal_flags(equilibrium)
    masks = operator.current_domain_masks(terminal_flux)
    core_cells = int(np.sum(np.asarray(masks.core)))
    support_current = float(target_current) / amplitude if amplitude else None
    return {
        "terminal_lambda": amplitude,
        "terminal_fixed_point_residual": (
            float(terminal.residual) if np.isfinite(terminal.residual) else None
        ),
        "converged": converged,
        "termination": termination,
        "terminal_core_cell_count": core_cells,
        "terminal_support_current_a": support_current,
        "normalisation_policy": equilibrium.normalisation.policy_name,
        "terminal_rescaled": bool(equilibrium.normalisation.rescaled),
    }


def _solve_row(
    case_name: str,
    requested_cells: int,
    clip_mode: str,
    route: str,
) -> dict[str, Any]:
    """Build one certificate row, run one route, and census its lambda."""

    started = perf_counter()
    set_support_clip_mode(clip_mode)
    (
        machine,
        _exact,
        _analytic,
        operator,
        profile,
        target_current,
        centroid,
        current_receipt,
    ) = _problem(case_name, requested_cells)
    seed, requested_class, seed_receipt = certificate._production_seed(
        profile, case_name, target_current, centroid, current_receipt
    )
    seed_moments = operator.cell_current_moments(
        jnp.asarray(seed), requested_class=requested_class
    )
    seed_support = float(jnp.sum(seed_moments.cell_current))
    seed_lambda = float(
        operator.current_normalisation_amplitude(target_current, seed_support)
    )
    request = ForwardSolveRequest.from_defaults(
        carrier_identity=(f"census:{case_name}:{requested_cells}:{clip_mode}:{route}"),
        source_profile=profile.source,
        seed_policy=ExplicitSolveSeed(seed),
        policy_overrides={
            "route": route,
            "newton_steps": recovery.NEWTON_STEPS,
            "gmres_iterations": recovery.KRYLOV_ITERATIONS,
            "warmup": 0,
            "kernel_tolerance": certificate.TERMINAL_RESIDUAL_BOUND,
            "qualification_tolerance": certificate.TERMINAL_RESIDUAL_BOUND,
        },
        target_current=target_current,
    )
    solve_receipt = profile.solve(request)
    equilibrium = solve_receipt.equilibrium
    jax.block_until_ready(equilibrium.flux)
    row = {
        "route": route,
        "clip_mode": clip_mode,
        "case": case_name,
        "requested_cells": requested_cells,
        "realised_cells": len(machine.node),
        "target_current_a": float(target_current),
        "seed_lambda": seed_lambda,
        "seed_support_current_a": seed_support,
        "seed": seed_receipt,
        "solver_route": solve_receipt.resolved_defaults.to_dict().get("route", route),
        "elapsed_seconds": perf_counter() - started,
    }
    row.update(_terminal_fields(equilibrium, operator, target_current))
    row["lambda_within_neighbourhood"] = (
        None
        if not row["converged"]
        else bool(
            LAMBDA_NEIGHBOURHOOD[0] <= row["terminal_lambda"] <= LAMBDA_NEIGHBOURHOOD[1]
        )
    )
    row["elapsed_seconds"] = perf_counter() - started
    return row


def _read_sol_ledger() -> dict[str, Any]:
    """Read the committed SOL ledger census; the bank carries no lambda."""

    if not SOL_LEDGER_RECEIPT.exists():
        return {
            "route": "sol-ledger-census",
            "status": "could-not-run",
            "reason": f"receipt absent at {SOL_LEDGER_RECEIPT.relative_to(ROOT)}",
        }
    receipt = json.loads(SOL_LEDGER_RECEIPT.read_text(encoding="utf-8"))
    return {
        "route": "sol-ledger-census",
        "status": "read-from-committed-receipt",
        "receipt": str(SOL_LEDGER_RECEIPT.relative_to(ROOT)),
        "support_clip_mode": receipt.get("support_clip_mode"),
        "plasma_current_a": receipt.get("plasma_current_a"),
        "common_sol_over_plasma_current": receipt.get("common_sol_over_plasma_current"),
        "fixed_point_residual": receipt.get("fixed_point_residual"),
        "converged": receipt.get("converged_boolean"),
        "verdict": receipt.get("verdict"),
        "seed_lambda": None,
        "terminal_lambda": None,
        "reason": (
            "the SOL ledger census books a terminal plasma current and its "
            "common-SOL split; it records no seed state and no declared-current "
            "amplitude, so a seed-to-terminal lambda pair cannot be read from it "
            "without re-evaluating the operator on its fixture"
        ),
    }


def _read_mast_bank() -> dict[str, Any]:
    """Read the twelve MAST bank rows; the bank carries no normalisation."""

    if not MAST_BANK_RECEIPT.exists():
        return {
            "route": "mast-bank-twelve-rows",
            "status": "could-not-run",
            "reason": f"receipt absent at {MAST_BANK_RECEIPT.relative_to(ROOT)}",
        }
    receipt = json.loads(MAST_BANK_RECEIPT.read_text(encoding="utf-8"))
    rows = receipt.get("rows", [])
    carried = sorted(
        {
            key
            for row in rows
            for key in row
            if "amplitude" in key or "lambda" in key or "normalis" in key
        }
    )
    return {
        "route": "mast-bank-twelve-rows",
        "status": "read-from-committed-receipt",
        "receipt": str(MAST_BANK_RECEIPT.relative_to(ROOT)),
        "row_count": len(rows),
        "converged_rows": int(sum(1 for row in rows if row.get("converged"))),
        "amplitude_fields_present": carried,
        "seed_lambda": None,
        "terminal_lambda": None,
        "reason": (
            "each bank row carries a terminal residual, termination reason and "
            "class verdict, not a declared-current amplitude; deriving seed and "
            "terminal lambda needs the MAST machine and corpus, which this node "
            "is not scoped or equipped to reach on the CPU lane"
        ),
    }


def _reconcile(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Compare measured amplitudes against the banked history."""

    index = {
        (
            row["route"],
            row["case"],
            row["requested_cells"],
            row["clip_mode"],
        ): row
        for row in rows
    }
    routes = sorted({row["route"] for row in rows})
    out: dict[str, Any] = {}
    for label, banked in BANKED_HISTORY.items():
        matched = False
        for route in routes:
            measured = index.get(
                (
                    route,
                    banked["case"],
                    banked["requested_cells"],
                    banked["clip_mode"],
                )
            )
            if measured is None:
                continue
            matched = True
            entry: dict[str, Any] = {"banked": banked, "measured": None, "route": route}
            entry["measured"] = {
                "seed_lambda": measured["seed_lambda"],
                "terminal_lambda": measured["terminal_lambda"],
                "converged": measured["converged"],
            }
            for state in ("seed", "terminal"):
                target = banked.get(f"{state}_lambda")
                got = measured.get(f"{state}_lambda")
                entry[f"{state}_delta"] = (
                    None if target is None or got is None else got - target
                )
            entry["reason"] = f"measured on the {route} route"
            out[f"{label} [{route}]"] = entry
        if not matched:
            out[label] = {
                "banked": banked,
                "measured": None,
                "reason": (
                    "no row in this run matches case="
                    f"{banked['case']} requested_cells={banked['requested_cells']} "
                    f"clip_mode={banked['clip_mode']}"
                ),
            }
    return out


def _verdicts(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """State per route whether lambda holds near unity on converged rows."""

    routes: dict[str, dict[str, Any]] = {}
    for row in rows:
        key = f"{row['route']}/{row['clip_mode']}"
        bucket = routes.setdefault(
            key,
            {
                "route": row["route"],
                "clip_mode": row["clip_mode"],
                "rows": 0,
                "converged_rows": 0,
            },
        )
        bucket["rows"] += 1
        if row["converged"]:
            bucket["converged_rows"] += 1
            bucket.setdefault("converged_terminal_lambda", []).append(
                row["terminal_lambda"]
            )
            bucket.setdefault("converged_seed_lambda", []).append(row["seed_lambda"])
    for bucket in routes.values():
        terminal = bucket.get("converged_terminal_lambda", [])
        seed = bucket.get("converged_seed_lambda", [])
        bucket["converged_terminal_lambda"] = terminal
        bucket["converged_seed_lambda"] = seed
        inside = [
            value
            for value in terminal
            if LAMBDA_NEIGHBOURHOOD[0] <= value <= LAMBDA_NEIGHBOURHOOD[1]
        ]
        bucket["terminal_lambda_min"] = min(terminal) if terminal else None
        bucket["terminal_lambda_max"] = max(terminal) if terminal else None
        bucket["all_converged_terminal_within_neighbourhood"] = bool(terminal) and len(
            inside
        ) == len(terminal)
    return routes


def _quantile(values: list[float], q: float) -> float | None:
    return None if not values else float(np.quantile(np.asarray(values), q))


def _plan_rows(routes: set[str]) -> list[tuple[str, str, str]]:
    plan: list[tuple[str, str, str]] = []
    if "certificate" in routes:
        plan.append(("newton_krylov", "chord", "certificate"))
        plan.append(("newton_krylov", "exact", "certificate"))
    if "reduced-newton" in routes:
        plan.append(("reduced_newton", "chord", "reduced-newton"))
    return plan


def _redirect_certificate_roots(output_root: Path) -> None:
    certificate.FIGURE_ROOT = output_root / "certificate" / "panels"
    certificate.PART_ROOT = output_root / "certificate" / "parts"
    certificate.DIAGNOSTIC_ROOT = output_root / "certificate" / "diagnostics"


def _cold_seed_reads(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    reads: list[dict[str, Any]] = []
    for row in rows:
        reads.append(
            {
                "route": "cold-seed",
                "clip_mode": row["clip_mode"],
                "case": row["case"],
                "requested_cells": row["requested_cells"],
                "seed_lambda": row["seed_lambda"],
                "seed_support_current_a": row["seed_support_current_a"],
                "terminal_lambda": row["terminal_lambda"],
                "converged": row["converged"],
                "status": "derived-from-row",
                "reason": (
                    "read off the certificate row above, which records the seed "
                    "state and the terminal state of the same forward solve"
                ),
            }
        )
    return reads


def run(output_root: Path, routes: set[str]) -> dict[str, Any]:
    configure_dtypes()
    if not jax.config.jax_enable_x64:
        raise RuntimeError("the net-current scaling census requires extended precision")

    _redirect_certificate_roots(output_root)

    report: dict[str, Any] = {
        "$id": "nova.net-current-scaling-census",
        "revision": _revision(),
        "lane": _lane(),
        "lambda_definition": (
            "declared-current amplitude target_current / sum(unscaled cell "
            "current); the amplitude admitted within SCALAR_CURRENT_AMPLITUDE_BAND"
        ),
        "amplitude_band": list(AMPLITUDE_BAND) if AMPLITUDE_BAND else None,
        "neighbourhood": list(LAMBDA_NEIGHBOURHOOD),
        "certificate_cells": CERTIFICATE_CELLS,
        "routes_requested": sorted(routes),
        "rows": [],
        "reads": [],
    }
    report_path = output_root / REPORT_JSON
    _write_json(report_path, report)

    plan: list[tuple[str, str, str]] = []
    if "certificate" in routes:
        plan.append(("newton_krylov", "chord", "certificate"))
        plan.append(("newton_krylov", "exact", "certificate"))
    if "reduced-newton" in routes:
        plan.append(("reduced_newton", "chord", "reduced-newton"))

    for route, clip_mode, _label in plan:
        for case_name in certificate.CASE_NAMES:
            row = _solve_row(case_name, CERTIFICATE_CELLS, clip_mode, route)
            report["rows"].append(row)
            _write_json(report_path, report)
            print(
                "NET_CURRENT_ROW "
                f"route={route} mode={clip_mode} case={case_name} "
                f"seed={row['seed_lambda']:.6f} terminal={row['terminal_lambda']:.6f} "
                f"converged={row['converged']} core={row['terminal_core_cell_count']}",
                flush=True,
            )

    if "cold-seed" in routes:
        report["reads"].extend(_cold_seed_reads(report["rows"]))
        _write_json(report_path, report)

    report["reads"].append(_read_sol_ledger())
    report["reads"].append(_read_mast_bank())
    _write_json(report_path, report)

    report["reconciliation"] = _reconcile(report["rows"])
    report["verdicts"] = _verdicts(report["rows"])
    report["completed_rows"] = len(report["rows"])
    report_path_md = output_root / REPORT_MD
    report_path_md.parent.mkdir(parents=True, exist_ok=True)
    report_path_md.write_text(_markdown(report), encoding="utf-8")
    _write_json(report_path, report)
    print(
        f"NET_CURRENT_EXIT rows={report['completed_rows']} "
        f"reads={len(report['reads'])}",
        flush=True,
    )
    return report


def run_single_row(
    output_root: Path,
    route: str,
    clip_mode: str,
    case_name: str,
    row_out: Path,
) -> dict[str, Any]:
    """Run one certificate row and flush it to its own file as it lands."""

    configure_dtypes()
    if not jax.config.jax_enable_x64:
        raise RuntimeError("the net-current scaling census requires extended precision")
    _redirect_certificate_roots(output_root)
    row = _solve_row(case_name, CERTIFICATE_CELLS, clip_mode, route)
    _write_json(
        row_out,
        {
            "$id": "nova.net-current-scaling-row",
            "revision": _revision(),
            "lane": _lane(),
            "row": row,
        },
    )
    print(
        "NET_CURRENT_ROW "
        f"route={route} mode={clip_mode} case={case_name} "
        f"seed={row['seed_lambda']:.6f} terminal={row['terminal_lambda']:.6f} "
        f"converged={row['converged']} core={row['terminal_core_cell_count']}",
        flush=True,
    )
    return row


def assemble(output_root: Path, rows_dir: Path) -> dict[str, Any]:
    """Merge per-row files into the report beside any rows already measured."""

    report = json.loads((output_root / REPORT_JSON).read_text(encoding="utf-8"))
    report["revision"] = _revision()
    seen = {(row["route"], row["clip_mode"], row["case"]) for row in report["rows"]}
    missing: list[str] = []
    for path in sorted(Path(rows_dir).glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        row = payload["row"]
        key = (row["route"], row["clip_mode"], row["case"])
        if key in seen:
            continue
        report["rows"].append(row)
        seen.add(key)

    order = {
        (plan_route, plan_clip, case): index
        for index, (plan_route, plan_clip, _label, case) in enumerate(
            (plan_row + (case,))
            for plan_row in _plan_rows({"certificate", "reduced-newton"})
            for case in certificate.CASE_NAMES
        )
    }
    report["rows"].sort(
        key=lambda row: order.get(
            (row["route"], row["clip_mode"], row["case"]), len(order)
        )
    )

    expected = len(order)
    for key, index in order.items():
        if key not in seen:
            missing.append(f"{key[0]}/{key[1]}/{key[2]}")

    report["reads"] = _cold_seed_reads(report["rows"])
    report["reads"].append(_read_sol_ledger())
    report["reads"].append(_read_mast_bank())
    report["reconciliation"] = _reconcile(report["rows"])
    report["reads"].insert(
        0,
        {
            "route": "assembly",
            "status": "assembled",
            "rows": len(report["rows"]),
            "expected_rows": expected,
            "missing_rows": missing,
            "reason": (
                "per-row files merged over the rows already measured in report.json"
            ),
        },
    )
    report["not_run"] = [
        {
            "row": key,
            "reason": (
                "the exact clip mode did not finish inside a one-hour all_debug "
                "allocation; its job log under logs/ records 'CANCELLED ... DUE TO "
                "TIME LIMIT' and no row file was written, so its seed and terminal "
                "lambda are absent from this census"
                if key.split("/")[1] == "exact"
                else "no per-row file was written for this row"
            ),
        }
        for key in missing
    ]
    report["verdicts"] = _verdicts(report["rows"])
    report["completed_rows"] = len(report["rows"])
    _write_json(output_root / REPORT_JSON, report)
    (output_root / REPORT_MD).write_text(_markdown(report), encoding="utf-8")
    print(
        f"NET_CURRENT_EXIT rows={report['completed_rows']} "
        f"expected={expected} missing={len(missing)} reads={len(report['reads'])}",
        flush=True,
    )
    return report


def _four_dp(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.4f}"


def _markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Net-current scaling census per forward route",
        "",
        f"Revision: `{report['revision']}`",
        "",
        f"lambda is `{report['lambda_definition']}`",
        "",
        f"Amplitude band: `{report['amplitude_band']}`; neighbourhood of unity: "
        f"`{report['neighbourhood']}`; certificate resolution: "
        f"`{report['certificate_cells']}` cells.",
        "",
        "## Per-row lambda at seed and terminal",
        "",
        "| route | clip | case | seed lambda | terminal lambda | converged | termination | core cells | support current (A) |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in report["rows"]:
        lines.append(
            "| {route} | {clip_mode} | {case} | {seed:.4f} | {term:.4f} | {conv} | "
            "{term_reason} | {core} | {support} |".format(
                route=row["route"],
                clip_mode=row["clip_mode"],
                case=row["case"],
                seed=row["seed_lambda"],
                term=row["terminal_lambda"],
                conv=row["converged"],
                term_reason=row["termination"],
                core=row["terminal_core_cell_count"],
                support=(
                    "n/a"
                    if row["terminal_support_current_a"] is None
                    else f"{row['terminal_support_current_a']:.3f}"
                ),
            )
        )
    lines += ["", "## How each route was read", ""]
    for entry in report["reads"]:
        lines.append(
            f"- **{entry['route']}** ({entry.get('status')}): {entry.get('reason', '')}"
        )
    if report.get("not_run"):
        lines += ["", "## Planned rows that did not run", ""]
        for entry in report["not_run"]:
            lines.append(f"- **{entry['row']}**: {entry['reason']}")
    lines += ["", "## Reconciliation against the banked history", ""]
    for label, entry in report["reconciliation"].items():
        measured = entry.get("measured")
        banked = entry["banked"]
        if measured is None:
            lines.append(
                f"- **{label}**: banked seed {banked['seed_lambda']} / terminal "
                f"{banked['terminal_lambda']}; {entry.get('reason', 'no matching row')}"
                f" (banked note: {banked.get('note', '')})."
            )
            continue
        lines.append(
            f"- **{label}**: banked seed {banked['seed_lambda']} / terminal "
            f"{banked['terminal_lambda']} (measured seed "
            f"{_four_dp(measured['seed_lambda'])} / terminal "
            f"{_four_dp(measured['terminal_lambda'])}, converged "
            f"{measured['converged']}); delta seed "
            f"{_four_dp(entry.get('seed_delta'))}, delta terminal "
            f"{_four_dp(entry.get('terminal_delta'))}."
        )
    lines += ["", "## Verdict per route on converged rows", ""]
    for key, bucket in report["verdicts"].items():
        lines.append(
            f"- **{key}**: {bucket['converged_rows']}/{bucket['rows']} converged; "
            f"terminal lambda range {bucket['terminal_lambda_min']} to "
            f"{bucket['terminal_lambda_max']}; all within neighbourhood: "
            f"{bucket['all_converged_terminal_within_neighbourhood']}."
        )
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--routes",
        default="certificate,cold-seed,reduced-newton",
        help="comma-separated subset of certificate,cold-seed,reduced-newton",
    )
    parser.add_argument(
        "--row-out",
        type=Path,
        default=None,
        help="run one --route/--clip-mode/--case row and flush it here",
    )
    parser.add_argument("--route", default=None)
    parser.add_argument("--clip-mode", default=None)
    parser.add_argument("--case", default=None)
    parser.add_argument("--rows-dir", type=Path, default=None)
    arguments = parser.parse_args()
    routes = {name.strip() for name in arguments.routes.split(",") if name.strip()}
    if arguments.rows_dir is not None:
        mode = f"assemble rows_dir={arguments.rows_dir}"
    elif arguments.row_out is not None:
        mode = (
            f"row route={arguments.route} clip={arguments.clip_mode} "
            f"case={arguments.case}"
        )
    else:
        mode = f"junctions={sorted(routes)}"
    print(
        f"NET_CURRENT_CENSUS revision={_revision()} tree={ROOT} "
        f"mode={mode} argv={' '.join(sys.argv)}",
        flush=True,
    )
    if arguments.rows_dir is not None:
        report = assemble(arguments.output_root, arguments.rows_dir)
        return 0 if report["completed_rows"] else 1
    if arguments.row_out is not None:
        absent = [
            name
            for name, value in (
                ("--route", arguments.route),
                ("--clip-mode", arguments.clip_mode),
                ("--case", arguments.case),
            )
            if not value
        ]
        if absent:
            parser.error("--row-out requires " + ", ".join(absent))
        run_single_row(
            arguments.output_root,
            arguments.route,
            arguments.clip_mode,
            arguments.case,
            arguments.row_out,
        )
        return 0
    report = run(arguments.output_root, routes)
    return 0 if report["completed_rows"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
