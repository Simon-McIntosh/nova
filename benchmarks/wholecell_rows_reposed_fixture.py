"""Bank the whole-cell control rows on the re-posed fixture exterior.

Solves the eight re-posed certificate rows (weak, moderate, strong,
single-null at 1000 then 2500 cells) with the production whole-cell clip mode
and the analytic-clipped fixture exterior, one rung per part, and tables the
three locked acceptance tests per row beside the 1000-to-2500 rung ratios.

These are the whole-cell control rows for the exact-clip re-verification: the
exact-clip rows await the solve-memory fix and are deliberately excluded here.
The driver reuses the production certificate seam (``profile.solve`` over the
re-posed exterior in ``benchmarks.solovev_certificate``), redirecting only the
panel and part roots into this node's output tree and re-rendering each panel
with the residual and converged flag in the caption.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import subprocess
import threading
from time import perf_counter
from typing import Any, Iterator

import jax
import numpy as np

from benchmarks import solovev_certificate as certificate
from nova.equilibrium.forward_operator import set_support_clip_mode, support_clip_mode
from nova.jax.config import configure_dtypes

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_ROOT = ROOT / "docs/figures/cut-cell-current-attribution/wholecell-rows"
ROW_ORDER = certificate.REPOSED_CERTIFICATE_ROWS
CLIP_MODE = "chord"


def _source_revision() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()


def _driver_sha256() -> str:
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def _device_record() -> dict[str, Any]:
    device = jax.devices()[0]
    stats = device.memory_stats() or {}
    return {
        "platform": device.platform,
        "kind": device.device_kind,
        "host": os.uname().nodename,
        "memory_bytes_limit": int(stats.get("bytes_limit", 0)) or None,
    }


def _peak_bytes() -> int:
    try:
        stats = jax.devices()[0].memory_stats() or {}
    except Exception:  # noqa: BLE001 - absent on the host backend
        return 0
    return int(stats.get("peak_bytes_in_use", 0))


def _live_bytes() -> int:
    try:
        stats = jax.devices()[0].memory_stats() or {}
    except Exception:  # noqa: BLE001 - absent on the host backend
        return 0
    return int(stats.get("bytes_in_use", 0))


@contextmanager
def _monitor_live_peak() -> Iterator[list[int]]:
    """Sample live device bytes in the background over one row."""
    peak: list[int] = [0]
    stopped = threading.Event()

    def sample() -> None:
        while not stopped.wait(0.25):
            live = _live_bytes()
            if live > peak[0]:
                peak[0] = live

    worker = threading.Thread(target=sample, daemon=True)
    worker.start()
    try:
        yield peak
    finally:
        stopped.set()
        worker.join()


def _render_caption(row: dict[str, Any], figure_path: Path) -> None:
    """Re-render the committed panel with residual and converged in the caption."""
    data = row["render_data"]
    residual = row["solver"]["terminal_fixed_point_residual"]
    converged = bool(row["solver"]["production_telemetry"]["converged"])
    title = (
        f"{row['case']} · {certificate._slug(row['requested_cells'])} · "
        f"residual={residual:.3e} · converged={'yes' if converged else 'no'}"
    )
    certificate._plot(
        np.asarray(data["coordinates_rz_m"], dtype=np.float64),
        np.asarray(data["terminal_flux_wb"], dtype=np.float64),
        np.asarray(data["analytic_flux_wb"], dtype=np.float64),
        np.asarray(data["derivative_coordinates_rz_m"], dtype=np.float64),
        {
            name: np.asarray(data["error_fields"][name], dtype=np.float64)
            for name in certificate.NORM_FIELDS
        },
        np.asarray(data["boundary_rz_m"], dtype=np.float64),
        np.asarray(data["wall_units_rz_m"][0], dtype=np.float64),
        data["terminal_topology"],
        data["analytic_topology"],
        figure_path,
        title,
    )


def _acceptance_part(
    row: dict[str, Any],
    *,
    peak_live_bytes: int,
    running_peak_bytes: int,
    wall_seconds: float,
) -> dict[str, Any]:
    """Fold solve, memory and wall measures onto the locked acceptance tests."""
    acceptance = certificate._certificate_acceptance_row(row)
    solver = row["solver"]
    telemetry = solver["production_telemetry"]
    topology = row["geometry"]["root_topology"]
    acceptance["solver"] = {
        "clip_mode": support_clip_mode(),
        "qualification": solver["qualification"],
        "trip_count": telemetry["trip_count"],
        "converged": telemetry["converged"],
        "termination": telemetry["termination"],
        "requested_seed_class": solver["requested_seed_class"],
        "terminal_class": topology["class"],
        "booked_amplitude": {
            "seed": solver["lambda_amplitude_history"]["samples"][0]["amplitude"],
            "terminal": solver["lambda_amplitude_history"]["samples"][1]["amplitude"],
        },
        "peak_device_memory_bytes": max(peak_live_bytes, running_peak_bytes),
        "peak_live_bytes_observed": peak_live_bytes,
        "running_peak_bytes": running_peak_bytes,
        "wall_seconds": wall_seconds,
    }
    return acceptance


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(certificate._strict(payload), indent=2, sort_keys=True),
        encoding="utf-8",
    )


def _solve_row(
    case_name: str,
    requested_cells: int,
    *,
    output_root: Path,
) -> dict[str, Any]:
    """Solve one row whole-cell, persist its part and panel, return acceptance."""
    row_started = perf_counter()
    live_peak: list[int] = [0]
    with _monitor_live_peak() as peak:
        row = certificate._measure(case_name, requested_cells)
        live_peak[:] = peak
    running_peak = _peak_bytes()
    wall_seconds = perf_counter() - row_started

    figure_path = certificate._figure_path(case_name, requested_cells)
    _render_caption(row, figure_path)
    figure_sha = hashlib.sha256(figure_path.read_bytes()).hexdigest()

    acceptance = _acceptance_part(
        row,
        peak_live_bytes=live_peak[0],
        running_peak_bytes=running_peak,
        wall_seconds=wall_seconds,
    )
    acceptance["figure"] = {
        "filesystem_path": str(figure_path.relative_to(ROOT)),
        "project_absolute_src": f"/nova/{figure_path.relative_to(ROOT / 'docs')}",
        "sha256": figure_sha,
        "render_source": "wholecell_reposed_row_with_caption",
    }

    part_path = certificate._part_path(case_name, requested_cells)
    part = json.loads(part_path.read_text(encoding="utf-8"))
    part["figure"]["sha256"] = figure_sha
    part["figure"]["caption"] = (
        "solved beside analytic on shared levels; both null sets; wall; "
        "residual and converged flag in the caption"
    )
    certificate._write_json(part_path, part)

    acceptance_part = output_root / "parts" / "acceptance"
    _write_json(
        acceptance_part / f"{case_name}-cells-{abs(requested_cells)}.json",
        acceptance,
    )
    print(
        f"WHOLECELL_ROW case={case_name} requested_cells={requested_cells} "
        f"residual={acceptance['fixed_point']['terminal_residual']} "
        f"converged={acceptance['fixed_point']['converged']} "
        f"axis_error_m={acceptance['distance_to_analytic']['axis_error_m']} "
        f"peak_device_gib="
        f"{acceptance['solver']['peak_device_memory_bytes'] / 2**30:.3f} "
        f"wall_seconds={wall_seconds:.3f}",
        flush=True,
    )
    return acceptance


def _run(output_root: Path, rows: list[tuple[str, int]]) -> dict[str, Any]:
    configure_dtypes()
    if not jax.config.jax_enable_x64:
        raise RuntimeError("the whole-cell control rows require binary64")
    if support_clip_mode() != CLIP_MODE:
        set_support_clip_mode(CLIP_MODE)
    if support_clip_mode() != CLIP_MODE:
        raise RuntimeError(f"whole-cell clip mode {CLIP_MODE!r} is not active")

    certificate.DIAGNOSTIC_ROOT = output_root / "diagnostics"
    certificate.FIGURE_ROOT = output_root / "panels"
    certificate.PART_ROOT = output_root / "parts" / "rows"

    solved: list[dict[str, Any]] = []
    receipt_path = output_root / "receipt.json"
    for case_name, requested_cells in rows:
        try:
            solved.append(
                _solve_row(case_name, requested_cells, output_root=output_root)
            )
        except Exception:
            certificate._write_json(
                receipt_path,
                {
                    "schema": "nova.wholecell-reposed-rows-acceptance",
                    "source_revision": _source_revision(),
                    "driver_sha256": _driver_sha256(),
                    "clip_mode": support_clip_mode(),
                    "lanes": "single H200 job",
                    "rows": solved,
                    "rung_ratios": certificate._certificate_rung_ratios(solved),
                    "completed": False,
                },
            )
            raise
        _write_json(
            receipt_path,
            {
                "schema": "nova.wholecell-reposed-rows-acceptance",
                "source_revision": _source_revision(),
                "driver_sha256": _driver_sha256(),
                "clip_mode": support_clip_mode(),
                "device": _device_record(),
                "rows": solved,
                "rung_ratios": certificate._certificate_rung_ratios(solved),
                "completed": len(solved) == len(rows),
            },
        )
    print("WHOLECELL_ROWS_EXIT=0", flush=True)
    return json.loads(receipt_path.read_text(encoding="utf-8"))


def _parse_rows(arguments: argparse.Namespace) -> list[tuple[str, int]]:
    if arguments.case or arguments.cells:
        if not (arguments.case and arguments.cells):
            raise SystemExit("--case and --cells must be given together")
        return [(arguments.case, arguments.cells)]
    return list(ROW_ORDER)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--case",
        choices=certificate.CASE_NAMES,
        help="solve only this case (must be paired with --cells)",
    )
    parser.add_argument(
        "--cells",
        type=int,
        choices=certificate.MEASUREMENT_REQUESTS,
        help="solve only this requested cell count (must be paired with --case)",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="where panels, parts and the receipt land",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="validate the row plan and clip mode without solving",
    )
    arguments = parser.parse_args()

    rows = _parse_rows(arguments)
    if arguments.dry_run:
        configure_dtypes()
        print("WHOLECELL_DRY_RUN rows=%d" % len(rows))
        for case_name, requested_cells in rows:
            print(f"WHOLECELL_DRY_RUN_ROW case={case_name} cells={requested_cells}")
        print(f"WHOLECELL_DRY_RUN clip_mode={support_clip_mode()} default->{CLIP_MODE}")
        return
    _run(arguments.output_root, rows)


if __name__ == "__main__":
    main()
