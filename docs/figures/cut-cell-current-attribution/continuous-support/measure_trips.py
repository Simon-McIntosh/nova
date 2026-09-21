"""Measure a pinned analytic map and capture one production solve's trips."""
# ruff: noqa: E501 -- Captions and persisted HTML retain their literal text.

from __future__ import annotations

import argparse
from contextlib import nullcontext
import hashlib
import json
import os
import subprocess
from pathlib import Path
from time import perf_counter

import numpy as np
from nova.jax.config import configure_dtypes

configure_dtypes()
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import sys  # noqa: E402

sys.path.insert(
    0,
    "/home/ITER/mcintos/.config/reckon/crew/reports/nova/s19-codex-20260916/cca-why-the-certificate-rows-do-not-converge-after-the-clip",
)
from instrument_boundary_band import _build, _cell_record  # noqa: E402
from benchmarks import solovev_certificate as certificate  # noqa: E402
from scripts.analytic_oracle_fixtures import measure as fixture  # noqa: E402

ROOT = Path(os.environ.get("CONTINUOUS_EVIDENCE_ROOT", Path(__file__).resolve().parent))
assert jax.config.jax_enable_x64
COMPILE_EVENTS = []


def compilation_event(name, duration, **metadata):
    if name.startswith("/jax/core/compile/"):
        COMPILE_EVENTS.append({"event": name, "seconds": duration, **metadata})
        if duration > 1.0:
            print("COMPILE_EVENT " + json.dumps(COMPILE_EVENTS[-1]), flush=True)


jax.monitoring.register_event_duration_secs_listener(compilation_event)


def clean(value):
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if isinstance(value, np.ndarray):
        return clean(value.tolist())
    if isinstance(value, np.generic):
        return clean(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def write(path, value):
    path.write_text(json.dumps(clean(value), indent=2, allow_nan=False))


def digest(value):
    return hashlib.sha256(np.asarray(value, dtype=np.float64).tobytes()).hexdigest()


def metric(mapped, state):
    mapped, state = np.asarray(mapped), np.asarray(state)
    delta = np.abs(mapped - state)
    return {
        "relative_sup": float(delta.max() / max(np.abs(mapped).max(), 1e-30)),
        "absolute_sup_wb": float(delta.max()),
        "worst_index": int(np.argmax(delta)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cells", type=int, required=True)
    args = parser.parse_args()
    started = perf_counter()
    row = ROOT / "record" / f"cells-{abs(args.cells)}"
    row.mkdir(parents=True, exist_ok=True)
    os.environ["TRIP_RECORD_DIRECTORY"] = str(row)
    # Existing immutable fixture groups may be consumed without writing locks.
    # A cache miss is refused rather than writing beyond the report fence.
    fixture._cache_lock = lambda store: nullcontext(0.0)

    def refuse_store(*args, **kwargs):
        raise RuntimeError("Fixture cache miss requires an out-of-scope cache write")

    fixture.ZarrStore.store = refuse_store
    case = "weak-rotation-reactor-static"
    (
        machine,
        operator,
        profile,
        exact,
        seed,
        target,
        seed_receipt,
        coordinates,
        analytic,
    ) = _build(case, args.cells)
    print(
        f"BUILT requested={args.cells} realised={len(machine.node)} seconds={perf_counter() - started:.2f} cache={machine.cache}",
        flush=True,
    )
    payload = {
        "revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "quadrature_sha256": hashlib.sha256(
            Path("nova/equilibrium/clip_quadrature.py").read_bytes()
        ).hexdigest(),
        "backend": jax.default_backend(),
        "devices": str(jax.devices()),
        "case": case,
        "requested_cells": args.cells,
        "realised_cells": len(machine.node),
        "target_current_a": target,
        "seed_receipt": seed_receipt,
        "coordinates_r_m": coordinates[:, 0],
        "coordinates_z_m": coordinates[:, 1],
        "analytic_state": analytic,
        "cell_polygons": list(machine.cell_polygons),
        "wall_r_m": np.asarray(machine.wall_node)[:, 0],
        "wall_z_m": np.asarray(machine.wall_node)[:, 1],
        "analytic_axis_rz_m": np.asarray(exact.magnetic_axis),
        "analytic_x_points_rz_m": [],
        "analytic_null_provenance": "Static limited reference: one magnetic axis and no analytic saddle",
        "trips": [],
        "map_checks": {},
        "cache": machine.cache,
    }
    write(row / "receipt.json", payload)
    external = operator.external()
    pin_program = jax.jit(operator.traced_flux_map(target_current=target))

    def mapped(state):
        return pin_program(jnp.asarray(state), external, operator, jnp.asarray(target))

    def snapshot(state, trip, live=None, raw=None):
        record = _cell_record(machine, operator, state, target)
        image = np.asarray(mapped(state))
        shadow = np.asarray(
            operator.residual_shadow_mask(jnp.asarray(state)), dtype=bool
        )
        current = np.asarray(record["cells"]["cell_current_a"])
        axis = record["read_axis_rz_m"]
        entry = {
            "trip": trip,
            "flux": np.asarray(state),
            "record": record,
            "residual": float(live)
            if live is not None
            else metric(image, state)["relative_sup"],
            "live_map_check": metric(image, state),
            "state_digest": digest(state),
            "analytic_error": metric(state, analytic),
            "analytic_max_abs_error_wb": float(np.max(np.abs(state - analytic))),
            "axis_error_m": float(
                np.linalg.norm(np.asarray(axis[0]) - np.asarray(exact.magnetic_axis))
            )
            if axis
            else None,
            "nonzero_current_cells": int(np.count_nonzero(current)),
            "shadow": shadow,
            "converged": bool(
                metric(image, state)["relative_sup"]
                <= certificate.TERMINAL_RESIDUAL_BOUND
            ),
        }
        if raw is not None:
            entry["solver_capture"] = {
                k: np.asarray(raw[k])
                for k in raw.files
                if k not in ("state", "trajectory")
            }
            entry["incoming_to_selected_max_wb"] = float(
                np.max(np.abs(state - raw["incoming"]))
            )
            entry["mask_difference_indices"] = np.flatnonzero(
                raw["mask"] != raw["incoming_mask"]
            )
            entry["mask_difference_cell_indices"] = np.flatnonzero(
                raw["mask"][: len(machine.node)]
                != raw["incoming_mask"][: len(machine.node)]
            )
        else:
            entry["mask_difference_indices"] = []
            entry["mask_difference_cell_indices"] = []
        write(row / f"state-{trip}.json", entry)
        print(
            f"SNAPSHOT {trip} residual={entry['residual']:.10g} amplitude={record['amplitude']:.10g} unscaled={record['unscaled_current_a']:.10g} support={record['supported_cell_count']} band={record['boundary_band_cell_count']} axis_error={entry['axis_error_m']}",
            flush=True,
        )
        return entry

    analytic_entry = snapshot(analytic, "analytic-input")
    payload["analytic_input"] = analytic_entry
    payload["map_checks"]["analytic_pinned"] = analytic_entry["live_map_check"]
    unpinned = jax.jit(operator.traced_flux_map())(
        jnp.asarray(analytic), external, operator
    )
    payload["map_checks"]["analytic_unpinned"] = metric(unpinned, analytic)
    payload["map_checks"]["pin_changes_analytic_image_max_wb"] = float(
        np.max(np.abs(mapped(analytic) - unpinned))
    )
    analytic_image = np.asarray(mapped(analytic))
    payload["analytic_mapped"] = snapshot(analytic_image, "analytic-mapped")
    seed_entry = snapshot(seed, 0)
    payload["trips"].append(seed_entry)
    seed_shadow = np.asarray(seed_entry["shadow"])
    analytic_shadow = np.asarray(analytic_entry["shadow"])
    seed_currents = np.zeros(len(machine.node))
    analytic_currents = np.zeros(len(machine.node))
    for entry, values in (
        (seed_entry, seed_currents),
        (analytic_entry, analytic_currents),
    ):
        values[np.asarray(entry["record"]["cells"]["index"], dtype=int)] = entry[
            "record"
        ]["cells"]["cell_current_a"]
    payload["positive_controls"] = {
        "seed_vs_analytic_shadow_difference_indices": np.flatnonzero(
            seed_shadow != analytic_shadow
        ),
        "seed_vs_analytic_nonzero_current_difference_cells": np.flatnonzero(
            (seed_currents != 0) != (analytic_currents != 0)
        ),
        "seed_vs_analytic_current_max_difference_a": float(
            np.max(np.abs(seed_currents - analytic_currents))
        ),
        "analytic_has_supported_cells": analytic_entry["record"]["supported_cell_count"]
        > 0,
        "analytic_has_boundary_band_cells": analytic_entry["record"][
            "boundary_band_cell_count"
        ]
        > 0,
    }
    write(row / "receipt.json", payload)
    print("ANALYTIC_CHECKS " + json.dumps(clean(payload["map_checks"])), flush=True)
    print(
        "POSITIVE_CONTROLS " + json.dumps(clean(payload["positive_controls"])),
        flush=True,
    )
    request = certificate._certificate_solve_request(
        profile, seed, target, carrier_identity=f"solovev:{case}:{args.cells}"
    )
    payload["policy"] = {
        name: getattr(request.policy, name)
        for name in request.policy.__dataclass_fields__
    }
    write(row / "receipt.json", payload)
    print("PRODUCTION_SOLVE_START " + json.dumps(clean(payload["policy"])), flush=True)
    compile_start = len(COMPILE_EVENTS)
    solve_started = perf_counter()
    receipt = profile.solve(request)
    jax.block_until_ready(receipt.equilibrium.flux)
    jax.effects_barrier()
    solve_wall = perf_counter() - solve_started
    events = COMPILE_EVENTS[compile_start:]
    compile_wall = sum(event["seconds"] for event in events)
    payload["timing"] = {
        "solve_wall_seconds": solve_wall,
        "compiler_events": events,
        "compiler_event_wall_seconds": compile_wall,
        "execute_and_host_wall_seconds": solve_wall - compile_wall,
        "compilation_cache_hit": bool(receipt.compilation_cache_hit),
    }
    history = receipt.equilibrium.fixed_point
    for path in sorted(row.glob("trip-*.npz")):
        with np.load(path) as raw:
            entry = snapshot(
                raw["state"], int(raw["index"]) + 1, float(raw["live_residual"]), raw
            )
        previous = payload["trips"][-1]
        old = np.zeros(len(machine.node))
        current = np.zeros(len(machine.node))
        for item, values in ((previous, old), (entry, current)):
            values[np.asarray(item["record"]["cells"]["index"], dtype=int)] = item[
                "record"
            ]["cells"]["cell_current_a"]
        entry["current_support_changed_cells"] = np.flatnonzero(
            (current != 0) != (old != 0)
        )
        entry["current_change_max_a"] = float(np.max(np.abs(current - old)))
        entry["previous_trip_state_max_difference_wb"] = float(
            np.max(np.abs(np.asarray(previous["flux"]) - entry["flux"]))
        )
        payload["trips"].append(entry)
        write(row / "receipt.json", payload)
    payload["terminal"] = {
        "residual": float(history.residual),
        "converged": bool(history.converged),
        "trips": int(history.active_set_iterations),
        "state_digest": digest(receipt.equilibrium.flux),
        "termination_reason": str(receipt.termination_reason),
        "solver": certificate._production_solver_receipt(receipt.equilibrium),
    }
    payload["wall_seconds"] = perf_counter() - started
    write(row / "receipt.json", payload)
    print("TERMINAL " + json.dumps(clean(payload["terminal"])), flush=True)


if __name__ == "__main__":
    main()
