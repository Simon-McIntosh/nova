"""Re-read the banked current census with explicit per-cell support evidence."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
from time import perf_counter

import numpy as np

from nova.jax.config import configure_dtypes

configure_dtypes()
import jax  # noqa: E402

from benchmarks import unit_amplitude_current_census as census  # noqa: E402
from nova.equilibrium.forward_operator import set_support_clip_mode  # noqa: E402


ROOT = Path(__file__).resolve().parents[4]
INSTRUMENT = Path(
    "/home/ITER/mcintos/.config/reckon/crew/reports/nova/"
    "s19-codex-20260916/cca-which-mechanism-puts-current-on-the-low-field-side/"
    "instrument_boundary_band.py"
)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--state", choices=("terminal", "seed"), required=True)
    parser.add_argument("--mode", choices=("chord", "exact"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assert jax.config.jax_enable_x64 is True
    assert jax.default_backend() == "cpu"
    started = perf_counter()
    print(f"building {args.state} {args.mode}", flush=True)
    if args.state == "terminal":
        control = census._read_control_row()
        context = census._build_context(control)
        machine, operator = context["machine"], context["operator"]
        state = control["terminal_flux_wb"]
        target = context["target_current"]
        provenance = str(census.CONTROL_PART)
    else:
        spec = importlib.util.spec_from_file_location("boundary_band", INSTRUMENT)
        instrument = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(instrument)
        machine, operator, _, _, state, target, _, _, _ = instrument._build(
            census.CASE_NAME, census.REQUESTED_CELLS
        )
        provenance = str(INSTRUMENT)
    set_support_clip_mode(args.mode)
    print("reading support", flush=True)
    probe = census._partition_probe(operator, state, args.mode)
    masks = probe["moment_masks"]
    support = probe["profile_support"]
    print("integrating current", flush=True)
    moments = operator.source.current_moments(
        masks,
        operator.support_current_moments,
        support,
        sample_flux=probe["sample_psi_norm"],
    )
    current = np.asarray(moments.cell_current)
    outside = np.asarray(masks.psi_norm) > 1.0
    total = float(current.sum())
    payload = {
        "revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "source_sha256": hashlib.sha256(
            (ROOT / "nova/equilibrium/forward_operator.py").read_bytes()
        ).hexdigest(),
        "state": args.state,
        "mode": args.mode,
        "input": provenance,
        "state_sha256": census._state_digest(state),
        "lane": census._lane_receipt(),
        "target_current_a": target,
        "total_current_a": total,
        "total_over_target": total / target,
        "outside_centroid_cells": int(outside.sum()),
        "outside_centroid_carrying_cells": int(
            np.count_nonzero(outside & (current != 0))
        ),
        "outside_centroid_current_fraction": float(current[outside].sum() / total),
        "positive_current_cells": int(np.count_nonzero(current > 0)),
        "per_cell": [
            {
                "cell": i,
                "current_a": float(current[i]),
                "label": int(probe["base_masks"].label[i]),
                "profile_participation": bool(
                    probe["base_masks"].profile_participation[i]
                ),
                "moment_participation": bool(masks.profile_participation[i]),
                "psi_norm": float(masks.psi_norm[i]),
                "included": bool(support.included[i]),
                "support_area_m2": float(support.area[i]),
                "atomic_area_m2": float(machine.area[i]),
                "centre_rz_m": np.asarray(machine.node[i]).tolist(),
            }
            for i in range(len(current))
        ],
        "topology": {
            name: np.asarray(value).tolist()
            for name, value in probe["topology"]._asdict().items()
        },
        "elapsed_seconds": perf_counter() - started,
    }
    args.output.write_text(
        json.dumps(census._strict(payload), indent=2, allow_nan=False) + "\n"
    )
    np.savez(
        args.output.with_suffix(".npz"), state=state, current=current, node=machine.node
    )
    print(
        json.dumps(
            {
                key: value
                for key, value in payload.items()
                if key not in ("per_cell", "topology")
            }
        ),
        flush=True,
    )
    print(
        "named_cells="
        + json.dumps([payload["per_cell"][i] for i in (106, 104, 108, 83, 116)]),
        flush=True,
    )


if __name__ == "__main__":
    main()
