"""Name the nan input that reaches the current-normalisation guard.

Reproduces the weak 300-cell row's arm-B reference-seed read of
``benchmarks/exact_clip_low_state_discriminator.py`` at the tree's revision and
stages the production moment path one stage at a time on that state, recording
each array the pipeline carries.  The report names which of the guard's two
inputs to ``ForwardOperatorBatch.current_normalisation_amplitude`` is nan, the
first nan array, the function that produced it, and the state (realised cells,
reference trip count, arm) it was reached from.

No file under ``nova/`` is modified; the driver calls the production path.
"""

from __future__ import annotations

import argparse
import inspect
import json
import os
import sys
import traceback
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve()
ROOT = HERE.parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

REQUESTED_CELLS = -300


def _location(function):
    """Return the defining path:line of a function or bound method."""
    target = function
    target = getattr(target, "__func__", target)
    try:
        path = inspect.getsourcefile(target)
        line = inspect.getsourcelines(target)[1]
    except (OSError, TypeError):
        return "unknown"
    if path is None:
        return "unknown"
    text = str(path)
    prefix = str(ROOT) + os.sep
    if text.startswith(prefix):
        text = text[len(prefix):]
    return text + ":" + str(line)


def _stage(label, producer, value):
    """Describe one pipeline array as a stage record."""
    try:
        array = np.asarray(value)
    except (TypeError, ValueError):
        return {"label": label, "producer": producer, "float": False}
    if array.dtype.kind != "f":
        return {"label": label, "producer": producer, "float": False,
                "dtype": str(array.dtype)}
    finite = np.isfinite(array)
    bad = int(np.count_nonzero(~finite))
    record = {
        "label": label,
        "producer": producer,
        "float": True,
        "shape": list(array.shape),
        "size": int(array.size),
        "nonfinite": bad,
        "nan": int(np.count_nonzero(np.isnan(array))),
        "inf": int(np.count_nonzero(np.isinf(array))),
        "all_nonfinite": bool(array.size and bad == array.size),
    }
    if not array.size:
        return record
    where = np.argwhere(~finite) if bad else np.argwhere(finite)
    record["first_index"] = [int(v) for v in where[0]]
    return record


def _scalar(value):
    array = np.asarray(value, dtype=np.float64)
    return {
        "shape": list(array.shape),
        "finite": bool(np.all(np.isfinite(array))),
        "is_nan": bool(array.ndim == 0 and np.isnan(array)),
        "is_inf": bool(array.ndim == 0 and np.isinf(array)),
        "value": float(array) if array.ndim == 0 else None,
    }


STAGES = []


def _first_nonfinite(label, producer, value):
    """Describe the first non-finite float array carried by a returned value."""
    candidates = [("", value)]
    for name in ("cell_current", "psi_norm", "sample_flux", "area", "full_area"):
        if hasattr(value, name):
            candidates.append((name, getattr(value, name)))
    record = None
    for name, item in candidates:
        entry = _stage(label + ("." + name if name else ""), producer, item)
        if not entry.get("float"):
            continue
        if entry.get("nonfinite") and record is None:
            record = entry
    if record is not None:
        STAGES.append(record)
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=HERE)
    parser.add_argument("--status-file", type=Path, default=HERE / "probe-status.txt")
    arguments = parser.parse_args()
    output_dir = arguments.output_dir
    status = "exit0"
    payload = {
        "driver": str(HERE),
        "revision": None,
        "requested_cells": REQUESTED_CELLS,
        "environment": {},
        "state": None,
        "outcome": None,
        "error": None,
        "amplitude_calls": [],
    }
    try:
        import jax

        import benchmarks.exact_clip_low_state_discriminator as disc
        import nova.equilibrium.forward_operator as opmod
        from nova.equilibrium.forward_operator import ForwardOperatorBatch
        from nova.jax.config import (
            configure_dtypes,
            configure_persistent_compilation_cache,
            default_persistent_compilation_cache_root,
        )

        configure_dtypes()
        assert jax.config.jax_enable_x64 is True
        opmod.set_support_clip_mode("exact")
        cache = configure_persistent_compilation_cache(
            default_persistent_compilation_cache_root())
        payload["revision"] = disc._revision()
        payload["lane"] = disc._lane()
        payload["compilation_cache"] = cache.receipt()
        payload["driver_sha256"] = disc._file_digest(Path(disc.__file__))

        context = disc._build_context(REQUESTED_CELLS)
        reference, reference_path = disc._load_reference_part(
            disc.DEFAULT_REFERENCE_PARTS, REQUESTED_CELLS)
        state = np.asarray(reference["render_data"]["terminal_flux_wb"],
                           dtype=np.float64)
        telemetry = reference["solver"]["production_telemetry"]
        payload["state"] = {
            "realised_cells": int(context["grid_count"]),
            "arm": "B",
            "mechanism": "production iteration from current-aligned cold seed",
            "reference_trip_count": int(telemetry["trip_count"]),
            "reference_converged": bool(telemetry["converged"]),
            "reference_part": str(reference_path),
            "reference_part_sha256": disc._file_digest(reference_path),
            "target_current_a": float(context["target_current"]),
            "terminal_flux_shape": list(state.shape),
            "terminal_flux_finite": bool(np.all(np.isfinite(state))),
        }

        original_amplitude = ForwardOperatorBatch.current_normalisation_amplitude

        def guarded_amplitude(target_current, unscaled_current):
            payload["amplitude_calls"].append({
                "label": _location(original_amplitude),
                "phase": "reference-seed-arm",
                "target_current": _scalar(target_current),
                "unscaled_current": _scalar(unscaled_current),
            })
            return original_amplitude(target_current, unscaled_current)

        ForwardOperatorBatch.current_normalisation_amplitude = staticmethod(guarded_amplitude)

        original_field = opmod.flux_field_polynomial
        original_moments = opmod.clipped_support_current_moments

        def guarded_field(*args, **kwargs):
            result = original_field(*args, **kwargs)
            _first_nonfinite("flux_field_polynomial()", _location(original_field), result)
            return result

        def guarded_moments(*args, **kwargs):
            result = original_moments(*args, **kwargs)
            _first_nonfinite("clipped_support_current_moments()",
                             _location(original_moments), result)
            return result

        opmod.flux_field_polynomial = guarded_field
        opmod.clipped_support_current_moments = guarded_moments

        try:
            disc._reference_seed_arm(context, reference, reference_path)
            payload["outcome"] = "no-error"
        except BaseException as error:
            payload["outcome"] = "raised"
            payload["error"] = {
                "type": type(error).__name__,
                "message": str(error),
                "traceback": traceback.format_exc(),
            }
            print("LOW_STATE_AMPLITUDE_PROBE_RAISED", flush=True)
    except BaseException as error:
        payload["outcome"] = "setup-failed"
        payload["error"] = {
            "type": type(error).__name__,
            "message": str(error),
            "traceback": traceback.format_exc(),
        }
        status = "exit1"
        print("LOW_STATE_AMPLITUDE_PROBE_SETUP_FAILED", flush=True)

    payload["stages"] = STAGES
    (output_dir / "nan-inputs.json").write_text(
        json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")
    arguments.status_file.write_text("PROBE_STATUS=" + status + "\n", encoding="utf-8")
    calls = payload["amplitude_calls"]
    summary = {
        "outcome": payload["outcome"],
        "amplitude_calls": len(calls),
        "stages": len(STAGES),
        "first_stage": STAGES[0] if STAGES else None,
        "last_amplitude_call": calls[-1] if calls else None,
        "error_message": None if payload["error"] is None else payload["error"]["message"],
    }
    print("LOW_STATE_AMPLITUDE_PROBE_SUMMARY "
          + json.dumps(summary, sort_keys=True, default=str), flush=True)
    print("LOW_STATE_AMPLITUDE_PROBE_DONE", flush=True)
    return 0 if status == "exit0" else 1


if __name__ == "__main__":
    raise SystemExit(main())