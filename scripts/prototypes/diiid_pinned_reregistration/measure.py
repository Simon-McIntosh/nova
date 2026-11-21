"""Measure the pinned DIII-D seed current through the benchmark's own entry point.

Two measurements are taken through ``benchmarks.diiid_current_pinned_forward``'s
``--preflight-frame`` entry path, which reads the corpus row, builds the
prescribed profile and sums the seed current exactly as the benchmark does:

``finite``
    The head as it stands: the prescribed p' and FF' carry EFIT's own boundary
    values through the finite-edge flux function.

``zero``
    The head with only the profile boundary form replaced by the zero-edge
    quintic end cap.  Both profiles are supplied to the builder with no
    EFIT boundary value, so it falls back to its compact quintic outer cap,
    which reaches zero at normalized flux one.  Nothing else is substituted.

The payload is the benchmark's own ``seed_preflight.json``; this module only
drives the entry point and reads the record back.  Edge values are read from
the shared DIII-D builder rather than restated here.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
import contextlib

from benchmarks import diiid_current_pinned_forward as pinned
from benchmarks import diiid_forward_gs_match as gs_match
from nova.jax.config import configure_dtypes


def _entry_point(output: Path, frame: int, *, bypass_edge: bool = False) -> dict:
    """Run the benchmark's own entry point and read back its seed record.

    A ``bypass_edge`` run substitutes the zero-edge cap for the builder's
    prescribed boundary value, so the record's edge block is dropped and the
    bypass is named instead: those values were not the ones the solve used.
    """

    stream = output / "seed_preflight.json"
    if stream.exists():
        stream.unlink()
    argv = sys.argv
    sys.argv = [
        "diiid-current-pinned-forward",
        "--preflight-frame",
        str(frame),
        "--output",
        str(output),
    ]
    error = None
    try:
        pinned.main()
    except RuntimeError as caught:
        error = caught
    finally:
        sys.argv = argv
    if not stream.exists():
        raise RuntimeError(f"entry point wrote no seed record at {stream}")
    record = json.loads(stream.read_text())
    record["entry_point_refused"] = error is not None
    if error is not None:
        record["entry_point_refusal"] = str(error)
    if bypass_edge:
        record.pop("prescribed_profile_edge", None)
        record["prescribed_profile_edge_bypassed"] = (
            "the zero-edge control supplies no boundary value, so these "
            "prescribed values were not the ones the solve used"
        )
        stream.write_text(
            json.dumps(record, indent=2, sort_keys=True, allow_nan=False) + "\n"
        )
    return record


@contextlib.contextmanager
def zero_edge_boundary():
    """Supply both prescribed profiles with no EFIT boundary value."""

    original = gs_match.SampledFluxFunction

    def zero_edge(coordinate, values, edge_value=None):
        del edge_value
        return original(coordinate, values)

    gs_match.SampledFluxFunction = zero_edge
    try:
        yield
    finally:
        gs_match.SampledFluxFunction = original


def _edge_values(declared: dict) -> dict:
    """Read EFIT's p' and FF' on the boundary shell at normalized flux one."""

    columns = tuple(
        dict.fromkeys(
            (
                *pinned._LABEL_COLUMNS,
                *pinned._CURRENT_COLUMNS,
                *pinned._GEOMETRY_COLUMNS,
                *pinned.PLASMA_CURRENT_COLUMNS,
            )
        )
    )
    path = pinned.DEFAULT_DATA / str(declared["shot"])
    row = pinned._read(path, columns)
    row["_source_path"] = str(path)
    psi_norm, p_prime, ff_prime = gs_match._profile_edge_values(
        row, int(declared["frame"])
    )
    return {"psi_norm": psi_norm, "p_prime": p_prime, "ff_prime": ff_prime}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--frame", type=int, default=102)
    arguments = parser.parse_args()

    configure_dtypes()
    declared = next(
        item
        for item in pinned.REPRESENTATIVE_COHORT
        if item["frame"] == arguments.frame
    )
    root = arguments.root
    root.mkdir(parents=True, exist_ok=True)

    head = _entry_point(root / "head", arguments.frame)
    with zero_edge_boundary():
        zero = _entry_point(root / "zero-edge", arguments.frame, bypass_edge=True)
    edge = _edge_values(declared)

    headline = {
        "frame": arguments.frame,
        "head_seed_current_a": head["unscaled_seed_plasma_current_a"],
        "head_seed_profile_amplitude": head["seed_profile_amplitude"],
        "head_qualified": head["qualified"],
        "zero_edge_seed_current_a": zero["unscaled_seed_plasma_current_a"],
        "zero_edge_qualified": zero["qualified"],
        "prescribed_profile_edge": edge,
        "registered_unscaled_source_ip_a": declared["unscaled_source_ip_a"],
    }
    (root / "seed_route.json").write_text(
        json.dumps(headline, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    print(json.dumps(headline, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
