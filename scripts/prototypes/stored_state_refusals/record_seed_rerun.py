"""Re-run the TCV comparison arm from the record-era terminal flux as its seed.

The earlier comparison run kept its own receipt, including the terminal flux on
the same 2995-node mesh this branch builds.  Its solve refused every globalised
step, so that state sits close to the seed it started from, and it is the only
surviving copy of a record-era state.

This arm applies that state as the seed and the comparison's own fitted
conductor currents, so every input it can compare is the record's own, and asks
whether the recorded terminal residual returns.  Run it with
``submit.sh --runner record_seed_rerun.py``.
"""

from __future__ import annotations

import json
import os
import pickle
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import attribute_input_drift as drift  # noqa: E402  (sets OUTPUT from the environment)

import jax  # noqa: E402
import pytest  # noqa: E402

from nova.equilibrium.solve_request import ExplicitSolveSeed  # noqa: E402
from nova.jax.config import configure_dtypes  # noqa: E402

RECORD_RECEIPT = Path(
    "/home/ITER/mcintos/.cache/nova-inverse-equilibrium-baseline/"
    "efit-comparison-run-copy/fitted-physical-measurement-1318907/"
    "physical-tcv/tcv-seed-1-receipt.pkl"
)


def _difference(seed, stored):
    seed = np.asarray(seed, dtype=float).ravel()
    stored = np.asarray(stored, dtype=float).ravel()
    if seed.shape != stored.shape:
        return dict(
            comparable=False,
            seed_shape=list(seed.shape),
            stored_shape=list(stored.shape),
        )
    difference = seed - stored
    gauge_free = difference - difference.mean()
    span = float(stored.max() - stored.min())
    return dict(
        comparable=True,
        seed_min=float(seed.min()),
        seed_max=float(seed.max()),
        seed_span=float(seed.max() - seed.min()),
        stored_min=float(stored.min()),
        stored_max=float(stored.max()),
        stored_span=span,
        rms=float(np.sqrt(np.mean(difference * difference))),
        gauge_free_rms=float(np.sqrt(np.mean(gauge_free * gauge_free))),
        gauge_free_rms_over_stored_span=float(
            np.sqrt(np.mean(gauge_free * gauge_free)) / span
        ),
        max_abs=float(np.max(np.abs(difference))),
    )


def main():
    if drift.OUTPUT is None:
        raise SystemExit(
            "STORED_STATE_REFUSALS_OUTPUT must name this job's output directory"
        )
    configure_dtypes()
    assert jax.config.jax_enable_x64 is True
    assert jax.default_backend() == "gpu", "a GPU lane is required"
    monkeypatch = pytest.MonkeyPatch()
    data = drift._prepare("tcv", monkeypatch, fitted=True)
    columns, provenance = drift._comparison_columns(data, "tcv")
    columns = np.asarray(columns, dtype=float)
    drift._bind_columns(data, columns)
    current = jax.numpy.asarray(columns)
    request = replace(data["request"], current=current)
    data["profile"] = replace(
        data["profile"],
        operator=replace(data["profile"].operator, external_current=current),
    )
    stored_seed = np.asarray(request.seed_policy.state, dtype=float)
    with open(RECORD_RECEIPT, "rb") as handle:
        record_receipt = pickle.load(handle)
    seed = np.asarray(record_receipt.terminal_state.flux, dtype=float).ravel()
    request = replace(
        request, seed_policy=ExplicitSolveSeed(state=jax.numpy.asarray(seed))
    )
    record = drift._arm(
        data, request, "tcv-record-seed", "tcv", f"{provenance} + record-era seed flux"
    )
    record["seed_source"] = str(RECORD_RECEIPT)
    record["record_seed_reason"] = int(np.asarray(record_receipt.termination_reason))
    record["seed_difference"] = _difference(seed, stored_seed)
    if not record.get("within_factor_three") or not record.get("same_reason_code"):
        raise SystemExit(
            "tcv-record-seed does not reproduce the recorded arm: residual "
            f"{record.get('residual')} against {record['recorded']['residual']} "
            f"(factor {record.get('factor_of_recorded_residual')}), reason "
            f"{record.get('termination_reason')} against "
            f"{record['recorded']['termination_reason']}"
        )
    record["finished"] = True
    drift._write(drift.OUTPUT / "tcv-record-seed.json", record)
    drift._write(
        drift.OUTPUT / "measurement.json",
        dict(
            case="tcv",
            arm="tcv-record-seed",
            device=str(jax.devices()[0]),
            source_revision=os.environ.get("SOURCE_REVISION"),
            record_source=drift.RECORD_SOURCE,
            seed_source=str(RECORD_RECEIPT),
            residual=record.get("residual"),
            termination_reason=record.get("termination_reason"),
            seed_difference=record["seed_difference"],
            finished=True,
        ),
    )
    print(
        "RECORD_SEED "
        + json.dumps(
            dict(
                residual=record.get("residual"),
                termination_reason=record.get("termination_reason"),
                recorded=record.get("recorded"),
            )
        ),
        flush=True,
    )
    print("RECORD_SEED_FINISHED", flush=True)


if __name__ == "__main__":
    main()
