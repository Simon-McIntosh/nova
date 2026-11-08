"""Check shared owners and replay recorded decisions through their driver."""

import argparse
import contextlib
import importlib.util
import io
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from scripts.prototypes.bank_fold_vertical import measure as shared


def load(source):
    spec = importlib.util.spec_from_file_location("radius_candidate", source)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def check(source, receipt):
    module = load(source)
    assert module.accepted_read is shared.accepted_read, "acceptance owner was forked"
    assert module.bank_pin_status is shared.bank_pin_status, "bank-row probe was forked"
    assert module.emit is shared.emit, "receipt emitter was forked"
    state = (np.array([[1.0]]), np.array([[0.1]]), np.array([[0.3]]))
    seeds = np.zeros((1, 2, 2))
    machine = SimpleNamespace(
        operands=SimpleNamespace(
            geometry=SimpleNamespace(centre=np.array([[2.0, 0.0]]))
        )
    )
    sentinel = object()
    with patch.object(
        shared,
        "run_unpinned",
        return_value=(sentinel, [(1.001, state, np.array([999.0]))]),
    ) as replay:
        result, accepted = module.run_unpinned(machine, state, seeds)
    replay.assert_called_once_with(machine, state, seeds)
    assert result is sentinel
    assert np.allclose(accepted[0][2], 2.1)

    recorded = json.loads(receipt.read_text())["pinned"]
    index = 0
    previous = 0
    expected_updates = expected_reads = 0

    def correct(
        _step, _evaluate, _state, field, requested, tolerance, max_steps, report_pin
    ):
        nonlocal index, previous, expected_updates, expected_reads
        row = recorded["steps"][index]
        assert requested == row["drive"], f"drive differs at attempt {index}"
        assert int(field[0]) == previous, "rejected state became a warm start"
        assert max_steps == 6 and tolerance == 1e-6
        index += 1
        if row["accepted"]:
            previous = index
        # Controlled counters test aggregation without claiming new physics.
        updates = row["iteration"]
        reads = 9 * updates + 1
        expected_updates += updates
        expected_reads += reads
        report_pin(np.ones(1, dtype=bool))
        return SimpleNamespace(
            state=state,
            field=np.array([index]),
            accepted=row["accepted"],
            residual=np.array([row["residual_max"]]),
            position_error=np.array([row["radius_error_max"]]),
            iteration=row["iteration"],
            steps=updates,
            reads=reads,
        )

    with (
        patch.object(module, "correct_pinned", side_effect=correct),
        patch.object(
            module.drive_continuation,
            "advance_drive",
            wraps=module.drive_continuation.advance_drive,
        ) as owner,
        contextlib.redirect_stdout(io.StringIO()),
    ):
        result, _, field, accepted = module.pinned_measure(
            machine, state, seeds, np.array([2.1]), return_state=True
        )
    assert owner.call_count == 1
    assert index == recorded["substeps"] == result["substeps"]
    assert result["drive"] == recorded["drive"]
    assert result["certified"] == recorded["certified"]
    assert result["updates"] == expected_updates and result["reads"] == expected_reads
    assert int(field[0]) == previous
    assert len(accepted) == sum(row["accepted"] for row in recorded["steps"])
    print(
        f"REUSE acceptance=shared probe=shared emitter=shared unpinned=shared "
        f"adaptive_driver=shared trace_attempts={index} reach={result['drive']}",
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    check(args.source, args.receipt)


if __name__ == "__main__":
    main()
