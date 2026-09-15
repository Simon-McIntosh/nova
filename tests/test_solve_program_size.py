from copy import deepcopy

import pytest

from benchmarks.program_scope_census import _executable_size
from benchmarks.solve_program_size_gate import (
    MAX_EXECUTABLE_BYTES,
    evaluate_gate,
)


def _program(instructions, moment_copies, topology_copies, executable_bytes=None):
    program = {
        "total_instructions": instructions,
        "replication": {
            "current-moment path": {"copy_count": moment_copies},
            "topology read": {"copy_count": topology_copies},
        },
    }
    if executable_bytes is not None:
        program["executable"] = {
            "serialized_bytes": executable_bytes,
            "generated_code_bytes": executable_bytes,
            "serialization_error": None,
        }
    return program


def _receipts():
    rows = {}
    for cells, solve, mapped in ((300, 654_832, 9_908), (1000, 662_331, 10_024)):
        rows[cells] = {
            "requested_cells": cells,
            "solve": _program(
                solve,
                120,
                8,
                executable_bytes=462 * (1 << 20),
            ),
            "map": _program(mapped, 1, 1),
        }
    return rows


def test_gate_accepts_one_operator_body_and_the_executable_budget():
    baseline = _receipts()
    candidate = deepcopy(baseline)
    for row in candidate.values():
        row["solve"]["replication"]["current-moment path"]["copy_count"] = 1
        row["solve"]["replication"]["topology read"]["copy_count"] = 1
        row["solve"]["executable"]["serialized_bytes"] = MAX_EXECUTABLE_BYTES

    result = evaluate_gate(baseline, candidate)

    assert result["passed"] is True
    assert result["failures"] == []
    assert result["rows"][1]["before"]["operator_copies"] == {
        "current-moment path": 120,
        "topology read": 8,
    }


def test_gate_refuses_the_recorded_repetition_and_oversized_executable():
    baseline = _receipts()

    result = evaluate_gate(baseline, deepcopy(baseline))

    assert result["passed"] is False
    assert any("120 traced copies" in failure for failure in result["failures"])
    assert any("8 traced copies" in failure for failure in result["failures"])
    assert any(
        "1000-cell solve executable" in failure for failure in result["failures"]
    )


def test_gate_refuses_an_empty_map_sentinel_instead_of_reporting_absence():
    baseline = _receipts()
    candidate = deepcopy(baseline)
    candidate[300]["map"]["replication"]["topology read"]["copy_count"] = 0

    result = evaluate_gate(baseline, candidate)

    assert result["passed"] is False
    assert any(
        "map sentinel 'topology read' saw 0" in item for item in result["failures"]
    )


def test_gate_requires_an_executable_measure():
    baseline = _receipts()
    candidate = deepcopy(baseline)
    del candidate[1000]["solve"]["executable"]

    with pytest.raises(ValueError, match="missing executable-size measurement"):
        evaluate_gate(baseline, candidate)


@pytest.mark.parametrize("generated_as_method", [False, True])
def test_executable_size_accepts_runtime_method_or_property(generated_as_method):
    class Runtime:
        def serialize(self):
            return b"compiled"

    runtime = Runtime()
    if generated_as_method:
        runtime.size_of_generated_code_in_bytes = lambda: 1234
    else:
        runtime.size_of_generated_code_in_bytes = 1234

    class Compiled:
        @staticmethod
        def runtime_executable():
            return runtime

    assert _executable_size(Compiled()) == {
        "serialized_bytes": 8,
        "generated_code_bytes": 1234,
        "serialization_error": None,
    }

