"""Record and compare material states, residual histories and fields.

Compare each array with rtol=1e-12 and an absolute allowance of 1e-12 times
the largest reference magnitude. The shared array scale admits roundoff in
analytically zero components without hiding relative changes in dominant values.
"""

import os
from pathlib import Path

import numpy as np
import pytest

from nova.biot import magnetisation as iron

_records = {}
_node = ""
_counts = {}


def _record(kind, value):
    key = (_node, kind)
    index = _counts.get(key, 0)
    _counts[key] = index + 1
    _records[f"{_node}/{kind}/{index}"] = np.asarray(value)


def pytest_runtest_setup(item):
    global _node
    _node = item.nodeid


def pytest_sessionstart(session):
    solve = iron.solve_magnetisation
    field = iron.MagnetisationSolution.field

    def observed(*args, **kwargs):
        result = solve(*args, **kwargs)
        _record("magnetisation", result.magnetisation)
        _record("permeability", result.permeability_relative)
        _record("history", result.residual_history)
        return result

    def observed_field(self, points):
        result = field(self, points)
        _record("field", result)
        return result

    iron.solve_magnetisation = observed
    iron.MagnetisationSolution.field = observed_field


def assert_compatible(actual, reference):
    """Require relative agreement with a reference-scaled roundoff floor."""
    scale = np.max(np.abs(reference), initial=0)
    np.testing.assert_allclose(actual, reference, rtol=1e-12, atol=1e-12 * scale)


@pytest.hookimpl(wrapper=True, tryfirst=True)
def pytest_sessionfinish(session, exitstatus):
    result = yield
    destination = Path(os.environ["NOVA_MATERIAL_CAPTURE"])
    np.savez(destination, **_records)
    baseline = os.environ.get("NOVA_MATERIAL_BASELINE")
    if baseline:
        with np.load(baseline) as reference:
            mismatches = {}
            if not reference.files:
                mismatches["baseline"] = "captured no material outputs"
            counts = dict(magnetisation=0, permeability=0, history=0, field=0)
            for key in reference.files:
                if key not in _records:
                    mismatches[key] = "missing baseline output"
                    continue
                counts[key.split("/")[-2]] += 1
                try:
                    assert_compatible(_records[key], reference[key])
                except AssertionError as error:
                    mismatches[key] = str(error)
            if not all(counts.values()):
                mismatches["output kinds"] = f"incomplete comparison counts {counts}"
        print(
            "COMPATIBILITY rtol=1e-12 atol=1e-12*reference_max "
            f"compared={counts} mismatches={len(mismatches)}",
            flush=True,
        )
        for key, message in mismatches.items():
            print(f"COMPATIBILITY FAILED {key}: {message}", flush=True)
        if mismatches:
            session.exitstatus = pytest.ExitCode.TESTS_FAILED
    return result
