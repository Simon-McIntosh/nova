"""Record and compare every existing material test's state, history and fields."""

import os
from pathlib import Path

import numpy as np

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


def pytest_sessionfinish(session, exitstatus):
    destination = Path(os.environ["NOVA_MATERIAL_CAPTURE"])
    np.savez(destination, **_records)
    baseline = os.environ.get("NOVA_MATERIAL_BASELINE")
    if baseline:
        with np.load(baseline) as reference:
            assert reference.files, "baseline captured no material outputs"
            counts = dict(magnetisation=0, permeability=0, history=0, field=0)
            for key in reference.files:
                assert key in _records, f"missing baseline output {key}"
                np.testing.assert_allclose(
                    _records[key], reference[key], rtol=1e-12, atol=0, err_msg=key
                )
                counts[key.split("/")[-2]] += 1
            assert all(counts.values()), counts
        print(f"COMPATIBILITY rtol=1e-12 atol=0 compared={counts}", flush=True)
