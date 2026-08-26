"""Authority and cache contracts for the topology visual generator."""

from importlib.util import module_from_spec, spec_from_file_location
import json
from pathlib import Path

import numpy as np
import pytest


SCRIPT = (
    Path(__file__).parents[1]
    / "docs/figures/topology-visual-corroboration/generate_topology_visuals.py"
)


def _generator():
    spec = spec_from_file_location("topology_visual_generator", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _operand() -> dict[str, object]:
    point = np.asarray(((1.0, 2.0),))
    return {
        "machine": "test",
        "identity": "fixture",
        "shot": 1,
        "frame": 2,
        "arm": "demonstration",
        "time": 0.0,
        "cell_rz": point,
        "domain_labels": np.asarray((1,), dtype=np.int8),
        "o_candidates": point,
        "x_candidates": point,
        "selected_o": point,
        "selected_x": point,
        "wall_point": point,
        "wall": point,
        "nova_boundary": point,
        "efit_axis": point,
        "efit_x": point,
        "efit_lcfs": point,
        "converged": False,
        "qualification": "test operand",
    }


def test_operand_cache_records_and_enforces_source_identity(tmp_path):
    generator = _generator()
    cache = tmp_path / "operands.npz"
    current_identity = "sha256:current"
    generator._write_cache(
        cache,
        [_operand()],
        {
            "source_path": "benchmarks/current_authority.py",
            "source_identity": current_identity,
        },
    )

    metadata = json.loads(cache.with_suffix(".metadata.json").read_text())
    assert metadata["authority"] == {
        "source_path": "benchmarks/current_authority.py",
        "source_identity": current_identity,
    }
    assert generator._read_cache(cache, current_identity)[0]["identity"] == "fixture"

    with pytest.raises(
        generator.StaleOperandCacheError,
        match=(
            "recorded source identity 'sha256:current'.*"
            "current authority 'sha256:changed'"
        ),
    ):
        generator._read_cache(cache, "sha256:changed")
