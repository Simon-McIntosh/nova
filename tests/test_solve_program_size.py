from copy import deepcopy
import json
from types import SimpleNamespace

import jax.numpy as jnp
import pytest

from benchmarks.program_scope_census import (
    _executable_size,
    _loop_inventory,
    _render_constant_svg,
    build_rung_report,
    reanalyze,
)
from benchmarks.solve_program_size_gate import (
    MAX_300_EXECUTABLE_BYTES,
    MAX_300_SOLVE_INSTRUCTIONS,
    evaluate_gate,
    write_semantic_report,
)
from nova.equilibrium import fixed_point
from nova.equilibrium.forward import ForwardProfile


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
            "compile_seconds": 370.0 if cells == 300 else 386.0,
            "solve": _program(
                solve,
                120,
                8,
                executable_bytes=462 * (1 << 20),
            ),
            "map": _program(mapped, 1, 1),
        }
    return rows


def test_accelerated_program_reuses_one_frozen_partition(monkeypatch):
    class Operator:
        use_linear_moments = False
        moment_geometry = None

        @staticmethod
        def traced_flux_map(_requested_class, _target_current):
            return lambda state, external: state + external

        @staticmethod
        def traced_flux_map_with_shadow(_requested_class, _target_current):
            return lambda state, shadow, external: state + external + shadow

        @staticmethod
        def residual_shadow_mask(state, _requested_class, previous_shadow=None):
            del previous_shadow
            return jnp.zeros_like(state, dtype=bool)

        @staticmethod
        def _frozen_topology_partition(state, _requested_class, previous_shadow=None):
            del previous_shadow
            return SimpleNamespace(residual_shadow=jnp.zeros_like(state, dtype=bool))

        @staticmethod
        def _internal_on_partition(state, _partition, target_current):
            return state * target_current

        @staticmethod
        def _exclude_shadow_residual(state, image, _requested_class, *, shadow):
            del state, shadow
            return image

    captured = {}

    def capture(_mapped, initial, **options):
        captured.update(options)
        return initial

    monkeypatch.setattr(fixed_point, "newton_krylov", capture)
    profile = object.__new__(ForwardProfile)
    profile.operator = Operator()
    profile.newton_steps = 1
    profile._accelerated_program_cache = {}
    raw_shadowed = profile.operator.traced_flux_map_with_shadow(None, 2.0)
    assert not hasattr(raw_shadowed, "_read_frozen_partition")

    program = profile._accelerated_history_program("newton_krylov", target_current=2.0)
    program.lower(jnp.ones(1), jnp.zeros(1))

    shadowed = captured["shadowed_map_fn"]
    partition = shadowed._read_frozen_partition(jnp.ones(1))
    assert callable(shadowed._map_frozen_partition)
    assert callable(shadowed._frozen_partition_shadow)
    assert shadowed._map_frozen_partition(
        jnp.ones(1), partition, jnp.zeros(1)
    ).tolist() == [2.0]


def test_gate_accepts_the_measured_interim_reduction():
    baseline = _receipts()
    candidate = deepcopy(baseline)
    for cells, row in candidate.items():
        row["compile_seconds"] = 20.0 if cells == 300 else 70.0
        row["solve"]["total_instructions"] = 203_078 if cells == 300 else 204_531
        row["solve"]["replication"]["topology read"]["copy_count"] = 2
    candidate[300]["solve"]["executable"]["serialized_bytes"] = 436_447_170
    candidate[1000]["solve"]["executable"] = {
        "serialized_bytes": None,
        "generated_code_bytes": 0,
        "serialization_error": "serialization refused",
    }

    result = evaluate_gate(baseline, candidate)

    assert result["passed"] is True
    assert result["failures"] == []
    assert result["findings"] == [
        "1000-cell solve executable size is absent: serialization refused"
    ]
    assert result["rows"][1]["before"]["operator_copies"] == {
        "current-moment path": 120,
        "topology read": 8,
    }
    assert result["rows"][0]["after"]["operator_copies"] == {
        "current-moment path": 120,
        "topology read": 2,
    }


def test_gate_refuses_a_solve_that_does_not_reduce_the_physical_measures():
    baseline = _receipts()

    result = evaluate_gate(baseline, deepcopy(baseline))

    assert result["passed"] is False
    assert any(
        "instructions did not shrink" in failure for failure in result["failures"]
    )
    assert any(
        "compile time did not shrink" in failure for failure in result["failures"]
    )
    assert any("executable did not shrink" in failure for failure in result["failures"])


def test_gate_refuses_the_300_cell_budgets():
    baseline = _receipts()
    candidate = deepcopy(baseline)
    for cells, row in candidate.items():
        row["compile_seconds"] /= 2
        row["solve"]["total_instructions"] -= 1
    candidate[300]["solve"]["total_instructions"] = MAX_300_SOLVE_INSTRUCTIONS + 1
    candidate[300]["solve"]["executable"]["serialized_bytes"] = (
        MAX_300_EXECUTABLE_BYTES + 1
    )

    result = evaluate_gate(baseline, candidate)

    assert result["passed"] is False
    assert any(
        "instruction" in failure and "limit" in failure
        for failure in result["failures"]
    )
    assert any(
        "executable" in failure and "limit" in failure for failure in result["failures"]
    )


def test_gate_refuses_an_empty_map_sentinel_instead_of_reporting_absence():
    baseline = _receipts()
    candidate = deepcopy(baseline)
    candidate[300]["map"]["replication"]["topology read"]["copy_count"] = 0

    result = evaluate_gate(baseline, candidate)

    assert result["passed"] is False
    assert any(
        "map sentinel 'topology read' saw no known-present path" in item
        for item in result["failures"]
    )


def test_gate_requires_the_300_cell_executable_measure():
    baseline = _receipts()
    candidate = deepcopy(baseline)
    for cells, row in candidate.items():
        row["compile_seconds"] /= 2
        row["solve"]["total_instructions"] -= 1
    del candidate[300]["solve"]["executable"]

    result = evaluate_gate(baseline, candidate)

    assert result["passed"] is False
    assert any(
        "300-cell solve missing executable-size measurement" in failure
        for failure in result["failures"]
    )


def test_gate_records_the_1000_cell_serialisation_refusal_as_a_finding():
    baseline = _receipts()
    candidate = deepcopy(baseline)
    for cells, row in candidate.items():
        row["compile_seconds"] /= 2
        row["solve"]["total_instructions"] -= 1
    candidate[300]["solve"]["total_instructions"] = 200_000
    candidate[300]["solve"]["executable"]["serialized_bytes"] = 430_000_000
    candidate[1000]["solve"]["executable"] = {
        "serialized_bytes": None,
        "generated_code_bytes": 0,
        "serialization_error": "serialization refused",
    }

    result = evaluate_gate(baseline, candidate)

    assert result["passed"] is True
    assert result["failures"] == []
    assert any("serialization refused" in finding for finding in result["findings"])


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


def test_loop_inventory_finds_both_compiled_slice_budgets():
    compiled = [row for row in _loop_inventory() if row["form"] == "jax.lax.fori_loop"]

    assert {row["loop"] for row in compiled} >= {
        "compiled Newton steps",
        "compiled active-set trips",
    }


def test_reanalysis_preserves_executable_measurement(tmp_path, monkeypatch):
    from benchmarks import program_scope_census

    case = "case"
    hlo_dir = tmp_path / "hlo"
    hlo_dir.mkdir()
    (hlo_dir / f"{case}_300c_solve.hlo.txt").write_text("solve")
    (hlo_dir / f"{case}_300c_map.hlo.txt").write_text("map")
    parts_dir = tmp_path / "parts"
    parts_dir.mkdir()
    executable = {
        "serialized_bytes": 123,
        "generated_code_bytes": 0,
        "serialization_error": None,
    }
    (parts_dir / f"{case}_300c.json").write_text(
        json.dumps(
            {
                "compile_seconds": 4.5,
                "solve": {"executable": executable},
                "map": {"executable": executable},
            }
        )
    )

    census = {
        "attributed": [],
        "total_instructions": 1,
        "total_bytes": 1,
    }
    monkeypatch.setattr(
        program_scope_census, "_census_module", lambda *_a, **_k: census.copy()
    )
    monkeypatch.setattr(program_scope_census, "_top", lambda *_a, **_k: [])

    results = reanalyze(case, [300], hlo_dir, tmp_path / "run")

    assert results[300]["compile_seconds"] == 4.5
    assert results[300]["solve"]["executable"] == executable
    assert results[300]["map"]["executable"] == executable


def test_partial_census_report_does_not_require_an_unmeasured_rung(tmp_path):
    row = _receipts()[300]
    row["solve"]["large_constants"] = {
        "groups": {"mesh arrays": {"captured_bytes": 123}}
    }
    figure = tmp_path / "captured.svg"

    report = build_rung_report({300: row})
    _render_constant_svg({300: row}, figure)

    assert "Missing cell counts were not measured" in report
    assert "654,832 / 9,908" in report
    assert "300 cells" in figure.read_text(encoding="utf-8")
    assert "1000 cells" not in figure.read_text(encoding="utf-8")


def test_semantic_report_requires_all_twelve_bit_identical_mast_rows(tmp_path):
    certificate = {
        "passed": True,
        "rows": [
            {
                "case": "certificate",
                "requested_cells": -300,
                "realised_state_values": 4,
                "baseline_seconds": 2.0,
                "candidate_seconds": 1.0,
                "terminal_state_bit_identical": True,
            }
        ],
    }
    mast_rows = [
        {
            "identity": f"member-{index}",
            "dispatch_reference_wall_s": 0.12,
            "same_job_direct_wall_s": 0.09,
            "compiled_host_terminal_flux_ulp": 0,
        }
        for index in range(12)
    ]
    dispatch_rows = [
        {
            "identity": f"member-{index}",
            "compiled": {"program_dispatch_trips": 3},
        }
        for index in range(12)
    ]
    certificate_path = tmp_path / "certificate.json"
    mast_path = tmp_path / "mast.json"
    dispatch_path = tmp_path / "dispatch.json"
    certificate_path.write_text(json.dumps(certificate), encoding="utf-8")
    mast_path.write_text(
        json.dumps(
            {
                "assignment": {"device": "H200"},
                "measurement_revision": "revision",
                "members": mast_rows,
            }
        ),
        encoding="utf-8",
    )
    dispatch_path.write_text(
        json.dumps({"width_one": {"members": dispatch_rows}}), encoding="utf-8"
    )

    result = write_semantic_report(
        certificate_path, mast_path, dispatch_path, tmp_path / "report"
    )

    assert result["passed"] is True
    assert len(result["mast_rows"]) == 12
    assert result["mast_rows"][0]["before_compiled_boundary_ms_per_trip"] == 40.0
    assert result["mast_rows"][0]["after_compiled_boundary_ms_per_trip"] == 30.0
