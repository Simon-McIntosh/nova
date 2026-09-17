from copy import deepcopy
import json

import pytest

from benchmarks.program_scope_census import (
    _executable_size,
    _loop_inventory,
    _render_constant_svg,
    _replication_targets,
    build_rung_report,
    reanalyze,
)
from benchmarks.solve_program_size_gate import (
    MAX_300_EXECUTABLE_BYTES,
    MAX_300_SOLVE_INSTRUCTIONS,
    MarkerCensusRefusal,
    _write_json,
    evaluate_gate,
    marker_census,
    require_live_markers,
    write_semantic_report,
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


def test_semantic_receipt_preserves_nonfinite_refusal_as_null(tmp_path):
    output = tmp_path / "receipt.json"

    _write_json(output, {"residual": float("nan"), "finite": False})

    assert json.loads(output.read_text(encoding="utf-8")) == {
        "finite": False,
        "residual": None,
    }


def _sentinel_line(function):
    for target in _replication_targets().values():
        for item in target["targets"]:
            if item["function"] == function:
                return item["sentinel"]["line"]
    raise KeyError(function)


def _hlo_module(copies):
    """Build synthetic optimised HLO whose traced frames carry the sentinels.

    ``copies`` maps a marker function to the read-body instruction count of
    each traced copy of it, so a degenerate census can be written by hand and
    refused without compiling anything.
    """
    names = sorted(copies)
    functions = "\n".join(f'{index + 1} "{name}"' for index, name in enumerate(names))
    frames = []
    locations = []
    blocks = []
    frame_id = 100
    for name, counts in copies.items():
        for count in counts:
            frames.append(
                f"{frame_id} {{file_location_id={frame_id} parent_frame_id=0}}"
            )
            locations.append(
                f"{frame_id} {{file_name_id=1 function_name_id="
                f"{names.index(name) + 1} line={_sentinel_line(name)}}}"
            )
            body = "\n".join(
                f"  %i{frame_id}.{step} = f64[4] add(f64[4] %x, f64[4] %x), "
                f'metadata={{op_name="jit(body)/jit(body)/add" '
                f"stack_frame_id={frame_id}}}"
                for step in range(count)
            )
            blocks.append(
                f"%fused.{frame_id} {{\n{body}\n"
                f"  ROOT %t{frame_id} = (f64[4]) tuple(%i{frame_id}.0)\n}}\n"
            )
            frame_id += 1
    return (
        "HloModule jit_synthetic, is_scheduled=true\n\n"
        f'FileNames\n1 "/repo/nova/equilibrium/forward_operator.py"\n\n'
        f"FunctionNames\n{functions}\n\n"
        f"FileLocations\n" + "\n".join(locations) + "\n\n"
        "StackFrames\n" + "\n".join(frames) + "\n\n" + "\n".join(blocks)
    )


def test_marker_census_refuses_a_module_without_markers():
    module = (
        "HloModule jit_synthetic, is_scheduled=true\n\n"
        'FileNames\n1 "/repo/nova/equilibrium/forward_operator.py"\n\n'
        'FunctionNames\n1 "ForwardFluxOperator.normalised_current_moments"\n\n'
        "FileLocations\n1 {file_name_id=1 function_name_id=1 line=7}\n\n"
        "StackFrames\n1 {file_location_id=1 parent_frame_id=0}\n\n"
        "%fused.0 {\n"
        '  %a = f64[4] sine(f64[4] %x), metadata={op_name="jit(body)/sine}\n'
        "  ROOT %t = (f64[4]) tuple(%a)\n}\n"
    )

    census = marker_census(module)

    with pytest.raises(MarkerCensusRefusal, match="zero markers"):
        require_live_markers(census)


def test_marker_census_refuses_a_uniform_read_body_column():
    module = _hlo_module(
        {
            "ForwardFluxOperator.normalised_current_moments": [1, 1],
            "ForwardFluxOperator._fixed_design_read": [2],
        }
    )

    census = marker_census(module)

    with pytest.raises(MarkerCensusRefusal, match="uniform column"):
        require_live_markers(census)


def test_marker_census_counts_distinct_copies_and_accepts_a_live_census():
    module = _hlo_module(
        {
            "ForwardFluxOperator.normalised_current_moments": [1, 2, 3],
            "ForwardFluxOperator._fixed_design_read": [2, 3],
        }
    )

    census = marker_census(module)
    require_live_markers(census)

    rows = census["paths"]
    assert rows["current-moment path"]["copies"] == 3
    assert rows["current-moment path"]["read_body_per_copy"] == [1, 2, 3]
    assert rows["current-moment path"]["marker_bearing_computations"] == 3
    assert rows["topology read"]["copies"] == 2
