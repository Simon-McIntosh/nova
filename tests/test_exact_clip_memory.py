"""Focused contracts for exact-clip memory attribution."""

import json
import os
from pathlib import Path
import subprocess
import sys
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from benchmarks import exact_clip_memory_scaling as memory_scaling
from nova.equilibrium.clip_quadrature import (
    clipped_support_current_moments,
    clipped_support_field_integrals,
)
from nova.equilibrium.forward_operator import (
    _cell_banked_current_moments,
    _cell_banked_field_integrals,
    _implicit_level_root,
    _implicit_traced_clip,
)
from nova.equilibrium.separatrix_clip import AtomicCellMesh, TracedClippedSupports
from nova.equilibrium.stencil_mesh import FluxFieldPolynomial
from nova.jax.config import configure_dtypes


MAIN_CHECKOUT = Path("/home/ITER/mcintos/Code/nova")
WORKTREE = Path(__file__).resolve().parents[1]
WHOLE_CELL_CONTROL = (
    WORKTREE
    / "docs/figures/cut-cell-current-attribution/limited-shadow/solve-parts/chord"
    / "diverted-single-null-production-route-cells-500.json"
)

TERMINAL_DRIVER = """\
from __future__ import annotations

import argparse
from pathlib import Path

import jax
import numpy as np

from benchmarks import solovev_certificate as certificate
from nova.equilibrium import clip_quadrature
from nova.equilibrium.forward_operator import set_support_clip_mode
from nova.jax.config import configure_dtypes


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    parser.add_argument("case")
    parser.add_argument("requested_cells", type=int)
    parser.add_argument("mode", choices=("chord", "exact", "whole_cell"))
    arguments = parser.parse_args()

    configure_dtypes()
    assert jax.config.jax_enable_x64 is True
    if not hasattr(certificate.observation, "_UNIT_NODE"):
        certificate.observation._UNIT_NODE = clip_quadrature._UNIT_NODE
    if arguments.mode == "whole_cell":
        from benchmarks import limited_row_shadow_census

        certificate.oracle_fixture.cached_fixture_exterior = (
            limited_row_shadow_census._whole_cell_fixture_exterior
        )
        set_support_clip_mode("chord")
    else:
        set_support_clip_mode(arguments.mode)
    profile, seed, request, dimensions = certificate._certificate_compile_problem(
        arguments.case, arguments.requested_cells
    )
    solved = profile.solve(request).equilibrium
    jax.block_until_ready(solved.flux)
    np.savez(
        arguments.output,
        flux=np.asarray(solved.flux, dtype=np.float64),
        residual=np.asarray(float(solved.fixed_point.residual), dtype=np.float64),
        amplitude=np.asarray(float(solved.normalisation.amplitude), dtype=np.float64),
        converged=np.asarray(bool(solved.fixed_point.converged)),
        realised_cells=np.asarray(dimensions["realised_cells"], dtype=np.int64),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
"""


def _run_terminal_driver(
    checkout: Path,
    driver: Path,
    output: Path,
    case: str,
    requested_cells: int,
    mode: str,
) -> None:
    """Run one production solve in a fresh CPU process and require its receipt."""
    if output.is_file():
        _terminal_arrays(output)
        return
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(checkout)
    environment["JAX_PLATFORMS"] = "cpu"
    result = subprocess.run(
        [
            sys.executable,
            str(driver),
            str(output),
            case,
            str(requested_cells),
            mode,
        ],
        cwd=str(WORKTREE),
        env=environment,
        capture_output=True,
        text=True,
        timeout=3500,
    )
    if result.returncode != 0 or not output.is_file():
        raise AssertionError(
            f"terminal driver failed against {checkout}:\nstdout:\n{result.stdout}"
            f"\nstderr:\n{result.stderr}"
        )


def _terminal_arrays(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as receipt:
        return {name: np.asarray(receipt[name]) for name in receipt.files}


def test_scaling_exponent_identifies_linear_and_pairwise_growth():
    """Power-law attribution distinguishes per-cell from all-cell pairs."""
    assert memory_scaling.scaling_exponent(100, 300, 10, 30) == pytest.approx(1.0)
    assert memory_scaling.scaling_exponent(100, 900, 10, 30) == pytest.approx(2.0)


@pytest.mark.parametrize("value", [0, -1])
def test_scaling_exponent_refuses_nonpositive_measurements(value):
    """An empty or signed measurement cannot support a memory-law claim."""
    with pytest.raises(ValueError, match="positive"):
        memory_scaling.scaling_exponent(value, 300, 10, 30)


def test_measure_uses_the_current_quadrature_node_owner(monkeypatch, tmp_path):
    """The memory probe reaches compilation after the quadrature module split."""
    monkeypatch.delattr(memory_scaling.certificate.observation, "_UNIT_NODE", False)
    monkeypatch.setattr(memory_scaling, "configure_dtypes", lambda: None)
    monkeypatch.setattr(memory_scaling, "support_clip_mode", lambda: "chord")
    monkeypatch.setattr(memory_scaling, "set_support_clip_mode", lambda _mode: None)
    monkeypatch.setattr(memory_scaling.certificate, "_source_revision", lambda: "abc")
    monkeypatch.setattr(
        memory_scaling.certificate, "_lane", lambda: {"platform": "cpu"}
    )

    def compiled(*_args, **_kwargs):
        assert (
            memory_scaling.certificate.observation._UNIT_NODE
            is memory_scaling.clip_quadrature._UNIT_NODE
        )
        return {
            "requested_cells": -110,
            "realised_cells": 132,
            "memory_analysis": {"temp_size_in_bytes": 1024},
            "qualifying_array_signatures": [],
        }

    monkeypatch.setattr(memory_scaling.certificate, "_compile_solve_memory", compiled)
    part_root = tmp_path / "parts"
    receipt = memory_scaling.measure(
        tmp_path / "receipt.json", tmp_path, [110], part_root=part_root
    )
    assert receipt["completed"] is True
    part = memory_scaling.json.loads(
        (part_root / "requested-110.json").read_text(encoding="utf-8")
    )
    assert part["row"] == receipt["rows"][0]


def test_measure_can_bank_memory_without_serializing_hlo(monkeypatch, tmp_path):
    """A protobuf-sized program still yields its executable memory receipt."""
    monkeypatch.setattr(memory_scaling, "configure_dtypes", lambda: None)
    monkeypatch.setattr(memory_scaling, "support_clip_mode", lambda: "chord")
    monkeypatch.setattr(memory_scaling, "set_support_clip_mode", lambda _mode: None)
    monkeypatch.setattr(memory_scaling.certificate, "_source_revision", lambda: "abc")
    monkeypatch.setattr(
        memory_scaling.certificate, "_lane", lambda: {"platform": "gpu"}
    )

    def compiled(*_args, **_kwargs):
        return {
            "requested_cells": -1000,
            "realised_cells": 1065,
            "memory_analysis": {"temp_size_in_bytes": 4 * 2**30},
            "qualifying_array_signatures": [],
        }

    monkeypatch.setattr(memory_scaling, "_compile_memory_only", compiled)
    monkeypatch.setattr(
        memory_scaling.certificate,
        "_compile_solve_memory",
        lambda *_args, **_kwargs: pytest.fail("HLO serialization was attempted"),
    )
    receipt = memory_scaling.measure(
        tmp_path / "receipt.json",
        tmp_path / "hlo",
        [1000],
        capture_hlo=False,
    )
    assert receipt["completed"] is True
    assert receipt["rows"][0]["memory_analysis"]["temp_size_in_bytes"] == 4 * 2**30


def test_jvp_terminal_state_reader_refuses_empty_receipt(tmp_path):
    """An empty state cannot make a derivative check vacuously pass."""
    path = tmp_path / "empty.npz"
    np.savez(path, flux=np.empty(0, dtype=np.float64))
    with pytest.raises(RuntimeError, match="empty or nonfinite"):
        memory_scaling._terminal_state(path)


class _MeasuredDevice:
    def __init__(self, peak_bytes: int):
        self.peak_bytes = peak_bytes

    def __str__(self):
        return "mock accelerator"

    def memory_stats(self):
        return {
            "bytes_in_use": self.peak_bytes // 2,
            "peak_bytes_in_use": self.peak_bytes,
        }


def _mock_production_solve(monkeypatch, peak_bytes: int) -> None:
    monkeypatch.setattr(memory_scaling, "configure_dtypes", lambda: None)
    monkeypatch.setattr(memory_scaling, "support_clip_mode", lambda: "chord")
    monkeypatch.setattr(memory_scaling, "set_support_clip_mode", lambda _mode: None)
    monkeypatch.setattr(
        memory_scaling.jax, "devices", lambda: [_MeasuredDevice(peak_bytes)]
    )
    monkeypatch.setattr(memory_scaling.certificate, "_source_revision", lambda: "abc")
    monkeypatch.setattr(
        memory_scaling.certificate, "_lane", lambda: {"platform": "gpu"}
    )

    def measured(_case, requested_cells):
        row = {
            "realised_cells": 1065,
            "solver": {"terminal_fixed_point_residual": 1.25e-8},
            "figure": {
                "project_absolute_src": (
                    "/nova/figures/cut-cell-current-attribution/"
                    "exact-clip-memory/solve-panels/weak.png"
                ),
                "sha256": "figure-digest",
            },
        }
        path = memory_scaling.certificate._part_path(_case, requested_cells)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(row), encoding="utf-8")
        figure = memory_scaling.certificate._figure_path(_case, requested_cells)
        figure.parent.mkdir(parents=True, exist_ok=True)
        figure.write_bytes(b"figure")
        return row

    monkeypatch.setattr(memory_scaling.certificate, "_measure", measured)


def test_production_solve_persists_positive_allocator_peak(monkeypatch, tmp_path):
    """The production receipt carries a peak from an allocator that saw work."""
    _mock_production_solve(monkeypatch, 3 * 2**30)
    original_figures = memory_scaling.certificate.FIGURE_ROOT
    original_parts = memory_scaling.certificate.PART_ROOT
    output = tmp_path / "solve.json"
    receipt = memory_scaling.solve_and_measure(
        output,
        tmp_path / "panels",
        tmp_path / "parts",
        1000,
    )
    assert output.is_file()
    assert receipt["allocator"]["peak_gib"] == 3.0
    assert receipt["allocator"]["instrument_check"] == {
        "byte_counter_count": 2,
        "largest_observed_byte_counter": 3 * 2**30,
        "positive_peak_bytes_in_use": True,
    }
    assert memory_scaling.certificate.FIGURE_ROOT == original_figures
    assert memory_scaling.certificate.PART_ROOT == original_parts


def test_production_solve_refuses_zero_allocator_report(monkeypatch, tmp_path):
    """A uniformly zero allocator report cannot masquerade as low memory."""
    _mock_production_solve(monkeypatch, 0)
    with pytest.raises(RuntimeError, match="did not see the solve"):
        memory_scaling.solve_and_measure(
            tmp_path / "solve.json",
            tmp_path / "panels",
            tmp_path / "parts",
            1000,
        )


class _ConstantCurrentProfile:
    def current_density(self, radius, psi_norm):
        return jnp.ones_like(radius) * 2.0 + 0.0 * psi_norm


def _three_cell_support() -> TracedClippedSupports:
    vertices = np.asarray(
        [
            [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]],
            [[1.0, 0.0], [2.0, 0.0], [1.0, 1.0], [0.0, 0.0]],
            [[2.0, 0.0], [3.0, 0.0], [3.0, 1.0], [2.0, 1.0]],
        ]
    )
    zeros = np.zeros(3)
    return TracedClippedSupports(
        support_vertices=vertices,
        vertex_count=np.asarray([4, 3, 4]),
        centroids=np.asarray([[0.5, 0.5], [4.0 / 3.0, 1.0 / 3.0], [2.5, 0.5]]),
        included=np.asarray([True, False, False]),
        boundary=np.asarray([False, True, False]),
        area=np.asarray([1.0, 0.5, 0.0]),
        full_area=np.asarray([1.0, 1.0, 1.0]),
        first_area_moment=np.zeros((3, 2)),
        second_area_moment=np.zeros((3, 2, 2)),
        contour_area=zeros,
        patch_area_sum=np.asarray(1.5),
        branch_support_vertices=np.zeros((3, 2, 4, 2)),
        branch_vertex_count=np.zeros((3, 2), dtype=np.int32),
        branch_area=np.zeros((3, 2)),
        branch_first_area_moment=np.zeros((3, 2, 2)),
        branch_second_area_moment=np.zeros((3, 2, 2, 2)),
        saddle=np.zeros(3, dtype=bool),
        saddle_vertex=np.zeros((3, 2)),
    )


def test_cell_banked_current_moments_are_bit_identical():
    """Per-cell cut integration retains the existing current-moment result."""
    support = _three_cell_support()
    field = FluxFieldPolynomial(
        coefficient=jnp.zeros((3, 6)),
        centre=jnp.asarray(support.centroids),
        scale=jnp.ones((3, 2)),
        active=jnp.ones(3, dtype=bool),
    )
    selection = jnp.asarray([True, True, False])
    profile = _ConstantCurrentProfile()
    expected = clipped_support_current_moments(
        support,
        selection,
        field,
        profile,
        cut_cell_capacity=3,
    )
    actual = _cell_banked_current_moments(
        support,
        selection,
        field,
        profile,
        cut_cell_capacity=3,
    )
    for one, other in zip(expected, actual, strict=True):
        assert np.array_equal(np.asarray(one), np.asarray(other))


def test_cell_banked_field_integrals_are_bit_identical():
    """Per-cell observation integration retains the existing field result."""
    support = _three_cell_support()
    field = FluxFieldPolynomial(
        coefficient=jnp.zeros((3, 6)),
        centre=jnp.asarray(support.centroids),
        scale=jnp.ones((3, 2)),
        active=jnp.ones(3, dtype=bool),
    )
    selection = jnp.asarray([True, True, False])

    def pressure(radius, psi_norm, boundary_pressure, flux_span):
        return radius + psi_norm + boundary_pressure + flux_span

    expected = clipped_support_field_integrals(
        support,
        selection,
        field,
        pressure,
        0.25,
        2.0,
        cut_cell_capacity=3,
    )
    actual = _cell_banked_field_integrals(
        support,
        selection,
        field,
        pressure,
        0.25,
        2.0,
        cut_cell_capacity=3,
    )
    for one, other in zip(expected, actual, strict=True):
        assert np.array_equal(np.asarray(one), np.asarray(other))


class _LinearCurve(NamedTuple):
    offset: jnp.ndarray

    def __call__(self, points):
        return self.offset - points[..., 0]


def test_implicit_traced_clip_primal_is_bit_identical():
    """The implicit root derivative leaves every primal support bit unchanged."""
    mesh = AtomicCellMesh.from_cells(
        (
            np.asarray([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]),
            np.asarray([[1.0, 0.0], [2.0, 0.0], [2.0, 1.0], [1.0, 1.0]]),
        )
    )
    signed = jnp.asarray(0.5 - mesh.node_coordinates[:, 0])
    participation = jnp.asarray([True, True])
    curve = _LinearCurve(jnp.asarray(0.5))

    expected = mesh.traced_clip(
        signed,
        curve_evaluator=curve,
        participating_cell=participation,
    )
    actual = _implicit_traced_clip(
        mesh.node_coordinates,
        mesh.cell_nodes,
        mesh.cell_vertex_count,
        mesh.centroids,
        mesh.support_capacity,
        signed,
        curve_evaluator=curve,
        participating_cell=participation,
    )
    for one, other in zip(expected, actual, strict=True):
        np.testing.assert_array_equal(np.asarray(one), np.asarray(other))


def test_implicit_level_root_jvp_matches_central_difference():
    """The polished root tangent is the derivative of its converged primal."""
    configure_dtypes()
    assert jax.config.jax_enable_x64 is True
    chord = jnp.zeros((1, 3, 2), dtype=jnp.float64)
    normal = jnp.asarray([[1.0, 0.0]], dtype=jnp.float64)
    lower = jnp.full((1, 1), -1.0, dtype=jnp.float64)
    upper = jnp.full((1, 1), 1.0, dtype=jnp.float64)

    def root(offset):
        return _implicit_level_root(
            chord,
            normal,
            lower,
            upper,
            _LinearCurve(offset),
        )[0, 1]

    offset = jnp.asarray(0.25, dtype=jnp.float64)
    _primal, tangent = jax.jvp(root, (offset,), (jnp.ones_like(offset),))
    step = jnp.asarray(1.0e-6, dtype=jnp.float64)
    central = (root(offset + step) - root(offset - step)) / (2.0 * step)
    assert float(tangent) == pytest.approx(float(central), rel=1.0e-10)

    clamped = jnp.asarray(2.0, dtype=jnp.float64)
    _primal, clamped_tangent = jax.jvp(root, (clamped,), (jnp.ones_like(clamped),))
    clamped_central = (root(clamped + step) - root(clamped - step)) / (2.0 * step)
    assert float(clamped_tangent) == pytest.approx(float(clamped_central), abs=1.0e-14)


@pytest.mark.slow
def test_exact_clip_terminal_state_matches_reference_bit_for_bit(tmp_path):
    """The weak 300-cell exact solve retains every terminal binary64 value."""
    output_root = Path(os.environ.get("NOVA_EXACT_CLIP_IDENTITY_ROOT", tmp_path))
    output_root.mkdir(parents=True, exist_ok=True)
    driver = output_root / "terminal_driver.py"
    driver.write_text(TERMINAL_DRIVER, encoding="utf-8")
    reference_path = output_root / "weak-300-reference.npz"
    current_path = output_root / "weak-300-bounded-current.npz"
    for checkout, output in (
        (MAIN_CHECKOUT, reference_path),
        (WORKTREE, current_path),
    ):
        _run_terminal_driver(
            checkout,
            driver,
            output,
            "weak-rotation-reactor-static",
            -300,
            "exact",
        )
    reference = _terminal_arrays(reference_path)
    current = _terminal_arrays(current_path)
    assert int(reference["realised_cells"]) == 342
    assert reference["flux"].size > int(reference["realised_cells"])
    assert reference.keys() == current.keys()
    for name in reference:
        np.testing.assert_array_equal(current[name], reference[name])


@pytest.mark.slow
def test_whole_cell_terminal_state_matches_committed_control(tmp_path):
    """The single-null 500-cell chord solve retains its committed state."""
    output_root = Path(os.environ.get("NOVA_EXACT_CLIP_IDENTITY_ROOT", tmp_path))
    output_root.mkdir(parents=True, exist_ok=True)
    driver = output_root / "terminal_driver.py"
    driver.write_text(TERMINAL_DRIVER, encoding="utf-8")
    reference_path = output_root / "single-null-500-base-cpu.npz"
    current_path = output_root / "single-null-500-bounded-current-cpu.npz"
    for checkout, output in (
        (MAIN_CHECKOUT, reference_path),
        (WORKTREE, current_path),
    ):
        _run_terminal_driver(
            checkout,
            driver,
            output,
            "diverted-single-null",
            -500,
            "whole_cell",
        )
    reference = _terminal_arrays(reference_path)
    current = _terminal_arrays(current_path)
    committed = json.loads(WHOLE_CELL_CONTROL.read_text(encoding="utf-8"))
    committed_flux = np.asarray(
        committed["render_data"]["terminal_flux_wb"], dtype=np.float64
    )
    assert current["flux"].size == committed_flux.size
    assert reference.keys() == current.keys()
    for name in reference:
        np.testing.assert_array_equal(current[name], reference[name])
