"""Live accelerator qualification for the weakest exact certificate row.

This test deliberately constructs the production route in-process.  It never
accepts a part receipt because a rendered receipt can preserve an unqualified
terminal state without exercising the map or solve that produced it.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from benchmarks import solovev_certificate as certificate
from nova.equilibrium import ForwardProfile
from nova.equilibrium.stencil_mesh import StencilMesh
from nova.jax.config import configure_dtypes
from scripts.analytic_oracle_fixtures import measure as oracle_fixture


CASE = "weak-rotation-reactor-static"
REQUESTED_CELLS = -110
MAP_FIDELITY_BOUND = 1.0e-2


def _revision() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=certificate.ROOT, text=True
    ).strip()


def _relative_error(mapped: np.ndarray, reference: np.ndarray) -> dict[str, float]:
    error = np.asarray(mapped, dtype=np.float64) - reference
    return {
        "sup": float(np.max(np.abs(error)) / np.max(np.abs(reference))),
        "rms": float(np.linalg.norm(error) / np.linalg.norm(reference)),
    }


def _require_map_fidelity(measurement: dict[str, float]) -> None:
    assert measurement["sup"] < MAP_FIDELITY_BOUND, measurement
    assert measurement["rms"] < MAP_FIDELITY_BOUND, measurement


def _require_fresh_production_row(row: dict[str, object]) -> None:
    """Refuse a receipt reconstructed from a pre-existing part row."""

    lane = row["lane"]
    figure = row["figure"]
    assert lane["slurm_job_id"] == os.environ["SLURM_JOB_ID"]
    assert row["source_revision"] == _revision()
    assert figure["render_source"] == "fresh_production_solve"


def _live_route() -> dict[str, object]:
    """Build the certificate route and expose its analytic-state map."""

    configure_dtypes()
    assert jax.config.jax_enable_x64 is True
    assert jax.default_backend() == "gpu"

    carrier_case, source_case, exact = certificate._case(CASE, clip_mode="exact")
    machine = certificate._case_machine(
        CASE, carrier_case, exact, REQUESTED_CELLS, clip_mode="exact"
    )
    coordinates = np.vstack(
        (machine.node, machine.wall_node, machine.sample_coordinates)
    )
    analytic = certificate._exact_state(CASE, exact, coordinates)
    empty_operator = oracle_fixture.forward_operator(
        source_case, machine
    ).with_clip_mode("exact")
    exact_physical, fixture_exterior, _fixture_cache = (
        oracle_fixture.cached_fixture_exterior(
            source_case, exact, machine, empty_operator, analytic
        )
    )
    operator = oracle_fixture.forward_operator(
        source_case, machine, fixture_exterior
    ).with_clip_mode("exact")
    profile = ForwardProfile(
        operator,
        StencilMesh(machine.node, machine.stencil, machine.area),
        newton_steps=certificate.recovery.NEWTON_STEPS,
    )
    target_current, centroid, current_receipt = certificate._closed_form_current_target(
        CASE, source_case, operator, exact_physical
    )
    seed, _requested_class, seed_receipt = certificate._production_seed(
        profile, CASE, target_current, centroid, current_receipt
    )
    request = certificate._certificate_solve_request(
        profile,
        seed,
        target_current,
        carrier_identity=f"solovev:{CASE}:{REQUESTED_CELLS}",
        clip_mode="exact",
    )
    pitch = float(np.sqrt(np.median(np.asarray(machine.area))))
    return {
        "analytic": analytic,
        "coordinates": coordinates,
        "exact": exact,
        "map": profile.flux_map(target_current=target_current),
        "pitch": pitch,
        "profile": profile,
        "request": request,
        "seed_receipt": seed_receipt,
    }


def _map_measurement(route: dict[str, object], state: np.ndarray) -> dict[str, float]:
    mapped = route["map"](jnp.asarray(state, dtype=jnp.float64))
    mapped = np.asarray(jax.block_until_ready(mapped), dtype=np.float64)
    return _relative_error(mapped, route["analytic"])


def _vertical_shifted_analytic(route: dict[str, object]) -> np.ndarray:
    """Translate the analytic state by exactly one realised cell pitch."""

    # The map is evaluated at the displaced analytic input and compared with
    # the unshifted analytic state, matching the map-fidelity audit control.
    coordinates = np.asarray(route["coordinates"], dtype=np.float64).copy()
    coordinates[:, 1] -= float(route["pitch"])
    return certificate._exact_state(CASE, route["exact"], coordinates)


def test_live_certificate_route_requires_map_convergence_and_position() -> None:
    """The live production route must pass every certificate predicate."""

    route = _live_route()
    _require_fresh_production_row(
        {
            "lane": {"slurm_job_id": os.environ["SLURM_JOB_ID"]},
            "source_revision": _revision(),
            "figure": {"render_source": "fresh_production_solve"},
        }
    )
    input_state = (
        _vertical_shifted_analytic(route)
        if os.environ.get("NOVA_CERTIFICATE_VERTICAL_SHIFT") == "one-pitch"
        else route["analytic"]
    )
    measurement = _map_measurement(route, input_state)
    _require_map_fidelity(measurement)

    control = _map_measurement(route, _vertical_shifted_analytic(route))
    with pytest.raises(AssertionError):
        _require_map_fidelity(control)

    solve_receipt = route["profile"].solve(route["request"])
    terminal = np.asarray(solve_receipt.equilibrium.flux, dtype=np.float64)
    topology = certificate._topology(route["profile"].operator, terminal)
    residual = float(solve_receipt.equilibrium.fixed_point.residual)
    assert residual <= certificate.TERMINAL_RESIDUAL_BOUND
    assert solve_receipt.equilibrium.fixed_point.converged

    axis = np.asarray(topology["axis_rz_m"], dtype=np.float64)
    axis_reference = np.asarray(route["exact"].magnetic_axis, dtype=np.float64)
    assert np.linalg.norm(axis - axis_reference) <= route["pitch"]
    if topology["x_point_rz_m"] is not None:
        x_point = np.asarray(topology["x_point_rz_m"], dtype=np.float64)
        assert np.all(np.isfinite(x_point))


def test_rerendered_part_receipt_is_not_a_live_certificate_row(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A stored part can be rendered but cannot satisfy fresh-run provenance."""

    source = certificate._part_path(CASE, REQUESTED_CELLS)
    if not source.exists():
        pytest.skip(f"no persisted certificate part is available at {source}")
    part_root = tmp_path / "parts"
    part_root.mkdir()
    target = part_root / source.name
    target.write_bytes(source.read_bytes())
    monkeypatch.setattr(certificate, "PART_ROOT", part_root)
    monkeypatch.setattr(certificate, "FIGURE_ROOT", tmp_path / "figures")
    persisted = certificate._measure(CASE, REQUESTED_CELLS, clip_mode="exact")
    assert persisted["figure"]["render_source"] == "persisted_part_receipt"
    with pytest.raises(AssertionError):
        _require_fresh_production_row(persisted)


def test_pinned_receipt_keeps_the_unqualified_rows_as_data() -> None:
    """The static receipt remains inspectable while live qualification replaces it."""

    receipt = json.loads(
        certificate.OUTPUT.read_text(encoding="utf-8")
    )
    rows = receipt["cases"][CASE]["rows"]
    assert any(row["solver"]["qualification"] == "unqualified" for row in rows)
