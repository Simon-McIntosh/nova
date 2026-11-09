"""The DIII-D probes select the governed limiter unless a control is explicit."""

from pathlib import Path
from types import SimpleNamespace
import sys

import jax
import numpy as np
import pytest

from nova.jax.config import configure_dtypes

configure_dtypes()
assert jax.config.jax_enable_x64

PROBE = (
    Path(__file__).resolve().parents[1] / "scripts/prototypes/diiid_forward_convergence"
)
sys.path.insert(0, str(PROBE))
sys.path.insert(0, str(PROBE.parent / "diiid_stopping_rule"))

import measure_branches as probe  # noqa: E402
import diagnose_step_gate as diagnosis  # noqa: E402
from nova.equilibrium.forward import FluxLattice  # noqa: E402


@pytest.fixture
def prepared(monkeypatch, tmp_path):
    """Keep the real surface builder while replacing expensive field operands."""
    bench = probe.bench
    lattice = FluxLattice(np.linspace(0.84, 2.54, 33), np.linspace(-1.6, 1.6, 33))
    limiter, _ = bench._physical_wall_ring(
        bench.DEFAULT_MACHINE_ARTIFACT_CACHE, bench.DEFAULT_MACHINE_ARTIFACT_DIGEST
    )
    row = {
        "efit_grid_R": lattice.radius,
        "efit_grid_Z": lattice.height,
        "efit_times": [0.0],
        "efit_lcfs_n": [len(limiter)],
        "efit_lcfs_r": [limiter[:, 0]],
        "efit_lcfs_z": [limiter[:, 1]],
        "efit_r_axis": [1.7],
        "efit_z_axis": [0.0],
    }

    def build(row, frame, expansion):
        wall, material, _, statement, _ = bench._topology_surface(
            row,
            lattice,
            expansion,
            bench.DEFAULT_MACHINE_ARTIFACT_CACHE,
            bench.DEFAULT_MACHINE_ARTIFACT_DIGEST,
        )
        profile = SimpleNamespace(
            lattice=lattice,
            operator=SimpleNamespace(inside_material=material),
            cold_seed_portfolio=lambda *args, **kwargs: None,
        )
        return (
            profile,
            np.zeros(lattice.node_count + len(wall)),
            np.zeros(lattice.shape),
            wall,
            19,
            statement,
        )

    monkeypatch.setattr(bench, "DEFAULT_DATA", tmp_path)
    monkeypatch.setattr(
        bench,
        "select_frames",
        lambda *args: [SimpleNamespace(frame=0, path=tmp_path / "shot.parquet")],
    )
    monkeypatch.setattr(bench, "polarity_population", lambda: None)
    monkeypatch.setattr(bench, "_read", lambda *args: dict(row))
    monkeypatch.setattr(bench, "build_profile", build)
    monkeypatch.setattr(
        bench,
        "dataset_machine_description",
        lambda *args, **kwargs: SimpleNamespace(physical=None),
    )
    monkeypatch.setattr(bench, "shipped_current_at", lambda *args: np.ones(1))
    monkeypatch.setattr(
        bench,
        "complete_profile_current_adapter",
        lambda profile, **kwargs: SimpleNamespace(
            profile=profile,
            resolution=SimpleNamespace(current=lambda shape: np.ones(1)),
        ),
    )
    monkeypatch.setattr(bench, "_target_current", lambda *args: 1.0)
    physical = bench._topology_surface(
        row,
        lattice,
        None,
        bench.DEFAULT_MACHINE_ARTIFACT_CACHE,
        bench.DEFAULT_MACHINE_ARTIFACT_DIGEST,
    )[1]
    return lattice, physical


def test_prepared_frame_excludes_cells_outside_governed_limiter(prepared):
    _, physical = prepared
    material = np.asarray(probe._prepare_frame(0)["profile"].operator.inside_material)
    assert physical.any() and (~physical).any()
    assert not material[~physical].any()


def test_prepared_frame_material_count_equals_physical_route(prepared):
    _, physical = prepared
    material = np.asarray(probe._prepare_frame(0)["profile"].operator.inside_material)
    assert int(physical.sum()) == 527
    assert int(material.sum()) == int(physical.sum())


def test_pseudo_wall_requires_explicit_selection(prepared):
    material = probe._prepare_frame(0, 0.02)["profile"].operator.inside_material
    assert np.asarray(material).all()


@pytest.mark.parametrize("expansion", [None, 0.02])
def test_stopping_rule_forwards_the_explicit_surface_choice(monkeypatch, expansion):
    class Prepared(Exception):
        pass

    def capture(frame, selected):
        assert frame == 179 and selected == expansion
        raise Prepared

    argv = ["diagnose_step_gate.py", "--frame", "179", "--branch", "limited"]
    if expansion is not None:
        argv.extend(["--pseudo-wall-expansion", str(expansion)])
    monkeypatch.setattr(sys, "argv", argv)
    monkeypatch.setattr(diagnosis, "_instrument", lambda: None)
    monkeypatch.setattr(probe, "_prepare_frame", capture)
    with pytest.raises(Prepared):
        diagnosis.main()


def test_probe_serialization_and_counter_checks():
    probe._self_check()
    assert probe._counters_self_check() == 0


@pytest.mark.parametrize("expansion", [None, 0.02])
def test_convergence_cli_requires_an_explicit_pseudo_wall(
    monkeypatch, tmp_path, expansion
):
    class Prepared(Exception):
        pass

    def capture(frame, selected, *args):
        assert frame == 179 and selected == expansion
        raise Prepared

    argv = [
        "measure_branches.py",
        "--frames",
        "179",
        "--output",
        str(tmp_path / "out.json"),
        "--artifact-dir",
        str(tmp_path / "arrays"),
    ]
    if expansion is not None:
        argv.extend(["--pseudo-wall-expansion", str(expansion)])
    monkeypatch.setattr(sys, "argv", argv)
    monkeypatch.setattr(probe, "run_frame", capture)
    with pytest.raises(Prepared):
        probe.main()
