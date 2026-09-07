"""Identity checks for the host-route corpus scheduler."""

from __future__ import annotations

from copy import deepcopy
import os
from types import SimpleNamespace

import numpy as np

from apps.playable.solovev import build_machine
from nova.equilibrium.observation import MomentIntegralSupport
from nova.equilibrium.topology import NoQualifiedAxisError, TopologyClass
from scripts.labeller_batch import shard
from scripts.labeller_batch.shard import LabellerPrograms, PreparedLabeller
from scripts.labeller_parallel import driver, scheduler


def _without_elapsed(record: dict[str, object]) -> dict[str, object]:
    stable = deepcopy(record)
    for name in ("wall_seconds", "free_wall_seconds", "conditioned_wall_seconds"):
        stable[name] = 0.0
    return stable


def test_host_route_matches_shard_slice_record_on_solovev(monkeypatch, tmp_path):
    """One real free and conditioned solve yields the shard's slice record."""
    machine = build_machine()
    profile = machine.profile
    seed = np.asarray(machine.seed)
    current = np.asarray(profile.operator.prescribed_current_field.current)
    target_current = abs(float(np.sum(np.asarray(profile.operator.cell_current(seed)))))
    observation = profile.current_moment_observation(
        seed,
        support=MomentIntegralSupport.ALL_DOMAIN,
        target_current=target_current,
    )
    target_centroid_z = float(np.asarray(observation.centroid_z)) + 0.1
    requested_class = int(TopologyClass.LIMITED)
    policy_evidence = {
        "active_mapping": [
            {"stored_circuit": index + 1, "family": f"circuit_{index}"}
            for index in range(current.size)
        ]
    }
    prepared = PreparedLabeller(
        profile=profile,
        wall=np.asarray(machine.wall),
        carrier_evidence={},
        policy_evidence=policy_evidence,
        cache_directory=str(tmp_path),
        setup_wall_seconds=0.0,
    )
    inputs = {
        "time": 0.25,
        "magnetic_axis_z": 0.0,
        "target_centroid_z": target_centroid_z,
        "reference_plasma_current": target_current,
        "current": current,
    }
    group = {
        "time": np.asarray([inputs["time"]]),
        "gridr": np.asarray([0.0]),
        "gridz": np.asarray([0.0]),
        "psi_norm": np.asarray([0.0, 1.0]),
        "pprime": np.asarray([[0.0, 0.0]]),
        "ffprime": np.asarray([[0.0, 0.0]]),
    }

    monkeypatch.setattr(
        shard.zarr, "open_group", lambda *_args, **_kwargs: {"efm": group}
    )
    monkeypatch.setattr(shard, "_slice_inputs", lambda *_args: inputs)
    monkeypatch.setattr(shard, "_slices_seed", lambda *_args: seed)
    monkeypatch.setattr(shard, "_requested_class", lambda *_args: requested_class)

    def receipt(_prepared, result, **_arguments):
        return SimpleNamespace(
            qualified=bool(result.converged), terminal_state=object()
        )

    def frame(_receipt, **_arguments):
        return SimpleNamespace(branch_guard_ok=True)

    for module in (shard, scheduler):
        monkeypatch.setattr(module, "_forward_receipt", receipt)
        monkeypatch.setattr(
            module, "_internal_geometry", lambda *_args, **_kwargs: None
        )
        monkeypatch.setattr(module, "assemble_frame", frame)
    monkeypatch.setattr(shard, "_write_session_file", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(shard, "_write_companion", lambda *_args, **_kwargs: None)

    shard_output = tmp_path / "shard"
    shard_output.mkdir()
    _programs, manifest = shard.label_shot(
        prepared,
        1,
        shard_output,
        programs=LabellerPrograms(),
        include_raster=False,
        condition_on_guard_failure=True,
        setup_wall_seconds=0.0,
        max_slices=1,
    )

    item = scheduler.SliceInput(
        shot=1,
        row=0,
        time=float(inputs["time"]),
        initial_state=seed,
        prescribed_current=current,
        target_current=target_current,
        requested_class=requested_class,
        centroid_target_z=target_centroid_z,
        p_prime_psi_norm=np.asarray(group["psi_norm"]),
        p_prime=np.asarray(group["pprime"])[0],
        ff_prime_psi_norm=np.asarray(group["psi_norm"]),
        ff_prime=np.asarray(group["ffprime"])[0],
    )
    batch = scheduler.EngineBatch(
        active=np.asarray([True]),
        shot=np.asarray([item.shot]),
        row=np.asarray([item.row]),
        time=np.asarray([item.time]),
        initial_state=np.asarray([item.initial_state]),
        prescribed_current=np.asarray([item.prescribed_current]),
        target_current=np.asarray([item.target_current]),
        requested_class=np.asarray([item.requested_class], dtype=np.int8),
        centroid_target_z=np.asarray([item.centroid_target_z]),
    )
    engine = scheduler.HostRouteEngine(
        prepared,
        device_count=1,
        condition_on_guard_failure=True,
    )
    engine._ensure_slots(1)
    solved = engine._solve_slot(batch, 0)
    monkeypatch.setattr(scheduler, "_ASSEMBLY_PREPARED", prepared)
    monkeypatch.setattr(scheduler, "_CONDITION_ON_GUARD_FAILURE", True)
    assembled = scheduler.assemble_frame_on_host(
        scheduler.AssemblyRequest(item, solved)
    )

    expected = dict(manifest["slices"][0])
    expected["requested_class"] = requested_class
    assert expected["conditioned"] is True
    assert assembled.record["conditioned"] is True
    assert _without_elapsed(assembled.record) == _without_elapsed(expected)


def test_host_worker_default_tracks_the_allocation(monkeypatch):
    monkeypatch.setenv("SLURM_CPUS_PER_TASK", "8")
    assert driver._default_host_workers() == 7


def test_host_worker_default_falls_back_to_cpu_count(monkeypatch):
    monkeypatch.delenv("SLURM_CPUS_PER_TASK", raising=False)
    monkeypatch.setattr(os, "cpu_count", lambda: 12)
    assert driver._default_host_workers() == 11


def test_explicit_host_workers_overrides_the_default(monkeypatch):
    monkeypatch.setenv("SLURM_CPUS_PER_TASK", "8")
    assert driver.resolve_host_workers(4) == 4
    assert driver.resolve_host_workers(None) == 7


# --------------------------------------------------------------------------
# conditioning-outcome recording: seed admission versus constrained solve
# --------------------------------------------------------------------------


def _solovev_fixture(tmp_path) -> SimpleNamespace:
    """Build the shared Solov'ev prepared labeller and one slice's inputs."""
    machine = build_machine()
    profile = machine.profile
    seed = np.asarray(machine.seed)
    current = np.asarray(profile.operator.prescribed_current_field.current)
    target_current = abs(float(np.sum(np.asarray(profile.operator.cell_current(seed)))))
    observation = profile.current_moment_observation(
        seed,
        support=MomentIntegralSupport.ALL_DOMAIN,
        target_current=target_current,
    )
    target_centroid_z = float(np.asarray(observation.centroid_z)) + 0.1
    requested_class = int(TopologyClass.LIMITED)
    policy_evidence = {
        "active_mapping": [
            {"stored_circuit": index + 1, "family": f"circuit_{index}"}
            for index in range(current.size)
        ]
    }
    prepared = PreparedLabeller(
        profile=profile,
        wall=np.asarray(machine.wall),
        carrier_evidence={},
        policy_evidence=policy_evidence,
        cache_directory=str(tmp_path),
        setup_wall_seconds=0.0,
    )
    inputs = {
        "time": 0.25,
        "magnetic_axis_z": 0.0,
        "target_centroid_z": target_centroid_z,
        "reference_plasma_current": target_current,
        "current": current,
    }
    group = {
        "time": np.asarray([inputs["time"]]),
        "gridr": np.asarray([0.0]),
        "gridz": np.asarray([0.0]),
        "psi_norm": np.asarray([0.0, 1.0]),
        "pprime": np.asarray([[0.0, 0.0]]),
        "ffprime": np.asarray([[0.0, 0.0]]),
    }
    return SimpleNamespace(
        prepared=prepared,
        seed=seed,
        current=current,
        target_current=target_current,
        target_centroid_z=target_centroid_z,
        requested_class=requested_class,
        inputs=inputs,
        group=group,
    )


def _raising(error: Exception):
    """Return a callable that always raises ``error`` (the injected failure)."""

    def _raiser(*_args, **_kwargs):
        raise error

    return _raiser


def _record_from_shard(monkeypatch, tmp_path, *, admission, solve) -> dict:
    """Run one shard slice with injected conditioning-stage failures.

    The free solve is forced to fail so a guarded re-solve is requested;
    ``admission`` replaces the centroid-pair derivation and ``solve`` the
    constrained solve.  The single written slice record is returned.
    """
    fixture = _solovev_fixture(tmp_path)
    monkeypatch.setattr(
        shard.zarr, "open_group", lambda *_args, **_kwargs: {"efm": fixture.group}
    )
    monkeypatch.setattr(shard, "_slice_inputs", lambda *_args: fixture.inputs)
    monkeypatch.setattr(shard, "_slices_seed", lambda *_args: fixture.seed)
    monkeypatch.setattr(
        shard, "_requested_class", lambda *_args: fixture.requested_class
    )
    monkeypatch.setattr(
        shard.reduced_newton,
        "solve_reduced_newton",
        _raising(RuntimeError("injected free-solve failure")),
    )
    monkeypatch.setattr(shard, "_centroid_pair", admission)
    monkeypatch.setattr(shard.reduced_newton, "solve_constrained_reduced_newton", solve)
    monkeypatch.setattr(shard, "_write_session_file", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(shard, "_write_companion", lambda *_args, **_kwargs: None)

    output = tmp_path / "shard-records"
    output.mkdir()
    _programs, manifest = shard.label_shot(
        fixture.prepared,
        1,
        output,
        programs=LabellerPrograms(),
        include_raster=False,
        condition_on_guard_failure=True,
        setup_wall_seconds=0.0,
        max_slices=1,
    )
    return dict(manifest["slices"][0])


def _record_from_scheduler(monkeypatch, tmp_path, *, admission, solve) -> dict:
    """Run one scheduler slot with injected conditioning-stage failures.

    Mirrors ``_record_from_shard`` on the parallel writer's ``_solve_slot``.
    """
    fixture = _solovev_fixture(tmp_path)
    item = scheduler.SliceInput(
        shot=1,
        row=0,
        time=float(fixture.inputs["time"]),
        initial_state=fixture.seed,
        prescribed_current=fixture.current,
        target_current=fixture.target_current,
        requested_class=fixture.requested_class,
        centroid_target_z=fixture.target_centroid_z,
        p_prime_psi_norm=np.asarray(fixture.group["psi_norm"]),
        p_prime=np.asarray(fixture.group["pprime"])[0],
        ff_prime_psi_norm=np.asarray(fixture.group["psi_norm"]),
        ff_prime=np.asarray(fixture.group["ffprime"])[0],
    )
    batch = scheduler.EngineBatch(
        active=np.asarray([True]),
        shot=np.asarray([item.shot]),
        row=np.asarray([item.row]),
        time=np.asarray([item.time]),
        initial_state=np.asarray([item.initial_state]),
        prescribed_current=np.asarray([item.prescribed_current]),
        target_current=np.asarray([item.target_current]),
        requested_class=np.asarray([item.requested_class], dtype=np.int8),
        centroid_target_z=np.asarray([item.centroid_target_z]),
    )
    engine = scheduler.HostRouteEngine(
        fixture.prepared,
        device_count=1,
        condition_on_guard_failure=True,
    )
    engine._ensure_slots(1)
    monkeypatch.setattr(
        scheduler.HostRouteEngine,
        "_free_solve",
        _raising(RuntimeError("injected free-solve failure")),
    )
    monkeypatch.setattr(scheduler, "_centroid_pair", admission)
    monkeypatch.setattr(scheduler.HostRouteEngine, "_conditioned_solve", solve)
    solved = engine._solve_slot(batch, 0)
    return dict(solved.record)


def test_seed_admission_failure_is_recorded_as_skipped(monkeypatch, tmp_path):
    """A centroid-pair NoQualifiedAxisError is its own skipped outcome.

    Both writers must record the reason under ``conditioning_skipped`` with
    ``conditioned`` false, never as a conditioned-solve failure.
    """
    message = "no qualified magnetic-axis candidate has a resolved component"
    expected = f"NoQualifiedAxisError: {message}"
    admission = _raising(NoQualifiedAxisError(message))
    solve = _raising(RuntimeError("constrained solve must not run"))
    records = [
        _record_from_shard(monkeypatch, tmp_path, admission=admission, solve=solve),
        _record_from_scheduler(monkeypatch, tmp_path, admission=admission, solve=solve),
    ]
    for record in records:
        assert record["conditioned"] is False
        assert record["conditioning_skipped"] == expected
        assert "conditioning_exception" not in record
        assert record["conditioned_converged"] is None
        assert record["converged"] is False


def test_failure_after_conditioned_solve_begins_stays_conditioned(
    monkeypatch, tmp_path
):
    """A raise after the constrained solve starts is a conditioned failure.

    Once the centroid pair is derived the slice is committed to the
    conditioned route, so the injected post-admission failure must keep
    ``conditioned`` true and never be reclassified as skipped.
    """

    def admission(*_args, **_kwargs):
        return (("dummy-pair",), None)

    solve = _raising(RuntimeError("injected post-admission solve failure"))
    records = [
        _record_from_shard(monkeypatch, tmp_path, admission=admission, solve=solve),
        _record_from_scheduler(monkeypatch, tmp_path, admission=admission, solve=solve),
    ]
    for record in records:
        assert record["conditioned"] is True
        assert record["conditioning_exception"] == (
            "RuntimeError: injected post-admission solve failure"
        )
        assert "conditioning_skipped" not in record
        assert record["conditioned_converged"] is None
