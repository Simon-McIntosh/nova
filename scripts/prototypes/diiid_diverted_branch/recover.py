"""Recover a completed baseline solve from saved operands, without advancing it."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import shlex
import shutil
import subprocess
import sys


def events(text, prefix):
    return [
        json.loads(line[len(prefix) :])
        for line in text.splitlines()
        if line.startswith(prefix)
    ]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--revision", required=True)
    args = parser.parse_args()
    import jax
    import jax.numpy as jnp
    import numpy as np

    from nova.equilibrium import fixed_point
    from nova.equilibrium.forward import ForwardProfile
    from nova.equilibrium.flux_surface_connectivity import polish_stationary_points
    from nova.jax.config import configure_dtypes
    from nova.linalg.tensor_spline import fit_tensor_spline

    configure_dtypes()
    assert jax.config.jax_enable_x64
    repository = Path(__file__).resolve().parents[3]
    changed = subprocess.check_output(
        [
            "git",
            "-C",
            str(repository),
            "diff",
            "--name-only",
            args.revision,
            "HEAD",
            "--",
            "nova",
        ],
        text=True,
    ).splitlines()
    assert set(changed) <= {
        "nova/equilibrium/fixed_point.py",
        "nova/equilibrium/constraint.py",
    }
    original_fixed = subprocess.check_output(
        [
            "git",
            "-C",
            str(repository),
            "show",
            args.revision + ":nova/equilibrium/fixed_point.py",
        ],
        text=True,
    )
    assert "warmup_refused_proposals" in original_fixed
    assert "newton_branch_refused_proposals" not in original_fixed

    def no_solve(*_a, **_kw):
        raise AssertionError("a saved-state recovery must not advance a forward solve")

    ForwardProfile.solve = no_solve
    fixed_point.newton_krylov = no_solve
    try:
        ForwardProfile.solve(None)
    except AssertionError as error:
        assert "must not advance" in str(error)
        print("RECOVERY_SOLVER_GUARD_REFUSAL=" + str(error), flush=True)
    else:
        raise AssertionError("recovery solver guard did not refuse the control")
    spec = importlib.util.spec_from_file_location(
        "branch_measure", Path(__file__).with_name("measure.py")
    )
    measure = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = measure
    spec.loader.exec_module(measure)
    material = measure.original_load(
        "recovery_material",
        repository / "scripts/prototypes/diiid_material_mask/measure.py",
    )
    prepared = material.probe._prepare_frame(137, None)
    profile = prepared["profile"]
    assert int(np.asarray(profile.operator.inside_material).sum()) == 527
    text = (args.source / "row.log").read_text()
    assert text.rstrip().endswith("EXIT=1")
    seeds = events(text, "SEED_ADMISSION ")
    timing = events(text, "SOLVE_TIMING ")
    request = events(text, "REQUEST ")
    assert len(seeds) == len(timing) == 1
    assert len(request) == 1 and request[0]["route"] == "newton_krylov"
    assert "own_mask_acceptance=True" in request[0]["policy"]
    assert "settled_exit=True" in request[0]["policy"]
    assert (
        hashlib.sha256(prepared["label_seed"].tobytes()).hexdigest()
        == seeds[0]["sha256"]
    )
    inner = json.loads((args.source / "inner.json").read_text())
    assert len(inner) == 3
    with np.load(args.source / "terminal.npz") as archive:
        data = {key: archive[key] for key in archive.files}
    count = data["state"].size
    saved = []
    for number in range(len(inner)):
        with np.load(args.source / f"inner-{number}.npz") as archive:
            saved.append({key: archive[key] for key in archive.files})
    last = saved[-1]
    scale = float(
        np.dot(data["state"], last["state"][:count])
        / np.dot(last["state"][:count], last["state"][:count])
    )
    state_error = float(np.max(np.abs(data["state"] - scale * last["state"][:count])))
    image_error = float(np.max(np.abs(data["mapped"] - scale * last["mapped"][:count])))
    assert max(state_error, image_error) < 1e-12
    state_hash = hashlib.sha256(data["state"].tobytes()).hexdigest()

    mask_read = jax.jit(
        lambda state, previous: profile.operator.residual_shadow_mask(
            state, previous_shadow=previous
        )
    )
    mask = profile.operator.residual_shadow_mask(
        jnp.asarray(saved[0]["initial"][:count] * scale)
    )
    differences = []
    for number, values in enumerate(saved):
        selected = (
            saved[number + 1]["initial"][:count] * scale
            if number + 1 < len(saved)
            else data["state"]
        )
        next_mask = mask_read(jnp.asarray(selected), mask)
        differences.append(int(jnp.sum(next_mask != mask)))
        mask = next_mask
    np.testing.assert_array_equal(mask, data["shadow"])
    assert differences[-1] == 0
    assert np.array_equal(last["state"], last["initial"])
    assert inner[-1]["accepted"] == 0 and inner[-1]["attempted"] > 0
    residual = float(
        np.max(np.abs(last["mapped"] - last["state"]))
        / max(np.max(np.abs(last["mapped"])), 1e-30)
    )
    assert np.isfinite(residual) and residual > 1e-8

    warmups = [[] for _ in inner]
    trip = 0
    for line in text.splitlines():
        if line.startswith("INNER "):
            trip += 1
        elif line.startswith("STAGE "):
            event = json.loads(line[6:])
            if event["stage"] == "warmup":
                warmups[trip].append(event)
    history = []
    checked_retention = 0
    for rows in warmups:
        assert len(rows) % 2 == 0
        proposals = rows[1::2]
        refused = [
            i
            for i, r in enumerate(proposals)
            if not (
                r["finite_state"] and r["finite_image"] and np.isfinite(r["residual"])
            )
        ]
        nonfinite = [i for i, r in enumerate(proposals) if not r["finite_image"]]
        for number in refused:
            if number + 1 < len(proposals):
                before, after = rows[2 * number], rows[2 * number + 2]
                with (
                    np.load(args.source / f"warmup-{before['index']}.npz") as b,
                    np.load(args.source / f"warmup-{after['index']}.npz") as a,
                ):
                    np.testing.assert_array_equal(b["state"], a["state"])
                checked_retention += 1
        history.append(
            dict(
                refused=len(refused),
                first_nonfinite_index=nonfinite[0] if nonfinite else -1,
            )
        )

    field = fit_tensor_spline(
        jnp.asarray(data["radius"]),
        jnp.asarray(data["height"]),
        jnp.asarray(data["label"].T),
    )
    reference = polish_stationary_points(
        field,
        jnp.asarray(np.vstack((data["reference_axis"], data["reference_saddle"]))),
        jnp.ones(2, bool),
    )
    assert np.asarray(reference["converged"]).all()
    lower = np.asarray(reference["position_rz"])[1]
    branches = [
        dict(
            warmup_branch_refused_proposals=[0],
            warmup_first_branch_refusal_index=[-1],
            newton_branch_refused_proposals=[0],
            newton_first_branch_refusal_index=[-1],
        )
        for _ in inner
    ]
    record = dict(
        machine="diiid",
        frame=137,
        branch="diverted",
        base_revision=args.revision,
        route="newton_krylov",
        storage_order="radius_height",
        residual=residual,
        converged=False,
        terminal_reason=int(
            fixed_point.FixedPointTerminationReason.ACTIVE_SET_STEP_REFUSED
        ),
        termination_reason_reconstructed=True,
        terminal_reason_basis=(
            "unchanged finite terminal state, unchanged carried mask, above tolerance, "
            "and 24 attempted/zero accepted Newton promotions on the unfrozen "
            "own-mask route"
        ),
        support=int(data["connected"].sum()),
        attempted=sum(r["attempted"] for r in inner),
        accepted=sum(r["accepted"] for r in inner),
        inner=inner,
        warmup=history,
        warmup_refused_proposals=sum(r["refused"] for r in history),
        efit_x_point_rz_m=lower.tolist(),
        x_point_distance_m=float(np.linalg.norm(data["saddle"][:2] - lower)),
        changed_mask_entries=differences,
        changed_mask_entries_reconstructed=True,
        residual_history=[r["residual"] for r in inner],
        residual_history_kind=(
            "saved inner best residuals; original outer receipt was not persisted"
        ),
        material_cells=527,
        timing=timing,
        augmented_topology_reader_connected=False,
        module=str(repository / "nova/equilibrium/forward_operator.py"),
        cwd=str(Path.cwd().resolve()),
        recovery=dict(
            source_path=str(args.source),
            original_reporting_exit_status=1,
            forward_solves_executed=0,
            state_sha256=state_hash,
            state_scale_error=state_error,
            image_scale_error=image_error,
            finite_refusal_retention_controls=checked_retention,
            receipt_fields_reconstructed=[
                "terminal_reason",
                "changed_mask_entries",
                "warmup",
            ],
            observation=(
                "saved completed solve; post hoc topology and receipt reconstruction"
            ),
        ),
    )
    args.output.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(args.source / "terminal.npz", args.output / "terminal.npz")
    for number in range(len(inner)):
        (args.output / f"inner-{number}.npz").symlink_to(
            args.source / f"inner-{number}.npz"
        )
    (args.output / "original-row.log").symlink_to(args.source / "row.log")
    measure.enrich(args.output, profile, scale, record, data, branches, seeds)
    with np.load(args.output / "terminal.npz") as archive:
        assert hashlib.sha256(archive["state"].tobytes()).hexdigest() == state_hash
    command = shlex.join([sys.executable, *sys.argv])
    (args.output / "row.log").write_text(
        f"# revision: {args.revision}; tree: {repository}; command: {command}\n"
        f"# module: {record['module']}; cwd: {Path.cwd().resolve()}\n"
        f"RECOVERED_SAVED_STATE_SHA256={state_hash}\nFORWARD_SOLVES_EXECUTED=0\n"
        f"ORIGINAL_REPORTING_LOG={args.source / 'row.log'}\nEXIT=0\n"
    )
    print("RECOVERY_COMPLETE " + json.dumps(record["recovery"]), flush=True)


if __name__ == "__main__":
    main()
