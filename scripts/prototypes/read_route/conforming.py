"""Measure Bell reads of booked states with durable per-arm receipts."""

# ruff: noqa: E402
from nova.jax.config import configure_dtypes

configure_dtypes()

import json
import os
import subprocess
import sys
from pathlib import Path
import jax
import jax.numpy as jnp
import numpy as np
import nova

from benchmarks import solovev_certificate as certificate
from nova.equilibrium import current, topology
from nova.equilibrium.clip_quadrature import ClippedCurrentMoments
from nova.equilibrium.solve_request import TopologyPolicy
from scripts.analytic_oracle_fixtures.measure import analytic_profile
from scripts.prototypes.read_route.bell import (
    field_from_jets,
    read_bell,
    read_cached_bell,
)
from scripts.prototypes.read_route.check_element import controls
from scripts.prototypes.read_route.jets import build_jet_matrix
from scripts.prototypes.read_route.measure import read_row, jet_reader, sample_field
from tests.equilibrium import test_topology_read as fixture

assert jax.config.jax_enable_x64


def run(args):
    print(f"NOVA_MODULE={nova.__file__} CWD={Path.cwd().resolve()}", flush=True)
    assert jax.default_backend() == "gpu"
    oracle, analytic, wall, axis, saddle = fixture._analytic_inputs(args.kind)
    geometry = fixture._realised_hex_geometry(wall, args.cells)
    convention = topology.TopologyConvention.from_cocos(17, 1.0)
    policy = TopologyPolicy(
        normal_form_radius=0.054806712567833996 if saddle is not None else 0.0
    )
    reference_fraction, uncertainty = fixture._reference_cell_fractions(
        args.kind, oracle, geometry
    )
    scoring = (
        geometry,
        convention,
        policy,
        reference_fraction,
        uncertainty,
        axis,
        saddle,
    )
    receipt = dict(kind=args.kind, cells=args.cells, rows=[])
    args.out.parent.mkdir(parents=True, exist_ok=True)

    def persist():
        pending = args.out.with_suffix(".pending")
        pending.write_text(json.dumps(receipt, indent=2) + "\n")
        pending.replace(args.out)

    def measure(field, name, reader=topology.read):
        row, result = read_row(field, *scoring, name, reader=reader)
        bound = (0.274714 if saddle is None else 2.1805365680219544) * (
            row["pitch_m"] / (6.2 if saddle is None else 1.7)
        ) ** 2
        row["membership_bound"] = bound
        row["passes_membership"] = (
            row["valid"]
            and row["qualified"]
            and max(row["smooth_membership_error"], row["saddle_membership_error"] or 0)
            <= bound
        )
        receipt["rows"].append(row)
        persist()
        np.savez(
            args.out.with_name(args.out.stem + "-" + name + ".npz"),
            membership=np.asarray(result.membership),
            axis=np.asarray(result.axis),
            x_points=np.asarray(result.x_points),
            x_point_valid=np.asarray(result.x_point_valid),
        )
        return row, result

    analytic_row, analytic_read = measure(analytic, "closed_form")
    if args.cells in (2000, 5000):
        golden = np.load(
            Path(__file__).resolve().parents[1]
            / "ladder"
            / "compatibility"
            / f"{args.kind}-{args.cells}.npz"
        )
        delta = float(
            np.max(np.abs(np.asarray(analytic_read.membership) - golden["membership"]))
        )
        print(f"LADDER_CONTROL membership_delta={delta:.12g}", flush=True)
        assert delta < 1e-9
    control, nodes, indices, mesh = controls(geometry, analytic)
    receipt["element_controls"] = control
    persist()
    _, source, _ = certificate._case(
        "weak-rotation-reactor-static"
        if args.kind == "limited"
        else "diverted-single-null",
        clip_mode="exact",
    )
    profile = analytic_profile(source)
    psi = jax.vmap(jax.vmap(analytic.value))(geometry.sample_points)
    quadrature = current.Quadrature(sigma=convention.sigma, segments=64)
    booking = current.book(psi, analytic_read, geometry, profile, quadrature)
    jax.block_until_ready(booking)
    assert bool(booking.valid)
    moments = ClippedCurrentMoments(*booking[:3])
    reference = fixture._kernel_backed_field(args.kind, oracle, analytic, geometry)
    matrix, exterior, built_nodes, built_indices, wall = build_jet_matrix(
        reference.coupling, geometry, analytic, reference.moments, persist=True
    )
    assert np.array_equal(nodes, np.asarray(built_nodes)) and np.array_equal(
        indices, np.asarray(built_indices)
    )
    receipt["jet_build_seconds"] = wall
    receipt["jet_bytes"] = matrix.size * matrix.dtype.itemsize
    receipt["jet_shape"] = list(matrix.shape)
    persist()
    jets = exterior + jnp.einsum("ncmq,cm->nq", matrix, jnp.stack(tuple(moments), -1))
    # The nonconforming read is measured before the replacement on this state.
    if args.kind == "diverted" and args.cells in (550, 2000):
        row, _ = measure(
            (matrix, moments, exterior, built_nodes, built_indices),
            "nonconforming_jet",
            jet_reader,
        )
        if args.cells == 2000:
            assert abs(row["saddle_membership_error"] - 0.221159) < 2e-5
            print(
                "NONCONFORMING_REPRODUCED saddle="
                + str(row["saddle_membership_error"]),
                flush=True,
            )
    payload = (matrix, moments, exterior, built_indices, mesh)
    row, result = measure(payload, "bell_jet", read_cached_bell)
    if args.kind == "diverted" and args.cells in (550, 2000):
        mutated = field_from_jets(jets.at[:, 3:].set(0), built_indices, mesh)
        negative, _ = measure(mutated, "bell_zero_hessian", read_bell)
        # Emit the actual guarded assertion as a separately readable red log.
        negative_log = args.out.with_name(args.out.stem + "-negative.log")
        negative_row = args.out.with_name(args.out.stem + "-negative.json")
        negative_row.write_text(json.dumps(negative))
        command = [
            sys.executable,
            "-m",
            "pytest",
            "-p",
            "no:cacheprovider",
            str(Path(__file__).with_name("check_membership.py")),
            "-q",
        ]
        with negative_log.open("w") as log:
            log.write("COMMAND=" + " ".join(command) + "\n")
            log.flush()
            outcome = subprocess.run(
                command,
                env=dict(os.environ, BELL_MEMBERSHIP_ROW=str(negative_row)),
                stdout=log,
                stderr=subprocess.STDOUT,
            )
            log.write(f"EXIT={outcome.returncode}\n")
        assert outcome.returncode == 1, "zero Hessian control did not fail membership"
    if args.kind == "diverted" and args.cells in (2000, 5000):
        sampled = sample_field(analytic, geometry)
        sampled_read = jax.jit(topology.read)(sampled, geometry, convention, policy)
        trial = current.book(psi, sampled_read, geometry, profile, quadrature)
        jax.block_until_ready(trial)
        assert bool(trial.valid)
        trial_moments = ClippedCurrentMoments(*trial[:3])
        picard, _ = measure(
            (matrix, trial_moments, exterior, built_indices, mesh),
            "picard_bell",
            read_cached_bell,
        )
        receipt["picard_reference_note"] = (
            "Read against analytic fraction in this row; cached live-kernel scalar "
            "receipt is a separate comparison, not a pointwise membership reference."
        )
        persist()
    print("ROW_COMPLETE " + str(args.out), flush=True)
