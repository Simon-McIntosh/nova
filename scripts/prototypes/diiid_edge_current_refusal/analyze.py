"""Read captured trip states and qualify finite boundary-source changes."""

from __future__ import annotations

import json
import hashlib
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from nova.equilibrium.fixed_point import (
    InnerIterationDecision,
    KrylovActionQualification,
)


def write(path, record):
    path.write_text(
        json.dumps(record, indent=2, default=lambda x: np.asarray(x).tolist()) + "\n"
    )


def analyze(capture):
    """Evaluate the production read at observed states, never advance the solve."""
    output = capture.output
    profile, system = capture.profile, capture.system
    operator = profile.operator
    data = dict(np.load(output / "terminal.npz"))
    receipt = json.loads((output / "receipt.json").read_text())
    scale = float(system.flux_scale)
    flux_size = data["state"].size
    coordinates = np.asarray(operator.grid.coordinate)
    node_coordinates = np.vstack((coordinates, np.asarray(operator.wall.coordinate)))
    assert node_coordinates.shape[0] == flux_size
    core = operator.source.core
    target = float(
        json.loads((output / "core-analysis.json").read_text())["target_current_a"]
    )

    @jax.jit
    def read(state):
        masks, topology, connected, admitted = operator._fixed_design_read(state)
        moments = operator.cell_current_moments(state)
        amplitude = operator.current_normalisation_amplitude(
            target, jnp.sum(moments.cell_current)
        )
        return (
            connected,
            admitted,
            topology.axis,
            topology.axis_flux,
            topology.boundary,
            topology.boundary_flux,
            topology.boundary_is_xpoint,
            moments.cell_current,
            operator.residual_shadow_mask(state),
            amplitude,
            masks.profile_participation,
        )

    def reading(state, stem):
        values = jax.device_get(read(jnp.asarray(state)))
        (
            connected,
            admitted,
            axis,
            axis_flux,
            boundary,
            boundary_flux,
            diverted,
            current,
            shadow,
            amplitude,
            participation,
        ) = values
        connected, current, shadow = (
            np.asarray(x).ravel() for x in (connected, current, shadow)
        )
        np.savez_compressed(
            output / f"{stem}.npz",
            connected=connected,
            current=current,
            shadow=shadow,
            participation=participation,
        )
        return (
            dict(
                support=int(connected.sum()),
                admitted=bool(admitted),
                axis_rz_m=axis.tolist(),
                axis_flux_wb=float(axis_flux),
                boundary_rz_m=boundary.tolist(),
                boundary_flux_wb=float(boundary_flux),
                boundary_class="diverted" if diverted else "limited",
                unscaled_enclosed_current_a=float(current.sum()),
                current_amplitude=float(amplitude),
                enclosed_current_a=float(amplitude * current.sum()),
                current_cell_count=int(current.size),
                shadow_count=int(shadow.sum()),
                source_cells=int(np.asarray(participation).sum()),
            ),
            connected,
            current,
            shadow,
        )

    terminal, terminal_support, _, _ = reading(data["state"], "terminal-read")
    assert terminal["support"] == receipt["support"], "terminal read positive control"
    assert len(coordinates) == terminal_support.size, "support coordinate binding"
    edge_p = np.asarray(
        core.pressure_gradient(coordinates[:, 0], np.ones(len(coordinates)))
    )
    edge_ff = np.asarray(core.ff_prime(np.ones(len(coordinates))))
    from scipy.constants import mu_0

    edge_density = (
        -2 * np.pi * (coordinates[:, 0] * edge_p + edge_ff / (mu_0 * coordinates[:, 0]))
    )
    summary = dict(
        terminal=terminal,
        trips=[],
        proposals=[],
        edge_density_min_max_a_m2=[
            float(edge_density.min()),
            float(edge_density.max()),
        ],
        state_units="captured augmented flux / flux_scale; spatial reads in Wb",
        current_units=(
            "unscaled integral of prescribed profiles; "
            "normalised current uses the production amplitude eliminated in each map"
        ),
        analysis_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    )
    changed_union = np.zeros(len(coordinates), bool)
    for index, inner in enumerate(capture.inner):
        arrays = dict(np.load(output / f"inner-{index}.npz"))
        diagnostics = dict(np.load(output / f"diagnostics-{index}.npz"))
        initial_state = arrays["initial"][:flux_size] * scale
        returned_state = arrays["state"][:flux_size] * scale
        initial, before, _, initial_shadow = reading(
            initial_state, f"initial-read-{index}"
        )
        returned, after, _, returned_shadow = reading(
            returned_state, f"returned-read-{index}"
        )
        entering, leaving = after & ~before, before & ~after
        shadow_entering = returned_shadow & ~initial_shadow
        shadow_leaving = initial_shadow & ~returned_shadow
        changed_union |= entering | leaving
        active = diagnostics["inner_iteration_accepted"] >= 0
        iterations = []
        for slot in np.flatnonzero(active):
            row = {
                name: np.asarray(value)[slot].item()
                for name, value in diagnostics.items()
                if name.startswith("inner_iteration_")
            }
            row["decision_name"] = InnerIterationDecision(
                row["inner_iteration_decisions"]
            ).name
            row["krylov_name"] = KrylovActionQualification(
                row["inner_iteration_krylov_qualifications"]
            ).name
            row["levenberg_damping"] = float(
                np.ravel(diagnostics["promotion_model_rebuild_damping"])[slot]
            )
            iterations.append(row)
        summary["trips"].append(
            dict(
                index=index,
                inner=inner,
                initial=initial,
                returned=returned,
                entering_indices=np.flatnonzero(entering).tolist(),
                leaving_indices=np.flatnonzero(leaving).tolist(),
                entering_rz_m=coordinates[entering].tolist(),
                leaving_rz_m=coordinates[leaving].tolist(),
                live_shadow_changes=int(np.sum(initial_shadow != returned_shadow)),
                shadow_entering_rz_m=node_coordinates[shadow_entering].tolist(),
                shadow_leaving_rz_m=node_coordinates[shadow_leaving].tolist(),
                frozen_shadow_count=int(np.sum(diagnostics["frozen_shadow"])),
                iterations=iterations,
            )
        )
        write(output / "spatial.json", summary)
    refused_union = np.zeros(len(coordinates), bool)
    for proposal in capture.proposals:
        arrays = dict(np.load(output / f"proposal-{proposal['index']}.npz"))
        state, step = (
            arrays["state"][:flux_size] * scale,
            arrays["step"][:flux_size] * scale,
        )
        # A proposal is spatially qualified at the full Newton displacement and
        # the smallest scored displacement. The complete ladder stays on disk.
        _, before, _, _ = reading(state, f"proposal-base-{proposal['index']}")
        comparisons = []
        for slot in (0, len(arrays["factors"]) - 1):
            factor = float(arrays["factors"][slot])
            readout, after, _, _ = reading(
                state + factor * step, f"proposal-read-{proposal['index']}-{slot}"
            )
            changed = before != after
            if not proposal["accepted"]:
                refused_union |= changed
            comparisons.append(
                dict(
                    factor=factor,
                    changed_cells=int(changed.sum()),
                    changed_rz_m=coordinates[changed].tolist(),
                    reading=readout,
                    merit=float(arrays["merits"][slot]),
                    predicted_merit=float(arrays["predicted_merits"][slot]),
                    residual=float(arrays["residuals"][slot]),
                )
            )
        summary["proposals"].append(dict(**proposal, comparisons=comparisons))
    np.savez_compressed(
        output / "spatial-marks.npz",
        coordinates=coordinates,
        node_coordinates=node_coordinates,
        changed=changed_union,
        refused=refused_union,
        edge_density=edge_density,
    )
    # A shrinking directional finite difference distinguishes a smooth tangent
    # mismatch from a jump caused by a discrete support transition.
    if capture.proposals:
        last = dict(np.load(output / f"proposal-{capture.proposals[-1]['index']}.npz"))
        point = jnp.asarray(last["state"][:flux_size] * scale)
        direction = jnp.asarray(last["step"][:flux_size] * scale)

        @jax.jit
        def current(state):
            return operator.cell_current_moments(state).cell_current

        @jax.jit
        def tangent(state, vector):
            return jax.jvp(current, (state,), (vector,))

        base, derivative = (
            np.asarray(x) for x in jax.device_get(tangent(point, direction))
        )
        _, initial_support, _, _ = reading(point, "discriminator-base")
        sweep = []
        for factor in (1.0, 0.1, 0.01, 0.001, 1e-4, 1e-5, 1e-6, 1e-7):
            readout, support, measured, _ = reading(
                point + factor * direction, f"discriminator-{factor}"
            )
            delta = measured - base
            remainder = delta - factor * derivative
            sweep.append(
                dict(
                    factor=factor,
                    changed_cells=int(np.sum(support != initial_support)),
                    current_delta_l2_a=float(np.linalg.norm(delta)),
                    tangent_remainder_l2_a=float(np.linalg.norm(remainder)),
                    derivative_error_l2_a=float(np.linalg.norm(remainder) / factor),
                    reading=readout,
                )
            )
        summary["directional_discriminator"] = sweep
    write(output / "spatial.json", summary)
    print(
        "SPATIAL_SUMMARY "
        + json.dumps(
            dict(
                trips=len(summary["trips"]),
                proposals=len(summary["proposals"]),
                changed_cells=int(changed_union.sum()),
                refused_change_cells=int(refused_union.sum()),
                terminal=terminal,
            )
        ),
        flush=True,
    )
