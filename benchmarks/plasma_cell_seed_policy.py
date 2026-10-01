"""Compare two seed policies on the exact-clip certificate route.

The certificate production route is measured at the reduced rung for both
declared cases under two seed policies: the explicit analytic flux sample
the route already carries, and the current-centroid disc seed the last
converging receipt was launched from.  Every solve is the certificate's
own construction -- its own operator, closed-form current target and
solve request.  Per arm the receipt records the seed policy, the seed
state digest, the converged flag, the trip count, the per-trip residual
history, the terminal residual, the terminal and reference saddle with
their distance in metres, and the count of non-finite clipped-support
current moments at the terminal state.

Each case's row also carries the flux range of every seed against the
analytic seed's own range, because seed amplitude is what separates the
policies: the route's saddle-anchored seed is several times the analytic
seed's flux extent, so the two are different initial states rather than
two spellings of one.

--complete-receipt splices a receipt that already carries a case: it
measures the declared one-trip control, records the seed flux ranges, and
measures only the cases the receipt lacks, leaving a case's arms as the
documented comparison state.  A crash in a later part of a measurement
therefore does not require the earlier arms to be measured again.

The per-case verdict and the receipt-wide verdict field are derived from
the arms the receipt carries, on every write and on demand, so a row whose
arms landed before a crash cannot disagree with the answer the receipt
reports for it.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
import hashlib
import json
import os
import subprocess
from time import perf_counter

import jax
import jax.numpy as jnp
import numpy as np

from benchmarks import solovev_certificate as certificate
from nova.equilibrium.forward import (
    ColdSeedConstruction,
    ForwardProfile,
)
from nova.equilibrium.stencil_mesh import StencilMesh
from nova.equilibrium.forward_operator import (
    set_support_clip_mode,
    support_clip_mode,
)
from nova.jax.config import configure_dtypes
from nova.media import poloidal
from nova.media.ink import DEFAULT_INK, poloidal_axes
from nova.media.sources.frame import WallUnit
from scripts.analytic_oracle_fixtures import measure as oracle_fixture
from scripts.oracle_rebaseline import measure as recovery

CASES = ("diverted-single-null", "weak-rotation-reactor-static")
REQUESTED_CELLS = -110
SEED_POLICIES = ("analytic", "current_centroid_disc")
# The route's own seed is not one of the two policies under comparison; it
# is measured in the forced one-trip control and labelled so every arm row
# carries a seed policy.
PRODUCTION_SEED_POLICY = "production_route_seed"
NEGATIVE_CONTROL = (
    "run the analytic-seed arm of the diverted case with the active-set "
    "budget forced to one trip and observe the receipt read converged "
    "false with residual 0.7166 at trip one, reproducing the production "
    "arm of the exact-mode construction comparison within one percent"
)
EXPECTED_ONE_TRIP_RESIDUAL = 0.7166
CONSTRUCTION_COMPARISON_RECEIPT = (
    "docs/figures/plasma-cell-read-fidelity/exact-mode-construction-comparison.json"
)
# The declared number is trip one of the diverted case's production arm in
# the construction comparison; carried at its stored precision so the
# reproduction band is measured against the receipt rather than the prose.
PRODUCTION_TRIP_ONE_RESIDUAL = 0.7165572322836796
COMMITTED_ONE_TRIP_RECEIPT = (
    "docs/figures/plasma-cell-read-fidelity/"
    "b43714114-diverted-single-null-one-trip.json"
)
COMMITTED_ONE_TRIP_RESIDUAL = 0.2586805230398537
DECLARED_CONTROL_SOURCE = "declared in the dispatch brief"
PRODUCTION_CONTROL_SOURCE = (
    "trip one of the production arm in " + CONSTRUCTION_COMPARISON_RECEIPT
)
COMMITTED_CONTROL_SOURCE = "forced one-trip arm in " + COMMITTED_ONE_TRIP_RECEIPT
ONE_TRIP_RELATIVE_TOLERANCE = 0.01
SEED_RANGE_RELATIVE_TOLERANCE = 1e-6
FIGURE_STEM = "seed-policy-at-main"
FIGURE_URL = "/nova/figures/plasma-cell-read-fidelity-seed"
MOMENT_FIELDS = ("cell_current", "radial_moment", "vertical_moment")


def _digest(state):
    return hashlib.sha256(np.asarray(state, dtype=np.float64).tobytes()).hexdigest()


def _point(value):
    point = np.asarray(value, dtype=np.float64)
    if np.all(np.isfinite(point)):
        return point.tolist()
    return None


def _write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def _jsonable(value):
    """Reduce a route receipt to JSON-safe containers and scalars."""
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    return value


def _flux_range(state):
    """Flux extent of a seed state over its finite samples."""
    values = np.asarray(state, dtype=np.float64).reshape(-1)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return {
            "min_wb": None,
            "max_wb": None,
            "amplitude_wb": None,
            "abs_max_wb": None,
            "nonfinite_count": int(values.size),
            "sample_count": int(values.size),
        }
    return {
        "min_wb": float(finite.min()),
        "max_wb": float(finite.max()),
        "amplitude_wb": float(finite.max() - finite.min()),
        "abs_max_wb": float(np.abs(finite).max()),
        "nonfinite_count": int(values.size - finite.size),
        "sample_count": int(values.size),
    }


def _seed_flux_ranges(seeds):
    """Flux range per seed beside the analytic seed's own range.

    Amplitude is the discriminating quantity: the route's saddle-anchored
    seed carries several times the analytic seed's flux extent, so the two
    are different initial states rather than two spellings of one.
    """
    ranges = []
    analytic_amplitude = None
    for policy, state in seeds:
        entry = {"seed_policy": policy, "seed_state_digest": _digest(state)}
        entry.update(_flux_range(state))
        if policy == "analytic":
            analytic_amplitude = entry["amplitude_wb"]
        ranges.append(entry)
    for entry in ranges:
        amplitude = entry["amplitude_wb"]
        entry["analytic_amplitude_wb"] = analytic_amplitude
        if amplitude is None or not analytic_amplitude:
            entry["amplitude_ratio_to_analytic"] = None
        else:
            entry["amplitude_ratio_to_analytic"] = float(amplitude / analytic_amplitude)
    return ranges


def _production_seed_receipt(state, route_receipt):
    """Label the route's own seed so its arm row carries a seed policy."""
    state = np.asarray(state, dtype=np.float64)
    return {
        "seed_policy": PRODUCTION_SEED_POLICY,
        "seed_state_digest": _digest(state),
        "seed_construction": "production_route",
        "route_seed_receipt": _jsonable(route_receipt),
    }


def _nulls(operator, state):
    _, topology = operator.read(jnp.asarray(state, dtype=jnp.float64))
    grid = operator._fixed_design_topology.grid
    census = grid.candidate_table_status(operator.null_flux_pool(jnp.asarray(state)))
    assert int(census["retained_count"][0]) > 0, "axis control empty"
    saddles = np.asarray(census["retained_candidate"])[1]
    valid = np.asarray(census["retained_valid"])[1]
    return {
        "axis_rz_m": _point(topology.axis),
        "x_point_rz_m": _point(topology.x_point),
        "qualified_saddles_rz_m": saddles[valid, :2].tolist(),
        "retained_count": np.asarray(census["retained_count"]).tolist(),
        "axis_flux_wb": float(topology.axis_flux),
        "boundary_flux_wb": float(topology.boundary_flux),
    }


def _terminal_moment_counts(operator, state):
    """Count non-finite clipped-support current moments at a state."""
    moments = operator.cell_current_moments(jnp.asarray(state, dtype=jnp.float64))
    counts = {}
    total = 0
    for name in MOMENT_FIELDS:
        field = np.asarray(getattr(moments, name), dtype=np.float64)
        count = int(np.count_nonzero(~np.isfinite(field)))
        counts["nonfinite_" + name] = count
        total += count
    counts["nonfinite_moment_total"] = total
    counts["moment_values"] = int(np.asarray(moments.cell_current).size)
    return counts


def _analytic_seed(analytic):
    state = np.asarray(analytic, dtype=np.float64)
    receipt = {
        "seed_policy": "analytic",
        "seed_state_digest": _digest(state),
    }
    return jnp.asarray(state, dtype=jnp.float64), receipt


def _disc_seed(profile, target_current, centroid):
    """Build the current-centroid disc seed with no diverted geometry."""
    portfolio = profile.cold_seed_portfolio(
        target_current, centroid, diverted_geometry=None
    )
    branches = portfolio.branches
    wanted = int(ColdSeedConstruction.CURRENT_CENTROID_DISC)
    index = branches.construction.tolist().index(wanted)
    state = np.asarray(branches.flux[index], dtype=np.float64)
    receipt = {
        "seed_policy": "current_centroid_disc",
        "seed_state_digest": _digest(state),
        "seed_construction": "current_centroid_disc",
        "seed_branch_index": index,
        "seed_radius_m": float(np.asarray(branches.radius)[index]),
        "supported_cell_count": int(np.asarray(branches.supported_cells)[index]),
        "stored_flux_samples_used": bool(
            np.asarray(branches.stored_flux_samples_used)[index]
        ),
    }
    return jnp.asarray(state, dtype=jnp.float64), receipt


def _arm(profile, request, operator, reference, pitch, seed_receipt):
    started = perf_counter()
    receipt = profile.solve(request)
    equilibrium = receipt.equilibrium
    state = np.asarray(jax.block_until_ready(equilibrium.flux), dtype=np.float64)
    assert np.all(np.isfinite(state)), "terminal state carries non-finite flux"
    history = equilibrium.fixed_point
    trips = int(history.active_set_iterations)
    residuals = np.asarray(history.active_set_residuals)[:trips]
    assert len(residuals) == trips and np.all(np.isfinite(residuals))
    nulls = _nulls(operator, state)
    reference_x = reference["x_point_rz_m"]
    terminal_x = nulls["x_point_rz_m"]
    distance = None
    if reference_x is not None and terminal_x is not None:
        distance = float(np.linalg.norm(np.asarray(terminal_x) - reference_x))
    row = {
        "seed_policy": seed_receipt["seed_policy"],
        "seed_receipt": seed_receipt,
        "active_set_budget": request.policy.active_set_steps,
        "terminal_residual": float(history.residual),
        "converged": bool(history.converged),
        "qualified": bool(receipt.qualified),
        "trip_count": trips,
        "per_trip_residual_history": [
            {"trip": index + 1, "residual": float(value)}
            for index, value in enumerate(residuals)
        ],
        "terminal_axis_rz_m": nulls["axis_rz_m"],
        "terminal_x_point_rz_m": terminal_x,
        "reference_x_point_rz_m": reference_x,
        "x_point_distance_m": distance,
        "certificate_position_bound_m": (
            certificate.TOPOLOGY_POSITION_BOUND_PITCHES * pitch
        ),
        "certificate_residual_bound": certificate.TERMINAL_RESIDUAL_BOUND,
        "jax_platform": jax.default_backend(),
        "terminal_nulls": nulls,
        "termination_reason": int(receipt.termination_reason),
        "wall_seconds": perf_counter() - started,
        "compilation_cache_hit": bool(receipt.compilation_cache_hit),
    }
    row.update(_terminal_moment_counts(operator, state))
    print("ARM " + json.dumps(row, allow_nan=False), flush=True)
    return row, state


def require_converged(row):
    """Refuse a convergence claim the solve receipt does not support."""
    if (
        not row["converged"]
        or not row["qualified"]
        or not np.isfinite(row["terminal_residual"])
        or row["terminal_residual"] > row["certificate_residual_bound"]
    ):
        raise ValueError(
            "converged assertion refused: "
            f"{row['seed_policy']} residual={row['terminal_residual']:.12g} "
            f"converged={row['converged']} trips={row['trip_count']}"
        )
    if row["reference_x_point_rz_m"] is not None:
        distance = row["x_point_distance_m"]
        if distance is None or distance > row["certificate_position_bound_m"]:
            raise ValueError(
                "converged assertion refused: terminal saddle "
                "exceeds the certificate position bound"
            )


def _policy_seeds(built, profile):
    """Build the two seed policies under comparison for one case."""
    analytic_seed, analytic_receipt = _analytic_seed(built["analytic"])
    disc_seed, disc_receipt = _disc_seed(
        profile, built["target_current"], built["centroid"]
    )
    return {
        "analytic": (analytic_seed, analytic_receipt),
        "current_centroid_disc": (disc_seed, disc_receipt),
    }


def _production_seed_state(profile, built):
    seed, branch, route_receipt = certificate._production_seed(
        profile,
        built["case_name"],
        float(built["target_current"]),
        built["centroid"],
        built["current_receipt"],
    )
    return np.asarray(seed, dtype=np.float64), branch, route_receipt


def _record_seed_flux_ranges(row, built, seeds):
    """Record every seed's flux range beside the arms, against the analytic range.

    The route's own seed is built here too, so each case's record answers the
    same question the first-step node raised: how much larger is the seed the
    route actually uses than the analytic state it is compared against.
    """
    production_state, branch, route_receipt = _production_seed_state(
        built["profile"], built
    )
    row["seed_flux_ranges"] = _seed_flux_ranges(
        [
            ("analytic", np.asarray(built["analytic"], dtype=np.float64)),
            (
                "current_centroid_disc",
                np.asarray(seeds["current_centroid_disc"][0], dtype=np.float64),
            ),
            (PRODUCTION_SEED_POLICY, production_state),
        ]
    )
    return production_state, branch, route_receipt


def _run_control(
    profile,
    built,
    row,
    reference,
    pitch,
    carrier,
    seeds,
    negative_log,
    production,
):
    """Run the declared one-trip control on each seed under a one-trip budget.

    The control is the analytic-seed arm with the active-set budget forced to
    one trip, alongside the route's own seed under the same budget so the
    declared residual is measured against the arm it came from.
    """
    operator = built["operator"]
    production_state, _branch, route_receipt = production
    control_arms = [
        (
            "analytic_seed",
            seeds["analytic"][0],
            seeds["analytic"][1],
            [
                (EXPECTED_ONE_TRIP_RESIDUAL, DECLARED_CONTROL_SOURCE),
                (COMMITTED_ONE_TRIP_RESIDUAL, COMMITTED_CONTROL_SOURCE),
            ],
        )
    ]
    control_arms.append(
        (
            PRODUCTION_SEED_POLICY,
            jnp.asarray(production_state, dtype=jnp.float64),
            _production_seed_receipt(production_state, route_receipt),
            [(PRODUCTION_TRIP_ONE_RESIDUAL, PRODUCTION_CONTROL_SOURCE)],
        )
    )
    records = []
    for label, seed, seed_receipt, expectations in control_arms:
        request = certificate._certificate_solve_request(
            profile, seed, float(built["target_current"]), carrier_identity=carrier
        )
        one_request = replace(
            request, policy=replace(request.policy, active_set_steps=1)
        )
        mutation, _state = _arm(
            profile, one_request, operator, reference, pitch, seed_receipt
        )
        records.extend(
            _one_trip_control(mutation, row, negative_log, label, expectations)
        )
    return records


def _recorded_range(row, policy):
    for entry in row.get("seed_flux_ranges", []):
        if entry["seed_policy"] == policy:
            return entry
    return None


def _draw_nulls(axis, nulls, wall_units, style):
    admitted = nulls["x_point_rz_m"]
    others = np.asarray(nulls["qualified_saddles_rz_m"]).reshape(-1, 2)
    if admitted is not None:
        others = others[np.linalg.norm(others - admitted, axis=1) > 1e-10]
    return poloidal.draw_nulls(
        axis,
        magnetic_axis=nulls["axis_rz_m"],
        x_points=admitted,
        other_x_points=others,
        contain=wall_units,
        style=style,
    )


def render(receipt, output):
    """Draw the analytic state beside both terminal states per case."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(
        len(receipt["cases"]), 3, figsize=(12, 9), constrained_layout=True
    )
    # A one-row grid comes back as a flat sequence of axes, so indexing it by
    # (row, column) fails; the receipt may carry a single case when a run is
    # read part-way through.
    axes = np.atleast_2d(axes)
    reference_style = DEFAULT_INK.variant(
        axis_color="#3366cc",
        xpoint_color="#3366cc",
        axis_markersize=9,
        xpoint_markersize=11,
    )
    panel_style = DEFAULT_INK.variant(axis_markersize=5, xpoint_markersize=6)
    for row_index, row in enumerate(receipt["cases"]):
        with np.load(output / row["state_file"]) as data:
            coordinates, wall = data["coordinates"], data["wall"]
            states = [data["analytic"], data["analytic_seed"], data["disc_seed"]]
        units = tuple(
            WallUnit(
                wall[start:stop, 0],
                wall[start:stop, 1],
                closed=bool(closed),
                kind=kind,
            )
            for start, stop, closed, kind in row["wall_units"]
        )
        reference = row["reference_nulls"]
        levels = poloidal.contour_levels(
            states[0],
            count=15,
            axis=reference["axis_flux_wb"],
            boundary=reference["boundary_flux_wb"],
        )
        row["contour_levels_wb"] = levels.tolist()
        row["panels"] = []
        for column, state in enumerate(states):
            axis = axes[row_index, column]
            radial, height, raster = certificate._raster_field(coordinates, state, wall)
            contour = poloidal.draw_flux_contours(
                axis, radial, height, raster, levels, color="#444444"
            )
            segments = sum(len(part) > 1 for group in contour.allsegs for part in group)
            assert segments > 0, "contour positive control is empty"
            poloidal.draw_wall(axis, units=units)
            if column == 0:
                nulls = reference
            else:
                nulls = row["arms"][column - 1]["terminal_nulls"]
            reference_tally = _draw_nulls(axis, reference, units, reference_style)
            panel_tally = _draw_nulls(axis, nulls, units, panel_style)
            poloidal_axes(axis)
            if column == 0:
                caption = "Analytic input\nresidual=n/a; converged=n/a; trips=0"
            else:
                arm = row["arms"][column - 1]
                caption = (
                    f"seed={arm['seed_policy']}\n"
                    f"residual={arm['terminal_residual']:.6g}; "
                    f"converged={arm['converged']}; trips={arm['trip_count']}"
                )
                extent = _recorded_range(row, arm["seed_policy"])
                if extent is not None and extent["amplitude_wb"] is not None:
                    caption += f"\nseed amplitude={extent['amplitude_wb']:.3g} Wb"
                    if extent["amplitude_ratio_to_analytic"] is not None:
                        caption += (
                            f" ({extent['amplitude_ratio_to_analytic']:.2f}x analytic)"
                        )
            axis.set_title(caption, fontsize=10)
            row["panels"].append(
                {
                    "caption": caption,
                    "contour_segments": segments,
                    "reference_markers": reference_tally,
                    "panel_markers": panel_tally,
                    "axis_off": not axis.axison,
                }
            )
    figure.suptitle(
        "Exact clip: analytic input and both seed-policy terminal states",
        fontsize=15,
    )
    figure.supxlabel(
        "Blue large markers: analytic-state read; "
        "red small markers: panel-state read.\n"
        "Filled triangle: axis; filled cross: admitted saddle; "
        "hollow crosses: other qualified saddles. One shared level array.",
        fontsize=10,
    )
    figure.savefig(output / (FIGURE_STEM + ".png"), dpi=160)
    figure.savefig(output / (FIGURE_STEM + ".svg"))
    plt.close(figure)


def _construction(case_name):
    """Build the certificate route for one case at the reduced rung."""
    carrier_case, source_case, exact = certificate._case(case_name)
    machine = certificate._case_machine(case_name, carrier_case, exact, REQUESTED_CELLS)
    coordinates = np.vstack(
        (machine.node, machine.wall_node, machine.sample_coordinates)
    )
    analytic = certificate._exact_state(case_name, exact, coordinates)
    empty = oracle_fixture.forward_operator(source_case, machine)
    exact_physical, exterior, _cache = oracle_fixture.cached_fixture_exterior(
        source_case, exact, machine, empty, analytic
    )
    operator = oracle_fixture.forward_operator(source_case, machine, exterior)
    profile = ForwardProfile(
        operator,
        StencilMesh(machine.node, machine.stencil, machine.area),
        newton_steps=recovery.NEWTON_STEPS,
    )
    target_current, centroid, current_receipt = certificate._closed_form_current_target(
        case_name, source_case, operator, exact_physical
    )
    return {
        "case_name": case_name,
        "operator": operator,
        "profile": profile,
        "machine": machine,
        "coordinates": coordinates,
        "analytic": analytic,
        "target_current": float(target_current),
        "centroid": np.asarray(centroid, dtype=np.float64),
        "current_receipt": current_receipt,
    }


def _case_verdict(row):
    """Answer the seed-policy question from the two arms a case carries."""
    arms = {arm["seed_policy"]: arm for arm in row["arms"]}
    analytic = arms.get("analytic")
    disc = arms.get("current_centroid_disc")
    if analytic is None or disc is None:
        return None
    if disc["converged"] and not analytic["converged"]:
        return "disc_converges_where_analytic_does_not"
    if analytic["converged"] and not disc["converged"]:
        return "analytic_converges_where_disc_does_not"
    if analytic["converged"] and disc["converged"]:
        return "both_converge"
    return "both_fail"


def _refresh_verdict(receipt):
    """One field answering the seed-policy question at the cases measured.

    The answer is carried as that verdict's own name when every case agrees
    and as a case-keyed mapping when they do not, so a reader never has to
    average two answers into one word.  Each case row's verdict is derived
    from the arms it carries whenever the row does not already state one,
    so a row whose arms landed before a crash still reports an answer its
    own arms support.
    """
    for row in receipt["cases"]:
        if "seed_policy_verdict" not in row and len(row.get("arms", ())) == len(
            SEED_POLICIES
        ):
            derived = _case_verdict(row)
            if derived is not None:
                row["seed_policy_verdict"] = derived
    verdicts = {
        row["case"]: row["seed_policy_verdict"]
        for row in receipt["cases"]
        if "seed_policy_verdict" in row
    }
    if not verdicts:
        receipt.pop("seed_policy_verdict", None)
        return
    distinct = set(verdicts.values())
    receipt["seed_policy_verdict"] = distinct.pop() if len(distinct) == 1 else verdicts


def _write_receipt(output, receipt):
    _refresh_verdict(receipt)
    _write_json(output / (FIGURE_STEM + ".json"), receipt)


def _run_case(built, receipt, output, negative_log):
    case_name = built["case_name"]
    operator = built["operator"]
    profile = built["profile"]
    machine = built["machine"]
    reference = _nulls(operator, built["analytic"])
    pitch = float(np.sqrt(np.median(np.asarray(machine.area))))
    offsets = np.asarray(operator.wall_unit_offsets)
    row = {
        "case": case_name,
        "requested_cells": REQUESTED_CELLS,
        "realised_cells": len(machine.node),
        "characteristic_pitch_m": pitch,
        "reference_nulls": reference,
        "arms": [],
        "state_file": case_name + "-seed-policy.npz",
        "wall_units": [
            [int(a), int(b), bool(c), k]
            for a, b, c, k in zip(
                offsets[:-1],
                offsets[1:],
                operator.wall_unit_closed,
                operator.wall_unit_kinds,
                strict=True,
            )
        ],
    }
    receipt["cases"].append(row)
    seeds = _policy_seeds(built, profile)
    carrier = f"solovev:{case_name}:{REQUESTED_CELLS}"
    states = {}
    for policy in SEED_POLICIES:
        seed, seed_receipt = seeds[policy]
        request = certificate._certificate_solve_request(
            profile, seed, float(built["target_current"]), carrier_identity=carrier
        )
        arm, state = _arm(profile, request, operator, reference, pitch, seed_receipt)
        row["arms"].append(arm)
        states["analytic_seed" if policy == "analytic" else "disc_seed"] = state
        _write_receipt(output, receipt)
    verdict = _case_verdict(row)
    row["seed_policy_verdict"] = verdict
    print("VERDICT " + case_name + " " + verdict, flush=True)
    np.savez(
        output / row["state_file"],
        coordinates=built["coordinates"],
        wall=machine.wall_node,
        analytic=built["analytic"],
        **states,
    )
    production = _record_seed_flux_ranges(row, built, seeds)
    if case_name == CASES[0]:
        receipt["negative_control"].extend(
            _run_control(
                profile,
                built,
                row,
                reference,
                pitch,
                carrier,
                seeds,
                negative_log,
                production,
            )
        )
    _write_receipt(output, receipt)


def _seed_rebuild_records(row, seeds):
    """Record how a rebuilt policy seed relates to the arm that used it.

    Construction identity is asserted, because a different branch, radius
    or support count would mean the seed under comparison had changed.  The
    arm-level flux range is asserted within a relative tolerance for the
    same reason.  The bit-level digest is recorded rather than asserted:
    the clipped moment seed was measured to rebuild to a different digest
    in a fresh process while the analytic seed reproduced exactly, so
    requiring digest equality would refuse a run that does reproduce the
    construction being checked.
    """
    records = []
    fields = ("seed_branch_index", "seed_radius_m", "supported_cell_count")
    for arm in row["arms"]:
        policy = arm["seed_policy"]
        if policy not in seeds:
            continue
        recorded = arm["seed_receipt"]
        rebuilt = seeds[policy][1]
        checked = []
        for field in fields:
            if field in recorded and field in rebuilt:
                assert recorded[field] == rebuilt[field], (
                    f"{row['case']} rebuilt {policy} with a different {field}"
                )
                checked.append(field)
        entry = {
            "case": row["case"],
            "seed_policy": policy,
            "recorded_digest": recorded["seed_state_digest"],
            "rebuilt_digest": rebuilt["seed_state_digest"],
            "digest_matches": (
                recorded["seed_state_digest"] == rebuilt["seed_state_digest"]
            ),
            "construction_fields_checked": checked,
        }
        prior = _recorded_range(row, policy)
        rebuilt_range = _flux_range(seeds[policy][0])
        entry["rebuilt_amplitude_wb"] = rebuilt_range["amplitude_wb"]
        if prior is not None and prior.get("amplitude_wb"):
            relative = abs(rebuilt_range["amplitude_wb"] / prior["amplitude_wb"] - 1.0)
            entry["recorded_amplitude_wb"] = prior["amplitude_wb"]
            entry["amplitude_relative_difference"] = relative
            assert relative <= SEED_RANGE_RELATIVE_TOLERANCE, (
                f"{row['case']} rebuilt {policy} with a different flux amplitude"
            )
        records.append(entry)
    return records


def complete_receipt(output, negative_log):
    """Measure the control and any case the receipt does not already carry.

    The receipt is spliced rather than rewritten: a case it already carries
    was measured by the revision it records, so only its control arms and
    seed flux ranges are added, and the rebuilt policy seeds are compared
    with the arms record under the construction-identity check.
    """
    output.mkdir(parents=True, exist_ok=True)
    configure_dtypes()
    assert jax.config.jax_enable_x64 is True
    assert jax.default_backend() == "gpu", "measurement requires the GPU lane"
    receipt_path = output / (FIGURE_STEM + ".json")
    receipt = json.loads(receipt_path.read_text())
    receipt["completion_revision"] = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], text=True
    ).strip()
    receipt["completion_job_id"] = os.environ.get("SLURM_JOB_ID")
    receipt["completion_slurm_partition"] = os.environ.get("SLURM_JOB_PARTITION")
    receipt.setdefault("negative_control", [])
    # The control is re-measured here, so drop any records a previous
    # attempt wrote for the case it belongs to rather than duplicate them.
    receipt["negative_control"] = [
        record for record in receipt["negative_control"] if record["case"] != CASES[0]
    ]
    negative_log.write_text(NEGATIVE_CONTROL + "\n")
    previous = support_clip_mode()
    set_support_clip_mode("exact")
    try:
        for case_name in CASES:
            print("BUILD case=" + case_name, flush=True)
            built = _construction(case_name)
            carrier = f"solovev:{case_name}:{REQUESTED_CELLS}"
            existing = next(
                (row for row in receipt["cases"] if row["case"] == case_name), None
            )
            if existing is None:
                _run_case(built, receipt, output, negative_log)
                continue
            seeds = _policy_seeds(built, built["profile"])
            existing["seed_rebuilds"] = _seed_rebuild_records(existing, seeds)
            production = _record_seed_flux_ranges(existing, built, seeds)
            if case_name == CASES[0]:
                receipt["negative_control"].extend(
                    _run_control(
                        built["profile"],
                        built,
                        existing,
                        existing["reference_nulls"],
                        float(existing["characteristic_pitch_m"]),
                        carrier,
                        seeds,
                        negative_log,
                        production,
                    )
                )
            _write_receipt(output, receipt)
        render(receipt, output)
        receipt["figure_src"] = FIGURE_URL + "/" + FIGURE_STEM + ".png"
        _write_receipt(output, receipt)
    finally:
        set_support_clip_mode(previous)
    print("MEASUREMENT_COMPLETE", flush=True)


def _one_trip_control(mutation, row, negative_log, arm_label, expectations):
    """Record the declared control for one forced one-trip arm.

    The refusal is the control and it is asserted.  An expectation is
    recorded beside the measured one-trip residual rather than asserted,
    so an expectation that fails to reproduce reads as a finding about
    the receipt it came from instead of stopping the measurement.
    """
    residual = mutation["terminal_residual"]
    try:
        require_converged(mutation)
    except ValueError as error:
        reason = str(error)
    else:
        raise AssertionError("forced-one-trip control did not refuse")
    with negative_log.open("a") as stream:
        stream.write("case=" + row["case"] + " arm=" + arm_label + " " + reason + "\n")
    base = {
        "case": row["case"],
        "control_arm": arm_label,
        "one_trip_residual": residual,
        "declared_control": NEGATIVE_CONTROL,
        "converged": mutation["converged"],
        "trip_count": mutation["trip_count"],
        "termination_reason": mutation["termination_reason"],
        "refused": True,
        "reason": reason,
        "receipt": mutation,
    }
    assert mutation["trip_count"] == 1
    records = []
    for expected, source in expectations:
        record = dict(base)
        if expected is None:
            relative = None
        else:
            relative = abs(residual / expected - 1.0)
        record["expected_residual"] = expected
        record["expected_source"] = source
        record["relative_difference"] = relative
        record["relative_tolerance"] = ONE_TRIP_RELATIVE_TOLERANCE
        record["reproduces_expected"] = (
            None if relative is None else relative <= ONE_TRIP_RELATIVE_TOLERANCE
        )
        records.append(record)
    return records


def measure(output, negative_log):
    configure_dtypes()
    assert jax.config.jax_enable_x64 is True
    assert jax.default_backend() == "gpu", "measurement requires the GPU lane"
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    receipt = {
        "revision": revision,
        "worktree": str(Path.cwd()),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_partition": os.environ.get("SLURM_JOB_PARTITION"),
        "devices": [str(device) for device in jax.devices()],
        "jax_platform": jax.default_backend(),
        "x64": True,
        "support_clip_mode": "exact",
        "requested_cells": REQUESTED_CELLS,
        "cases": [],
        "negative_control": [],
    }
    output.mkdir(parents=True, exist_ok=True)
    negative_log.write_text(NEGATIVE_CONTROL + "\n")
    previous = support_clip_mode()
    set_support_clip_mode("exact")
    try:
        for case_name in CASES:
            print("BUILD case=" + case_name, flush=True)
            built = _construction(case_name)
            _run_case(built, receipt, output, negative_log)
        render(receipt, output)
        receipt["figure_src"] = FIGURE_URL + "/" + FIGURE_STEM + ".png"
        _write_receipt(output, receipt)
    finally:
        set_support_clip_mode(previous)
    print("MEASUREMENT_COMPLETE", flush=True)


def derive_verdicts(output):
    """Re-derive the receipt's seed-policy answers from the arms it carries.

    No measurement and no GPU: the arms are the evidence, and a row written
    before a crash carries the arms its verdict rests on, so the answer is
    recomputed here rather than left absent.
    """
    receipt_path = output / (FIGURE_STEM + ".json")
    receipt = json.loads(receipt_path.read_text())
    _write_receipt(output, receipt)
    print(
        "DERIVED " + json.dumps(receipt.get("seed_policy_verdict"), allow_nan=False),
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--negative-control-log", type=Path, required=False)
    parser.add_argument("--render-only", action="store_true")
    parser.add_argument(
        "--derive-verdicts",
        action="store_true",
        help="refresh the receipt's seed-policy answers from its arms; no GPU",
    )
    parser.add_argument(
        "--complete-receipt",
        action="store_true",
        help="add the control and any case the existing receipt lacks",
    )
    arguments = parser.parse_args()
    if arguments.derive_verdicts:
        derive_verdicts(arguments.output)
        return
    if arguments.negative_control_log is None or not (
        arguments.negative_control_log.parent.is_dir()
    ):
        parser.error("negative control log directory is absent")
    if arguments.complete_receipt:
        complete_receipt(arguments.output, arguments.negative_control_log)
        return
    if arguments.render_only:
        receipt_path = arguments.output / (FIGURE_STEM + ".json")
        receipt = json.loads(receipt_path.read_text())
        render(receipt, arguments.output)
        _write_receipt(arguments.output, receipt)
        return
    if arguments.negative_control_log is None:
        parser.error("--negative-control-log is required for measurement")
    measure(arguments.output, arguments.negative_control_log)


if __name__ == "__main__":
    main()
