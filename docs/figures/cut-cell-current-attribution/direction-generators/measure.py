"""Compare support-frozen, geometry-continuation and branch Newton proposals."""

from __future__ import annotations

import argparse
from contextlib import nullcontext
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import subprocess
from time import perf_counter

import numpy as np

from nova.jax.config import configure_dtypes

configure_dtypes()
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

from nova.equilibrium import clip_quadrature as quadrature  # noqa: E402
from nova.equilibrium import fixed_point as fixed  # noqa: E402
from nova.equilibrium.domain import DomainMasks  # noqa: E402
from nova.equilibrium.stencil_mesh import (  # noqa: E402
    CellCurrentMoments,
    flux_field_polynomial,
)

ROOT = Path(__file__).resolve().parents[4]
spec = importlib.util.spec_from_file_location(
    "merit_instrument", Path(__file__).parents[1] / "merit-model/measure.py"
)
previous = importlib.util.module_from_spec(spec)
spec.loader.exec_module(previous)
FRACTIONS = previous.FRACTIONS


def terms(image, value):
    result = previous.norm_terms(image, value)
    return {
        k: result[k]
        for k in ("merit", "relative_sup", "numerator_wb", "denominator_wb")
    }


def changed(a, b):
    return np.flatnonzero(np.asarray(a) != np.asarray(b)).tolist()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cells", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    start = perf_counter()
    assert jax.config.jax_enable_x64 and jax.default_backend() == "cpu"
    previous.fixture._cache_lock = lambda store: nullcontext(0.0)

    def refuse_store(*_args, **_kwargs):
        raise RuntimeError("Fixture cache miss requires an out-of-scope cache write")

    previous.fixture.ZarrStore.store = refuse_store
    machine, op, _profile, _exact, _seed, target, _seed_record, _coord, analytic = (
        previous._build("weak-rotation-reactor-static", -args.cells)
    )
    compact_path = (
        ROOT
        / "docs/figures/cut-cell-current-attribution/continuous-support"
        / "rows-titan-adjoint/record"
        / f"cells-{args.cells}/receipt.json"
    )
    compact = json.loads(compact_path.read_text())
    receipt_path = Path(compact["raw_receipt"])
    assert (
        hashlib.sha256(receipt_path.read_bytes()).hexdigest() == compact["raw_sha256"]
    )
    receipt = json.loads(receipt_path.read_text())
    state = jnp.asarray(receipt["trips"][-1]["flux"], dtype=jnp.float64)
    assert previous.digest(state) == compact["terminal"]["state_digest"]
    np.testing.assert_array_equal(analytic, receipt["analytic_state"])
    target = jnp.asarray(target, dtype=jnp.float64)
    external = op.external()
    program = jax.jit(op.traced_flux_map(target_current=target))

    def mapped(value):
        return program(value, external, op, target)

    @jax.jit
    def tangent(value, vector):
        return jax.jvp(mapped, (value,), (vector,))[1]

    @jax.jit
    def diagnostic(value, operator):
        masks, topology, _sample, support = operator._support_partition(value)
        current = operator.cell_current_moments(value).cell_current
        return {
            "confined": masks.confined_profile,
            "open": masks.open_field_line,
            "support": current != 0.0,
            "profile_support": support.included,
            "area": support.area,
            "shadow": operator.residual_shadow_mask(value),
            "axis": topology.axis,
            "x_point": topology.x_point,
            "axis_flux": topology.axis_flux,
            "boundary_flux": topology.boundary_flux,
        }

    @jax.jit
    def classification(value, operator):
        masks, _topology, _connected, _admitted = operator._fixed_design_read(
            value[: operator.physical_node_number], None
        )
        return jnp.stack(
            (masks.confined_profile, masks.open_field_line, masks.private_flux)
        )

    def qualify(mapper, value):
        image, action = jax.linearize(mapper, value)
        return fixed._qualified_krylov_step(
            lambda v: v - action(v),
            image - value,
            fixed._relative_residual(image, value),
            gmres_iterations=30,
            condition_ratio_limit=math.e,
            preceding_condition_baseline=jnp.asarray(jnp.nan),
        )

    @jax.jit
    def newton(value, operator, ext, current):
        return qualify(lambda v: program(v, ext, operator, current), value)

    @jax.jit
    def geometry(value, operator):
        masks, topology, sample, support = operator._support_partition(value)
        field = flux_field_polynomial(
            operator._support_moment_stencils, masks.psi_norm, sample
        )
        coefficient = (-field.coefficient).at[:, 0].add(1.0)
        selected = (
            masks.profile_participation & field.active & (support.vertex_count >= 3)
        )
        clipped = quadrature._quadratic_support(
            support.support_vertices,
            support.vertex_count,
            support.centroids,
            coefficient,
            field.centre,
            field.scale,
            selected,
        )
        return (
            masks.label,
            topology,
            clipped,
            selected,
            operator.residual_shadow_mask(value),
        )

    def frozen_map(value, geom, operator, ext, current):
        label, topology, clipped, selected, shadow = geom
        grid_flux, _wall = operator.topology.split_flux_map(
            value[: operator.physical_node_number]
        )
        normalized = operator.topology.normalize(
            topology.axis_flux, topology.boundary_flux, grid_flux
        )
        sample = (
            operator.sample_node_flux(value) - topology.axis_flux
        ) / topology.flux_span
        field = flux_field_polynomial(
            operator._support_moment_stencils, normalized, sample
        )
        moments = quadrature.clipped_support_current_moments(
            clipped,
            selected & clipped.included,
            field,
            operator.source.core,
            cut_cell_capacity=operator._cut_cell_bank_capacity,
            boundary_reduction=True,
        )
        mask = DomainMasks(label=label, psi_norm=normalized).profile_participation
        moments = CellCurrentMoments(*(jnp.where(mask, m, 0.0) for m in moments))
        coupled = operator.coupling_current_moments(moments)
        amplitude = operator.current_normalisation_amplitude(
            current, jnp.sum(coupled.cell_current)
        )
        coupled = operator.scaled_current_moments(coupled, amplitude)
        return jnp.where(shadow, value, ext + operator.current_moment_image(coupled))

    frozen_program = jax.jit(frozen_map)

    @jax.jit
    def frozen_newton(value, geom, operator, ext, current):
        return qualify(lambda v: frozen_map(v, geom, operator, ext, current), value)

    assert op.source.common_sol is None and op.source.private_flux is None
    image = mapped(state)
    incumbent = terms(image, state)
    base_diag = diagnostic(state, op)
    output_path = out / f"cells-{len(machine.node)}.json"
    data = {
        "revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "worktree": str(ROOT),
        "backend": jax.default_backend(),
        "devices": str(jax.devices()),
        "requested_cells": args.cells,
        "realised_cells": len(machine.node),
        "input_receipt": str(receipt_path),
        "input_sha256": compact["raw_sha256"],
        "state_digest": previous.digest(state),
        "state": state,
        "incumbent": incumbent,
        "base_diagnostics": base_diag,
        "characteristic_pitch_m": compact["characteristic_pitch_m"],
        "locked_residual_bound": previous.certificate.TERMINAL_RESIDUAL_BOUND,
        "directions": {},
        "completed": False,
    }

    def checkpoint(message):
        data["elapsed_seconds"] = perf_counter() - start
        previous.write(output_path, data)
        print(message, flush=True)

    checkpoint(
        f"BASE cells={len(machine.node)} merit={incumbent['merit']:.12g} "
        f"residual={incumbent['relative_sup']:.12g}"
    )

    def ladder(direction, branch_origin=None, branch_geom=None, details=True):
        derivative = tangent(state, direction)
        if branch_geom is not None:
            model_origin = state

            def local(v):
                return frozen_program(v, branch_geom, op, external, target)
        else:
            model_origin = state if branch_origin is None else branch_origin
            local = mapped
        model_image = local(model_origin)
        # This branch model is distinct from the origin's unchanged production model.
        model_current = (
            model_image + jax.jvp(local, (model_origin,), (state - model_origin,))[1]
        )
        model_slope = jax.jvp(local, (model_origin,), (direction,))[1]
        predicted_incumbent = terms(model_current, state)["merit"]
        rows = []
        for fraction in FRACTIONS:
            candidate = state + fraction * direction
            actual_image = mapped(candidate)
            actual = terms(actual_image, candidate)
            predicted = terms(image + fraction * derivative, candidate)
            branch_predicted = terms(model_current + fraction * model_slope, candidate)
            rule = previous.rules(actual, predicted, incumbent, fraction)
            branch_trust = bool(
                fixed._model_decrease_is_trusted(
                    jnp.asarray(branch_predicted["merit"]),
                    jnp.asarray(actual["merit"]),
                    jnp.asarray(incumbent["merit"]),
                    jnp.asarray(predicted_incumbent),
                )
            )
            row = {
                "fraction": fraction,
                "actual": actual,
                "predicted": predicted,
                "rules": rule,
                "branch_predicted": branch_predicted,
                "branch_predicted_incumbent": predicted_incumbent,
                "branch_model_trust": branch_trust,
                "branch_model_accepted": rule["sufficient_merit"]
                and rule["strict_residual_decrease"]
                and branch_trust,
            }
            if details:
                diag = diagnostic(candidate, op)
                row.update(
                    {
                        "confined_flip_indices": changed(
                            base_diag["confined"], diag["confined"]
                        ),
                        "support_change_indices": changed(
                            base_diag["support"], diag["support"]
                        ),
                        "shadow_flip_indices": changed(
                            base_diag["shadow"], diag["shadow"]
                        ),
                        "diagnostics": diag,
                        "state": candidate,
                    }
                )
            rows.append(row)
        return rows

    qualified = newton(state, op, external, target)
    jax.block_until_ready(qualified)
    data["newton_receipt"] = qualified._asdict()
    for name, direction in (
        ("analytic", jnp.asarray(analytic) - state),
        ("newton", qualified.step),
        ("map_defect", image - state),
    ):
        data["directions"][name] = {"direction": direction, "ladder": ladder(direction)}
    expected = json.loads(
        (
            ROOT
            / "docs/figures/cut-cell-current-attribution/merit-model"
            / f"cells-{len(machine.node)}.json"
        ).read_text()
    )
    for name in ("analytic", "newton", "map_defect"):
        np.testing.assert_allclose(
            [r["actual"]["merit"] for r in data["directions"][name]["ladder"]],
            [r["actual"]["merit"] for r in expected["directions"][name]["ladder"]],
            rtol=2e-8,
            atol=2e-12,
        )
    assert all(
        r["actual"]["merit"] >= incumbent["merit"]
        for r in data["directions"]["newton"]["ladder"]
    )
    assert data["directions"]["analytic"]["ladder"][0]["rules"]["refusing_rules"] == [
        "model trust"
    ]
    positive = np.asarray(base_diag["confined"]).copy()
    positive[0] = ~positive[0]
    data["controls"] = {
        "prior_ladders_reproduced": True,
        "classification_single_flip": len(changed(base_diag["confined"], positive)),
        "known_nonzero_current_cells": int(np.sum(base_diag["support"])),
    }
    assert data["controls"]["classification_single_flip"] == 1
    assert data["controls"]["known_nonzero_current_cells"] > 0
    checkpoint(
        "REPRODUCED both controls: every Newton fraction nondecreasing; "
        "full analytic sufficient decrease passes but model trust refuses"
    )

    origin_geometry = geometry(state, op)
    frozen_image = frozen_program(state, origin_geometry, op, external, target)
    error = previous.maximum(frozen_image - image)
    data["controls"]["frozen_diagonal_error_wb"] = error
    np.testing.assert_allclose(frozen_image, image, rtol=2e-12, atol=2e-12)
    frozen_step = frozen_newton(state, origin_geometry, op, external, target)
    jax.block_until_ready(frozen_step)
    data["directions"]["branch_frozen"] = {
        "direction": frozen_step.step,
        "linear_solve": frozen_step._asdict(),
        "ladder": ladder(frozen_step.step, branch_geom=origin_geometry),
    }
    data["controls"]["frozen_direction_minus_map_defect_wb"] = previous.maximum(
        frozen_step.step - (image - state)
    )
    checkpoint("BRANCH_FROZEN measured with confined polygons fixed")

    provisional = state + qualified.step
    proposed_geometry = geometry(provisional, op)
    geometry_image = frozen_program(state, proposed_geometry, op, external, target)
    proposal_diagonal = frozen_program(
        provisional, proposed_geometry, op, external, target
    )
    np.testing.assert_allclose(
        proposal_diagonal, mapped(provisional), rtol=2e-12, atol=2e-12
    )
    continuation_step = frozen_newton(state, proposed_geometry, op, external, target)
    jax.block_until_ready(continuation_step)
    data["directions"]["clip_geometry_continuation"] = {
        "direction": continuation_step.step,
        "linear_solve": continuation_step._asdict(),
        "support_proposal": provisional,
        "support_proposal_fraction": 1.0,
        "fixed_flux_geometry_image": geometry_image,
        "geometry_jump_wb": previous.maximum(geometry_image - frozen_image),
        "geometry_confined_flips": changed(
            base_diag["confined"], diagnostic(provisional, op)["confined"]
        ),
        "ladder": ladder(continuation_step.step, branch_geom=proposed_geometry),
    }
    checkpoint(
        "CLIP_GEOMETRY_CONTINUATION measured after Newton-endpoint support update"
    )

    census_cache = {}

    def classify_fraction(fraction):
        fraction = float(fraction)
        if fraction not in census_cache:
            masks = np.asarray(classification(state + fraction * qualified.step, op))
            census_cache[fraction] = (np.packbits(masks).tobytes().hex(), masks)
        return census_cache[fraction][0]

    def census(intervals):
        grid = np.linspace(0, 1, intervals + 1)
        for t in grid:
            classify_fraction(t)
        pending = [
            (float(a), float(b))
            for a, b in zip(grid[:-1], grid[1:], strict=True)
            if classify_fraction(a) != classify_fraction(b)
        ]
        transitions = []
        while pending:
            a, b = pending.pop()
            if b - a <= 1e-6:
                transitions.append((a, b))
                continue
            m = (a + b) / 2
            key = classify_fraction(m)
            if classify_fraction(a) != key:
                pending.append((a, m))
            if key != classify_fraction(b):
                pending.append((m, b))
        return transitions

    transitions = census(256)
    coarse_keys = {item[0] for item in census_cache.values()}
    transitions = census(512)
    fine_keys = {item[0] for item in census_cache.values()}
    representatives = {}
    # Prefer ordinary grid interiors over one-sided transition endpoints.
    for t in sorted(
        census_cache, key=lambda t: (abs(t * 512 - round(t * 512)) > 1e-12, t)
    ):
        key = classify_fraction(t)
        representatives.setdefault(key, t)
    data["branch_census"] = {
        "coarse_branches": len(coarse_keys),
        "fine_branches": len(fine_keys),
        "additional_after_doubling": len(fine_keys - coarse_keys),
        "sample_count": len(census_cache),
        "transition_tolerance": 1e-6,
        "transitions": sorted(transitions),
        "samples": [
            {"fraction": t, "classification": v[0]}
            for t, v in sorted(census_cache.items())
        ],
        "predictors": [],
        "empirical_not_exhaustive_proof": True,
    }
    checkpoint(
        f"BRANCH_CENSUS coarse={len(coarse_keys)} fine={len(fine_keys)} "
        f"samples={len(census_cache)}"
    )
    best = None
    best_key = None
    best_origin = None
    for index, (key, t) in enumerate(
        sorted(representatives.items(), key=lambda item: item[1])
    ):
        origin = state + t * qualified.step
        proposal = newton(origin, op, external, target)
        jax.block_until_ready(proposal)
        endpoint = origin + proposal.step
        direction = endpoint - state
        finite = bool(jnp.all(jnp.isfinite(direction)))
        item = {
            "index": index,
            "origin_fraction": t,
            "origin_classification": key,
            "linear_solve": proposal._asdict(),
            "finite": finite,
        }
        if finite:
            endpoint_key = (
                np.packbits(np.asarray(classification(endpoint, op))).tobytes().hex()
            )
            rows = ladder(direction, branch_origin=origin, details=False)
            item.update(
                {
                    "endpoint_classification": endpoint_key,
                    "branch_consistent": key == endpoint_key,
                    "ladder": rows,
                    "direction": direction,
                }
            )
            accepted = [r for r in rows if r["rules"]["accepted_if_considered"]]
            finite_rows = [r for r in rows if np.isfinite(r["actual"]["merit"])]
            if finite_rows:
                pick = min(accepted or finite_rows, key=lambda r: r["actual"]["merit"])
                rank = (not bool(accepted), pick["actual"]["merit"])
                if best_key is None or rank < best_key:
                    best_key, best, best_origin = rank, item, origin
        data["branch_census"]["predictors"].append(item)
        if index % 25 == 0:
            checkpoint(f"PREDICTORS {index + 1}/{len(representatives)} best={best_key}")
    assert best is not None
    data["directions"]["multi_branch"] = {
        "direction": best["direction"],
        "selected_predictor": best["index"],
        "selected_origin_fraction": best["origin_fraction"],
        "branch_consistent": best["branch_consistent"],
        "ladder": ladder(jnp.asarray(best["direction"]), branch_origin=best_origin),
    }
    checkpoint("MULTI_BRANCH all encountered predictors scored")

    for name, group in data["directions"].items():
        direction = jnp.asarray(group["direction"])
        score = fixed._backtracking_scores(
            mapped,
            lambda v: image + tangent(state, v - state),
            state,
            direction,
            jnp.asarray(incumbent["merit"]),
            True,
            own_mask_acceptance=True,
        )
        jax.block_until_ready(score)
        group["production_selector"] = {
            k: v for k, v in score._asdict().items() if k != "candidates"
        }
    data["completed"] = True
    checkpoint("COMPLETED numerical measurement")
    draw(data, receipt, out)
    checkpoint("COMPLETED figures")


def draw(data, receipt, output):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from nova.media import poloidal
    from nova.media.ink import DEFAULT_INK, poloidal_axes, trace_axes

    colors = {
        "analytic": "#2266aa",
        "newton": "#777777",
        "branch_frozen": "#c28b20",
        "clip_geometry_continuation": "#8b4092",
        "multi_branch": "#168575",
    }
    fig, ax = plt.subplots(figsize=(8, 5), layout="constrained")
    for name, color in colors.items():
        rows = data["directions"][name]["ladder"][::-1]
        ax.loglog(
            [r["fraction"] for r in rows],
            [r["actual"]["merit"] for r in rows],
            marker="o",
            markersize=3,
            color=color,
            label=name.replace("_", " "),
            linewidth=DEFAULT_INK.trace_linewidth,
        )
    ax.axhline(
        data["incumbent"]["merit"],
        color="#333333",
        linestyle=":",
        linewidth=1,
        label="incumbent",
    )
    trace_axes(ax)
    ax.set_xlabel("fraction of direction")
    ax.set_ylabel("production eighth-norm merit")
    ax.legend(frameon=False, fontsize=9)
    for ext in ("png", "svg"):
        fig.savefig(output / f"merit-cells-{data['realised_cells']}.{ext}", dpi=170)
    plt.close(fig)
    candidates = [
        (name, r)
        for name in ("branch_frozen", "clip_geometry_continuation", "multi_branch")
        for r in data["directions"][name]["ladder"]
        if np.isfinite(r["actual"]["merit"])
    ]
    accepted = [(n, r) for n, r in candidates if r["rules"]["accepted_if_considered"]]
    name, chosen = min(
        accepted or candidates, key=lambda pair: pair[1]["actual"]["merit"]
    )
    data["panel_candidate"] = {
        "generator": name,
        "fraction": chosen["fraction"],
        "accepted": bool(accepted),
        "actual": chosen["actual"],
    }
    coordinates = np.c_[receipt["coordinates_r_m"], receipt["coordinates_z_m"]]
    wall = np.c_[receipt["wall_r_m"], receipt["wall_z_m"]]
    radial, height, field = previous.certificate._raster_field(
        coordinates, np.asarray(receipt["analytic_state"]), wall
    )
    _, _, trial = previous.certificate._raster_field(
        coordinates, np.asarray(chosen["state"]), wall
    )
    record = receipt["analytic_input"]["record"]
    levels = poloidal.contour_levels(
        field, 12, axis=record["axis_flux_wb"], boundary=record["boundary_flux_wb"]
    )
    data["panel_levels_wb"] = levels
    fig, ax = plt.subplots(figsize=(6, 7), layout="constrained")
    for scalar, color in (
        (field, previous.certificate.ANALYTIC_INK_COLOR),
        (trial, previous.certificate.SOLVED_INK_COLOR),
    ):
        poloidal.draw_flux_contours(ax, radial, height, scalar, levels, color=color)
    poloidal.draw_wall(ax, units=(wall,))
    for axis, saddles, color, size in (
        (
            receipt["analytic_axis_rz_m"],
            receipt["analytic_x_points_rz_m"],
            previous.certificate.ANALYTIC_INK_COLOR,
            8,
        ),
        (
            chosen["diagnostics"]["axis"],
            chosen["diagnostics"]["x_point"],
            previous.certificate.SOLVED_INK_COLOR,
            4,
        ),
    ):
        poloidal.draw_nulls(
            ax,
            magnetic_axis=axis,
            x_points=saddles,
            style=DEFAULT_INK.variant(
                axis_color=color,
                xpoint_color=color,
                axis_marker="^",
                axis_markersize=size,
                xpoint_marker="X",
            ),
            contain=(wall,),
        )
    poloidal_axes(ax)
    converged = chosen["actual"]["relative_sup"] <= data["locked_residual_bound"]
    ax.set_title(
        f"{name.replace('_', ' ')} · fraction {chosen['fraction']:g}\n"
        f"{'accepted' if accepted else 'best rejected'} · "
        f"residual {chosen['actual']['relative_sup']:.7g}\n"
        f"converged: {converged}",
        fontsize=10,
    )
    for ext in ("png", "svg"):
        fig.savefig(output / f"candidate-cells-{data['realised_cells']}.{ext}", dpi=170)
    plt.close(fig)


if __name__ == "__main__":
    main()
