"""Measure a local flux model against finite moves of its current support."""
# ruff: noqa: E501 -- Persisted diagnostic text and captions retain literal wording.

from __future__ import annotations

import argparse
from contextlib import nullcontext
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
from time import perf_counter

import numpy as np

from nova.jax.config import configure_dtypes

configure_dtypes()
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

from benchmarks import solovev_certificate as certificate  # noqa: E402
from nova.equilibrium import fixed_point as fixed  # noqa: E402
from nova.equilibrium.stencil_mesh import CellCurrentMoments  # noqa: E402
from scripts.analytic_oracle_fixtures import measure as fixture  # noqa: E402

REFERENCE = Path(
    "/home/ITER/mcintos/.config/reckon/crew/reports/nova/s19-codex-20260916/"
    "cca-why-the-certificate-rows-do-not-converge-after-the-clip"
)
sys.path.insert(0, str(REFERENCE))
from instrument_boundary_band import _build  # noqa: E402

FRACTIONS = (1.0, 0.5, 0.25, 0.125, 0.0625, 0.01, 0.001)


def clean(value):
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [clean(v) for v in value]
    if isinstance(value, (np.ndarray, jax.Array)):
        return clean(np.asarray(value).tolist())
    if isinstance(value, np.generic):
        return clean(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def write(path, data):
    path.write_text(json.dumps(clean(data), indent=2, allow_nan=False) + "\n")


def maximum(value):
    return float(np.max(np.abs(value)))


def digest(value):
    return hashlib.sha256(np.asarray(value, dtype=np.float64).tobytes()).hexdigest()


def norm_terms(image, state):
    residual = np.asarray(image) - np.asarray(state)
    image = np.asarray(image)
    numerator = float(np.linalg.norm(residual, ord=8))
    denominator = float(np.linalg.norm(np.r_[image, 1e-30], ord=8))
    return {
        "numerator_wb": numerator,
        "denominator_wb": denominator,
        "merit": numerator / denominator,
        "relative_sup": maximum(residual) / max(maximum(image), 1e-30),
        "residual_eighth_power": residual**8,
        "image_eighth_power": image**8,
    }


def rules(actual, predicted, incumbent, factor):
    sufficient_merit = actual["merit"] <= incumbent["merit"] * (
        1 - fixed._SUFFICIENT_DECREASE_SLOPE * factor
    )
    sufficient_residual = actual["relative_sup"] < incumbent["relative_sup"]
    trusted = bool(
        fixed._model_decrease_is_trusted(
            jnp.asarray(predicted["merit"]),
            jnp.asarray(actual["merit"]),
            jnp.asarray(incumbent["merit"]),
        )
    )
    actual_drop = incumbent["merit"] - actual["merit"]
    predicted_drop = incumbent["merit"] - predicted["merit"]
    failures = []
    if not sufficient_merit or not sufficient_residual:
        failures.append("sufficient decrease")
    if not trusted:
        failures.append("model trust")
    return {
        "sufficient_merit": sufficient_merit,
        "strict_residual_decrease": sufficient_residual,
        "model_trust": trusted,
        "accepted_if_considered": not failures,
        "refusing_rules": failures,
        "actual_decrease": actual_drop,
        "predicted_decrease": predicted_drop,
        "actual_over_predicted_decrease": actual_drop / predicted_drop
        if predicted_drop != 0
        else None,
        "fallback_same_direction_refused": bool(failures),
        "fallback_note": "Continuation uses these same inequalities; production continuation uses the map defect, measured separately.",
    }


def draw_ladder(data, output):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from nova.media.ink import DEFAULT_INK

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), layout="constrained")
    for ax, direction in zip(axes, ("analytic", "newton"), strict=True):
        rows = data["directions"][direction]["ladder"]
        order = np.argsort([r["fraction"] for r in rows])
        x = np.array([r["fraction"] for r in rows])[order]
        a = np.array([r["actual"]["merit"] for r in rows])[order]
        p = np.array([r["predicted"]["merit"] for r in rows])[order]
        ax.loglog(
            x,
            a,
            "o-",
            color=certificate.SOLVED_INK_COLOR,
            label="actual",
            linewidth=DEFAULT_INK.trace_linewidth,
            markersize=DEFAULT_INK.trace_markersize,
        )
        ax.loglog(
            x, p, "s--", color=certificate.ANALYTIC_INK_COLOR, label="linear model"
        )
        for i in order:
            r = rows[i]
            label = (
                "+".join(
                    {"sufficient decrease": "D", "model trust": "T"}[v]
                    for v in r["rules"]["refusing_rules"]
                )
                or "accept"
            )
            ax.annotate(
                label,
                (r["fraction"], r["actual"]["merit"]),
                xytext=(0, -14),
                textcoords="offset points",
                ha="center",
                fontsize=8,
            )
        ax.set_xlabel("fraction of direction")
        ax.set_ylabel("eighth-norm merit")
        ax.set_title(direction + " direction")
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(False)
        ax.legend(frameon=False)
    fig.suptitle(
        f"{data['realised_cells']} cells · D: sufficient decrease · T: model trust"
    )
    for extension in ("png", "svg"):
        fig.savefig(
            output / f"merit-cells-{data['realised_cells']}.{extension}", dpi=160
        )
    plt.close(fig)


def draw_fields(data, receipt, output):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from nova.media import poloidal
    from nova.media.ink import DEFAULT_INK, poloidal_axes

    coordinates = np.c_[receipt["coordinates_r_m"], receipt["coordinates_z_m"]]
    wall = np.c_[receipt["wall_r_m"], receipt["wall_z_m"]]
    reference = np.asarray(receipt["analytic_state"])
    radial, height, field = certificate._raster_field(coordinates, reference, wall)
    record = receipt["analytic_input"]["record"]
    levels = poloidal.contour_levels(
        field, 12, axis=record["axis_flux_wb"], boundary=record["boundary_flux_wb"]
    )
    fig, axes = plt.subplots(1, 2, figsize=(11, 7), layout="constrained")
    full = data["directions"]["analytic"]["ladder"][0]
    for ax, title, state, nulls, merit, residual in (
        (
            axes[0],
            "stalled terminal",
            data["state"],
            data["base_diagnostics"],
            data["incumbent"]["merit"],
            data["incumbent"]["relative_sup"],
        ),
        (
            axes[1],
            "full analytic-direction candidate",
            reference,
            full["diagnostics"],
            full["actual"]["merit"],
            full["actual"]["relative_sup"],
        ),
    ):
        _, _, trial = certificate._raster_field(coordinates, np.asarray(state), wall)
        for scalar, color in (
            (field, certificate.ANALYTIC_INK_COLOR),
            (trial, certificate.SOLVED_INK_COLOR),
        ):
            poloidal.draw_flux_contours(ax, radial, height, scalar, levels, color=color)
        poloidal.draw_wall(ax, units=(wall,))
        for axis, saddles, color, size in (
            (
                receipt["analytic_axis_rz_m"],
                receipt["analytic_x_points_rz_m"],
                certificate.ANALYTIC_INK_COLOR,
                8,
            ),
            (nulls["axis"], nulls["x_point"], certificate.SOLVED_INK_COLOR, 4),
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
        ax.set_title(
            f"{title}\nmerit {merit:.7g}; residual {residual:.7g}\nconverged: {residual <= certificate.TERMINAL_RESIDUAL_BOUND}",
            fontsize=10,
        )
    fig.suptitle(
        "135 cells · reference blue / candidate ochre · shared Wb contours and both axes"
    )
    for extension in ("png", "svg"):
        fig.savefig(output / f"candidate-cells-135.{extension}", dpi=160)
    plt.close(fig)
    return levels


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cells", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    start = perf_counter()
    assert jax.config.jax_enable_x64 and jax.default_backend() == "cpu"
    fixture._cache_lock = lambda store: nullcontext(0.0)

    def refuse_store(*_args, **_kwargs):
        raise RuntimeError("Fixture cache miss requires an out-of-scope cache write")

    fixture.ZarrStore.store = refuse_store
    (
        machine,
        operator,
        profile,
        exact,
        seed,
        target,
        seed_receipt,
        coordinates,
        analytic,
    ) = _build("weak-rotation-reactor-static", -args.cells)
    print(
        f"BUILT {args.cells} realised={len(machine.node)} elapsed={perf_counter() - start:.2f}",
        flush=True,
    )
    root = Path(__file__).resolve().parents[4]
    compact_path = (
        root
        / "docs/figures/cut-cell-current-attribution/continuous-support/rows-titan-adjoint/record"
        / f"cells-{args.cells}/receipt.json"
    )
    compact = json.loads(compact_path.read_text())
    receipt_path = Path(compact["raw_receipt"])
    assert (
        hashlib.sha256(receipt_path.read_bytes()).hexdigest() == compact["raw_sha256"]
    )
    receipt = json.loads(receipt_path.read_text())
    state = jnp.asarray(receipt["trips"][-1]["flux"], dtype=jnp.float64)
    assert digest(state) == compact["terminal"]["state_digest"]
    np.testing.assert_array_equal(analytic, receipt["analytic_state"])
    external = operator.external()
    target = jnp.asarray(target, dtype=jnp.float64)
    program = jax.jit(operator.traced_flux_map(target_current=target))

    def mapped(value):
        return program(value, external, operator, target)

    @jax.jit
    def map_jvp(value, vector):
        return jax.jvp(mapped, (value,), (vector,))[1]

    @jax.jit
    def moment_program(value, op):
        raw = jnp.stack(op.cell_current_moments(value))
        amplitude = op.current_normalisation_amplitude(target, raw[0].sum())
        return raw, raw * amplitude

    @jax.jit
    def moment_jvp(value, vector, op):
        return jax.jvp(lambda v: moment_program(v, op), (value,), (vector,))[1]

    @jax.jit
    def image_program(moments, op):
        return op.current_moment_image(CellCurrentMoments(*moments))

    @jax.jit
    def diagnostic_program(value, op):
        masks, topology, _sample, support = op._support_partition(value)
        return {
            "confined": masks.confined_profile,
            "open": masks.open_field_line,
            "labels": masks.label,
            "support_area": support.area,
            "axis": topology.axis,
            "x_point": topology.x_point,
            "axis_flux": topology.axis_flux,
            "boundary_flux": topology.boundary_flux,
            "shadow": op.residual_shadow_mask(value),
        }

    @jax.jit
    def newton_program(value, op, ext, target_value):
        image, tangent = jax.linearize(
            lambda v: program(v, ext, op, target_value), value
        )
        return fixed._qualified_krylov_step(
            lambda v: v - tangent(v),
            image - value,
            fixed._relative_residual(image, value),
            gmres_iterations=30,
            condition_ratio_limit=math.e,
            preceding_condition_baseline=jnp.asarray(jnp.nan),
        )

    image = mapped(state)
    raw_base, scaled_base = moment_program(state, operator)
    base_diag = diagnostic_program(state, operator)
    incumbent = norm_terms(image, state)
    data = {
        "revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "worktree": str(root),
        "backend": jax.default_backend(),
        "devices": str(jax.devices()),
        "realised_cells": len(machine.node),
        "requested_cells": args.cells,
        "input_receipt": str(receipt_path),
        "input_sha256": compact["raw_sha256"],
        "state_digest": digest(state),
        "state": state,
        "incumbent": incumbent,
        "base_diagnostics": base_diag,
        "raw_moments": raw_base,
        "scaled_moments": scaled_base,
        "reconstruction_error_wb": maximum(
            image - external - image_program(scaled_base, operator)
        ),
        "directions": {},
        "completed": False,
        "locked_residual_bound": certificate.TERMINAL_RESIDUAL_BOUND,
        "characteristic_pitch_m": compact["characteristic_pitch_m"],
    }
    path = output / f"cells-{len(machine.node)}.json"
    write(path, data)
    print(
        f"BASE merit={incumbent['merit']:.12g} residual={incumbent['relative_sup']:.12g}",
        flush=True,
    )

    # Replay the exact historical witness before diagnosing the later terminals.
    if args.cells == 110:
        historical_path = REFERENCE / "record/cells-110/receipt.json"
        historical = jnp.asarray(
            json.loads(historical_path.read_text())["trips"][-1]["flux"]
        )
        hi = mapped(historical)
        witness = {}
        for name, vector in (
            ("map_defect", hi - historical),
            ("toward_analytic", jnp.asarray(analytic) - historical),
        ):
            score = fixed._backtracking_scores(
                mapped,
                lambda v: hi + map_jvp(historical, v - historical),
                historical,
                vector,
                fixed._smooth_relative_sup_merit(hi, historical),
                True,
                own_mask_acceptance=True,
            )
            jax.block_until_ready(score)
            witness[name] = {
                k: getattr(score, k) for k in score._fields if k != "candidates"
            }
        expected = json.loads(
            (
                root
                / "docs/figures/cut-cell-current-attribution/continuous-support/merit-scores.json"
            ).read_text()
        )
        for name in witness:
            for key in ("merits", "predicted_merits"):
                np.testing.assert_allclose(
                    witness[name][key], expected[name][key], rtol=2e-9, atol=2e-12
                )
        assert not bool(witness["toward_analytic"]["ladder_accepted"])
        assert bool(witness["map_defect"]["ladder_accepted"])
        data["historical_reproduction"] = {
            "path": str(historical_path),
            "scores": witness,
            "passed": True,
        }
        vector = jnp.asarray(analytic) - historical
        derivative = map_jvp(historical, vector)
        hr, hs = moment_program(historical, operator)
        hrd, hsd = moment_jvp(historical, vector, operator)
        ar, ass = moment_program(jnp.asarray(analytic), operator)
        full_error = ass - hs - hsd
        components = jnp.stack(
            [
                image_program(
                    jnp.zeros_like(full_error).at[i].set(full_error[i]), operator
                )
                for i in range(3)
            ]
        )
        actual_terms = norm_terms(mapped(jnp.asarray(analytic)), jnp.asarray(analytic))
        predicted_terms = norm_terms(hi + derivative, jnp.asarray(analytic))
        hd = diagnostic_program(historical, operator)
        ad = diagnostic_program(jnp.asarray(analytic), operator)
        curvature = []
        for fraction in (0.0, *FRACTIONS):
            at = historical + fraction * vector
            for h in (1e-4, 1e-5):
                second = (
                    map_jvp(at + h * vector, vector) - map_jvp(at - h * vector, vector)
                ) / (2 * h)
                curvature.append(
                    {
                        "fraction": fraction,
                        "h": h,
                        "map_second_directional_max_wb": maximum(second),
                    }
                )
        data["historical_reproduction"]["decomposition"] = {
            "actual": actual_terms,
            "predicted": predicted_terms,
            "map_error_wb": mapped(jnp.asarray(analytic)) - hi - derivative,
            "coupling_component_error_max_wb": [maximum(c) for c in components],
            "coupling_remainder_max_wb": maximum(
                mapped(jnp.asarray(analytic)) - hi - derivative - components.sum(axis=0)
            ),
            "raw_current_error_max_a": maximum(ar[0] - hr[0] - hrd[0]),
            "scaled_current_error_max_a": maximum(full_error[0]),
            "per_cell_raw_current_error_a": ar[0] - hr[0] - hrd[0],
            "per_cell_scaled_current_error_a": full_error[0],
            "confined_flip_indices": np.flatnonzero(
                np.asarray(hd["confined"] != ad["confined"])
            ),
            "shadow_flip_indices": np.flatnonzero(
                np.asarray(hd["shadow"] != ad["shadow"])
            ),
            "curvature": curvature,
            "tangent_fd_relative_error": float(
                jnp.linalg.norm(
                    (
                        mapped(historical + 1e-6 * vector)
                        - mapped(historical - 1e-6 * vector)
                    )
                    / 2e-6
                    - derivative
                )
                / jnp.linalg.norm(derivative)
            ),
        }
        write(path, data)
        print(
            "REPRODUCED actual=9.62674026e-5 predicted=0.348071531 full analytic refused; half defect accepted",
            flush=True,
        )

    qualified = newton_program(state, operator, external, target)
    jax.block_until_ready(qualified)
    data["newton_receipt"] = {k: getattr(qualified, k) for k in qualified._fields}
    data["newton_scope"] = (
        "Production state-local qualified GMRES(30) with no preceding condition baseline or history-dependent cap; not a replay of the solver's hidden globalization carry."
    )
    print(
        "NEWTON "
        + json.dumps(
            clean(
                {k: getattr(qualified, k) for k in qualified._fields if "step" not in k}
            )
        ),
        flush=True,
    )
    write(path, data)
    for name, direction in (
        ("analytic", jnp.asarray(analytic) - state),
        ("newton", qualified.step),
        ("map_defect", image - state),
    ):
        derivative = map_jvp(state, direction)
        raw_dot, scaled_dot = moment_jvp(state, direction, operator)
        fd = (
            mapped(state + 1e-6 * direction) - mapped(state - 1e-6 * direction)
        ) / 2e-6
        group = {
            "direction": direction,
            "map_jvp_wb": derivative,
            "raw_moment_jvp": raw_dot,
            "scaled_moment_jvp": scaled_dot,
            "tangent_fd_relative_error": float(
                jnp.linalg.norm(fd - derivative)
                / jnp.maximum(jnp.linalg.norm(fd), 1e-30)
            ),
            "moment_image_jvp_identity_error_wb": maximum(
                derivative - image_program(scaled_dot, operator)
            ),
            "ladder": [],
        }
        data["directions"][name] = group
        for fraction in FRACTIONS:
            candidate = state + fraction * direction
            actual_image = mapped(candidate)
            predicted_image = image + fraction * derivative
            actual = norm_terms(actual_image, candidate)
            predicted = norm_terms(predicted_image, candidate)
            raw, scaled = moment_program(candidate, operator)
            raw_linear = raw_base + fraction * raw_dot
            scaled_linear = scaled_base + fraction * scaled_dot
            normalised_raw_linear = target * raw_linear / raw_linear[0].sum()
            diag = diagnostic_program(candidate, operator)
            error_moments = scaled - scaled_linear
            component_images = []
            for component in range(3):
                selected = (
                    jnp.zeros_like(error_moments)
                    .at[component]
                    .set(error_moments[component])
                )
                component_images.append(image_program(selected, operator))
            component_images = jnp.stack(component_images)
            curvature = []
            if name != "map_defect":
                for h in (1e-4, 1e-5):
                    plus, minus = candidate + h * direction, candidate - h * direction
                    second = (map_jvp(plus, direction) - map_jvp(minus, direction)) / (
                        2 * h
                    )
                    second_raw = (
                        moment_jvp(plus, direction, operator)[0]
                        - moment_jvp(minus, direction, operator)[0]
                    ) / (2 * h)
                    dp, dm = (
                        diagnostic_program(plus, operator),
                        diagnostic_program(minus, operator),
                    )
                    curvature.append(
                        {
                            "h": h,
                            "map_second_directional_max_wb": maximum(second),
                            "raw_current_second_directional_max_a": maximum(
                                second_raw[0]
                            ),
                            "local_confined_flips": int(
                                jnp.count_nonzero(dp["confined"] != dm["confined"])
                            ),
                            "map_second_directional": second,
                        }
                    )
            row = {
                "fraction": fraction,
                "actual": actual,
                "predicted": predicted,
                "predicted_over_actual": predicted["merit"] / actual["merit"],
                "rules": rules(actual, predicted, incumbent, fraction),
                "diagnostics": diag,
                "confined_flip_indices": np.flatnonzero(
                    np.asarray(diag["confined"] != base_diag["confined"])
                ),
                "open_flip_indices": np.flatnonzero(
                    np.asarray(diag["open"] != base_diag["open"])
                ),
                "shadow_flip_indices": np.flatnonzero(
                    np.asarray(diag["shadow"] != base_diag["shadow"])
                ),
                "nonzero_current_flip_indices": np.flatnonzero(
                    np.asarray((raw[0] != 0) != (raw_base[0] != 0))
                ),
                "raw_moments": raw,
                "raw_linear_moments": raw_linear,
                "scaled_moments": scaled,
                "scaled_linear_moments": scaled_linear,
                "raw_moment_error_by_component": raw - raw_linear,
                "scaled_moment_error_by_component": error_moments,
                "raw_current_error_max_a": maximum(raw[0] - raw_linear[0]),
                "scaled_current_error_max_a": maximum(error_moments[0]),
                "coupling_component_error_wb": component_images,
                "coupling_component_error_max_wb": [
                    maximum(v) for v in component_images
                ],
                "coupling_remainder_max_wb": maximum(
                    actual_image - predicted_image - component_images.sum(axis=0)
                ),
                "raw_curvature_image_max_wb": maximum(
                    image_program(scaled - normalised_raw_linear, operator)
                ),
                "normalisation_curvature_image_max_wb": maximum(
                    image_program(normalised_raw_linear - scaled_linear, operator)
                ),
                "map_prediction_error_max_wb": maximum(actual_image - predicted_image),
                "actual_numerator_predicted_denominator_merit": actual["numerator_wb"]
                / predicted["denominator_wb"],
                "predicted_numerator_actual_denominator_merit": predicted[
                    "numerator_wb"
                ]
                / actual["denominator_wb"],
                "curvature": curvature,
            }
            group["ladder"].append(row)
            write(path, data)
            print(
                f"ROW {name} fraction={fraction:g} actual={actual['merit']:.10g} predicted={predicted['merit']:.10g} flips={len(row['confined_flip_indices'])} refuses={row['rules']['refusing_rules']}",
                flush=True,
            )
        # A nonzero map defect and a seeded bit flip establish instrument sensitivity.
        changed = np.array(base_diag["confined"], copy=True)
        changed[0] = ~changed[0]
        group["positive_controls"] = {
            "classification_detector_single_flip": int(
                np.count_nonzero(changed != np.asarray(base_diag["confined"]))
            ),
            "map_jvp_max_wb": maximum(derivative),
            "current_nonzero_cells": int(np.count_nonzero(np.asarray(raw_base[0]))),
            "moment_image_control_wb": maximum(image_program(scaled_base, operator)),
        }
    data["wall_seconds"] = perf_counter() - start
    data["completed"] = True
    write(path, data)
    draw_ladder(data, output)
    if args.cells == 110:
        data["panel_levels_wb"] = draw_fields(data, receipt, output)
    write(path, data)
    print(
        f"COMPLETED cells={len(machine.node)} seconds={perf_counter() - start:.2f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
