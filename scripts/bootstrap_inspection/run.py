"""Run the gentle window with zeros and Sauter bootstrap-current closures."""

from __future__ import annotations

import csv
import dataclasses
import html
import json
import re
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from nova.transport import forward as forward_module
from scripts.window_demonstration import figures as demonstration_figures
from scripts.window_demonstration import run_window as demonstration


OUTPUT_DIRECTORY = Path(__file__).resolve().parent
PROFILE_PATH = OUTPUT_DIRECTORY / "profiles.tsv"
RECEIPT_PATH = OUTPUT_DIRECTORY / "receipt.json"
REPORT_PATH = OUTPUT_DIRECTORY / "report.md"
FIGURE_PATH = (
    OUTPUT_DIRECTORY.parent.parent
    / "docs"
    / "figures"
    / "flux-function-forward-transport"
    / "bootstrap-model-comparison.svg"
)
BASELINE_REPORT_PATH = OUTPUT_DIRECTORY.parent / "window_demonstration" / "report.md"

WINDOW_SECONDS = 0.0025
AUXILIARY_MULTIPLIER = 0.5
BASELINE_MODEL = "zeros"
INSPECTION_MODEL = "sauter"
OWNER_RULING_DATE = "2026-08-23"
MODEL_PATH = ("neoclassical", "bootstrap_current", "model_name")
SELECTOR_EQUIVALENCE_TOLERANCE = 1.0e-12
HISTORICAL_REPRODUCTION_TOLERANCE = 1.0e-9
CURRENT_TREE_SAME_CONFIG_CONTRACTION = 0.23619506586175637
CURRENT_TREE_SAME_CONFIG_DRIFT = 1.4484219379440333e-11
MEASURED_BACKEND_DRIFT = 1.7e-9


@dataclasses.dataclass(frozen=True)
class RunCapture:
    """Window outcome and terminal TORAX profiles for one closure choice."""

    model_name: str
    result: Any
    curves: tuple[demonstration_figures.Curve, ...]
    config: Mapping[str, Any]
    seconds: float


def _model_config(
    model_name: str,
    model_factory=demonstration._torax_model,
) -> dict[str, Any]:
    """Return the demonstration model config with one declared closure choice."""
    model = model_factory(WINDOW_SECONDS)
    config = forward_module._thaw_mapping(model.torax_config)
    neoclassical = dict(config.get("neoclassical", {}))
    bootstrap_current = dict(neoclassical.get("bootstrap_current", {}))
    bootstrap_current["model_name"] = model_name
    neoclassical["bootstrap_current"] = bootstrap_current
    config["neoclassical"] = neoclassical
    return config


def _validated_model_name(config: Mapping[str, Any]) -> str:
    """Resolve a source dictionary through TORAX validation without mutating it."""
    from torax._src.torax_pydantic.model_config import ToraxConfig

    validated = ToraxConfig.from_dict(forward_module._thaw_mapping(config))
    return validated.neoclassical.bootstrap_current.model_name


def _implicit_baseline_model() -> str:
    """Confirm the banked raw dictionary resolves its absent selector to zeros."""
    model = demonstration._torax_model(WINDOW_SECONDS)
    config = forward_module._thaw_mapping(model.torax_config)
    if "neoclassical" in config:
        raise RuntimeError(
            "the banked raw config unexpectedly carries a neoclassical subtree"
        )
    return _validated_model_name(config)


def _changed_leaves(left: Any, right: Any, prefix: tuple[str, ...] = ()) -> list[str]:
    """List value-level mapping differences using stable dotted paths."""
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        paths: list[str] = []
        for key in sorted(set(left) | set(right)):
            if key not in left or key not in right:
                paths.append(".".join((*prefix, str(key))))
            else:
                paths.extend(
                    _changed_leaves(left[key], right[key], (*prefix, str(key)))
                )
        return paths
    if isinstance(left, np.ndarray | list | tuple) or isinstance(
        right, np.ndarray | list | tuple
    ):
        equal = np.array_equal(np.asarray(left), np.asarray(right))
    else:
        equal = left == right
    return [] if equal else [".".join(prefix)]


def _run_configuration(shared: Mapping[str, Any], model_name: str | None) -> RunCapture:
    """Run one closure choice and replay its terminal TORAX interval directly."""
    from nova.jax.config import configure_dtypes
    from nova.transport.forward import TransportModel
    from torax._src.orchestration.run_simulation import run_simulation

    configure_dtypes()
    executions: list[Any] = []
    configs: list[Any] = []
    prepare_config = forward_module._prepare_torax_config
    run_steps = forward_module._run_torax_steps
    torax_model = demonstration._torax_model
    label = "implicit-default" if model_name is None else model_name

    def selected_model(window_length: float = demonstration.WINDOW_LENGTH_SECONDS):
        if window_length != WINDOW_SECONDS:
            raise RuntimeError(f"unexpected window length {window_length}")
        if model_name is None:
            return torax_model(window_length)
        return TransportModel(
            demonstration.TransportRung.TORAX_MULTI_CHANNEL,
            torax_config=_model_config(model_name),
        )

    def record_config(inputs):
        config = prepare_config(inputs)
        configs.append(config)
        return config

    def record_execution(config, durations, multiplier):
        execution = run_steps(config, durations, multiplier)
        executions.append(execution)
        return execution

    demonstration._torax_model = selected_model
    forward_module._prepare_torax_config = record_config
    forward_module._run_torax_steps = record_execution
    started = time.perf_counter()
    try:
        regime = demonstration.RegimeConfig(
            "gentle",
            WINDOW_SECONDS,
            AUXILIARY_MULTIPLIER,
            1,
        )
        result = demonstration._run_regime(regime, **shared)
    finally:
        demonstration._torax_model = torax_model
        forward_module._prepare_torax_config = prepare_config
        forward_module._run_torax_steps = run_steps
    seconds = time.perf_counter() - started

    if not result.converged:
        raise RuntimeError(
            f"{label} gentle window did not converge: "
            f"{result.outcome_type}: {result.outcome}"
        )
    if not executions or len(executions) != len(configs):
        raise RuntimeError(
            f"incomplete {label} capture: "
            f"executions={len(executions)}, configs={len(configs)}"
        )
    terminal_execution = executions[-1]
    terminal_config = configs[-1]
    direct_output, direct_history = run_simulation(
        terminal_config,
        progress_bar=False,
    )
    if direct_history.sim_error.name != "NO_ERROR":
        raise RuntimeError(
            f"{label} standalone replay failed with {direct_history.sim_error.name}"
        )
    curves = demonstration_figures._collect_curves(
        terminal_execution,
        terminal_config,
        direct_output,
    )
    bootstrap = tuple(
        curve for curve in curves if curve.quantity == "bootstrap_fraction"
    )
    if model_name is None:
        source_config = forward_module._thaw_mapping(
            torax_model(WINDOW_SECONDS).torax_config
        )
    else:
        source_config = _model_config(model_name)
    return RunCapture(
        model_name=label,
        result=result,
        curves=bootstrap,
        config=source_config,
        seconds=seconds,
    )


def _banked_contraction() -> float:
    """Read the current committed baseline contraction from its receipt report."""
    text = BASELINE_REPORT_PATH.read_text(encoding="utf-8")
    match = re.search(
        r"Measured gating-norm contraction estimate: `([^`]+)`",
        text,
    )
    if match is None:
        raise RuntimeError("the banked baseline contraction is unavailable")
    return float(match.group(1))


def _curve(capture: RunCapture, route: str, state: str):
    matches = [
        curve
        for curve in capture.curves
        if curve.route == route and curve.state == state
    ]
    if len(matches) != 1:
        raise RuntimeError(
            f"expected one {capture.model_name}/{route}/{state} curve, "
            f"found {len(matches)}"
        )
    return matches[0]


def _route_gap(capture: RunCapture) -> float:
    """Measure maximum facade-to-direct separation for both plotted states."""
    gap = 0.0
    for state in ("initial", "final"):
        facade = _curve(capture, "facade", state)
        direct = _curve(capture, "torax-standalone", state)
        values = np.interp(facade.rho, direct.rho, direct.value)
        gap = max(gap, float(np.max(np.abs(facade.value - values))))
    return gap


def _profile_extrema(capture: RunCapture) -> tuple[float, float]:
    values = np.concatenate([curve.value for curve in capture.curves])
    return float(np.min(values)), float(np.max(values))


def _verify_selector_equivalence(
    implicit: RunCapture,
    zeros: RunCapture,
    banked_contraction: float,
) -> None:
    """Prove implicit and explicit zeros agree before permitting Sauter."""
    implicit_contraction = float(implicit.result.convergence.contraction_estimate)
    explicit_contraction = float(zeros.result.convergence.contraction_estimate)
    selector_difference = abs(implicit_contraction - explicit_contraction)
    if selector_difference > SELECTOR_EQUIVALENCE_TOLERANCE:
        raise RuntimeError(
            "implicit and explicit zeros contractions differ by "
            f"{selector_difference}, above {SELECTOR_EQUIVALENCE_TOLERANCE}"
        )
    historical_difference = abs(explicit_contraction - banked_contraction)
    if historical_difference > HISTORICAL_REPRODUCTION_TOLERANCE:
        raise RuntimeError(
            "explicit zeros historical reproduction differs by "
            f"{historical_difference}, above {HISTORICAL_REPRODUCTION_TOLERANCE}"
        )
    if _validated_model_name(implicit.config) != BASELINE_MODEL:
        raise RuntimeError("the implicit baseline selector does not validate to zeros")
    if _validated_model_name(zeros.config) != BASELINE_MODEL:
        raise RuntimeError("the explicit baseline selector does not validate to zeros")
    for capture in (implicit, zeros):
        minimum, maximum = _profile_extrema(capture)
        if max(abs(minimum), abs(maximum)) > 1.0e-15:
            raise RuntimeError(
                f"the {capture.model_name} baseline has non-zero bootstrap current"
            )
    print(
        "selector_equivalence="
        f"implicit={implicit_contraction:.17g},"
        f"explicit={explicit_contraction:.17g},"
        f"absolute_difference={selector_difference:.17g},"
        f"historical_difference={historical_difference:.17g}",
        flush=True,
    )


def _write_profiles(captures: Sequence[RunCapture]) -> None:
    """Write every plotted value with its configuration and route identity."""
    with PROFILE_PATH.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=("model_name", "route", "state", "rho", "bootstrap_fraction"),
            delimiter="\t",
            lineterminator="\n",
        )
        writer.writeheader()
        for capture in captures:
            for curve in capture.curves:
                for rho, value in zip(curve.rho, curve.value, strict=True):
                    writer.writerow(
                        {
                            "model_name": capture.model_name,
                            "route": curve.route,
                            "state": curve.state,
                            "rho": f"{float(rho):.17g}",
                            "bootstrap_fraction": f"{float(value):.17g}",
                        }
                    )


def _polyline(curve, x_position, y_position, css_class: str, dash: str) -> str:
    points = " ".join(
        f"{x_position(float(rho)):.2f},{y_position(float(value)):.2f}"
        for rho, value in zip(curve.rho, curve.value, strict=True)
    )
    return (
        f'<polyline points="{points}" class="{css_class}" fill="none" '
        f'stroke-width="1.8" stroke-dasharray="{dash}" '
        'stroke-linejoin="round" stroke-linecap="round"/>'
    )


def _write_figure(zeros: RunCapture, sauter: RunCapture) -> None:
    """Draw a compact direct-labelled overlay matching the banked visual style."""
    left, right, top, bottom = 43.0, 218.0, 23.0, 202.0
    plotted = (
        (zeros, "final", "zero-curve", "4 3"),
        (sauter, "initial", "initial-curve", "none"),
        (sauter, "final", "final-curve", "none"),
    )
    values = np.concatenate(
        [_curve(capture, "facade", state).value for capture, state, _, _ in plotted]
    )
    lower = min(0.0, float(np.min(values)))
    upper = max(0.0, float(np.max(values)))
    padding = 0.08 * max(upper - lower, abs(upper), 1.0e-4)
    lower -= padding
    upper += padding

    def x_position(rho: float) -> float:
        return left + rho * (right - left)

    def y_position(value: float) -> float:
        return bottom - (value - lower) / (upper - lower) * (bottom - top)

    lines = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="320" height="238" '
        'viewBox="0 0 320 238" role="img" '
        'aria-labelledby="bootstrap_comparison_title bootstrap_comparison_desc">',
        '<title id="bootstrap_comparison_title">Zeros and Sauter '
        "bootstrap-current fractions</title>",
        '<desc id="bootstrap_comparison_desc">Owner-authorised non-identical '
        "Sauter inspection overlaid beside the banked zeros baseline for the "
        "identical gentle window. Circle markers show the direct TORAX replay."
        "</desc>",
        "<style>",
        (
            ".text{fill:#202124;font-family:system-ui,sans-serif}"
            ".muted{fill:#62666a;font-family:system-ui,sans-serif}"
            ".axis{stroke:#202124}.grid{stroke:#d4d6d8}"
            ".zero-curve{stroke:#8a8f94}.initial-curve{stroke:#62666a}"
            ".final-curve{stroke:#202124}.leader{stroke:#62666a}"
        ),
        (
            "@media (prefers-color-scheme:dark){"
            ".text{fill:#f1f3f4}.muted{fill:#c2c7cc}"
            ".axis{stroke:#f1f3f4}.grid{stroke:#4b5055}"
            ".zero-curve{stroke:#a2a8ae}.initial-curve{stroke:#c2c7cc}"
            ".final-curve{stroke:#f1f3f4}.leader{stroke:#c2c7cc}}"
        ),
        "</style>",
    ]
    for tick in np.linspace(lower, upper, 5):
        y = y_position(float(tick))
        lines.append(
            f'<line x1="{left:.2f}" y1="{y:.2f}" x2="{right:.2f}" '
            f'y2="{y:.2f}" class="grid" stroke-width="0.65"/>'
        )
        lines.append(
            f'<text x="{left - 5:.2f}" y="{y + 3:.2f}" class="text" '
            f'text-anchor="end" font-size="7.8">{tick:.3g}</text>'
        )
    lines.extend(
        (
            f'<line x1="{left}" y1="{top}" x2="{left}" y2="{bottom}" '
            'class="axis" stroke-width="0.9"/>',
            f'<line x1="{left}" y1="{bottom}" x2="{right}" '
            f'y2="{bottom}" class="axis" stroke-width="0.9"/>',
        )
    )
    for tick in (0.0, 0.25, 0.5, 0.75, 1.0):
        x = x_position(tick)
        lines.append(
            f'<text x="{x:.2f}" y="216" class="text" text-anchor="middle" '
            f'font-size="8">{tick:g}</text>'
        )
    for capture, state, css_class, dash in plotted:
        facade = _curve(capture, "facade", state)
        direct = _curve(capture, "torax-standalone", state)
        lines.append(_polyline(facade, x_position, y_position, css_class, dash))
        marker_indices = np.unique(np.linspace(0, direct.rho.size - 1, 5, dtype=int))
        for index in marker_indices:
            lines.append(
                f'<circle cx="{x_position(float(direct.rho[index])):.2f}" '
                f'cy="{y_position(float(direct.value[index])):.2f}" r="1.6" '
                f'class="{css_class}" fill="none" stroke-width="0.8"/>'
            )
    label_specs = (
        (zeros, "final", "zeros baseline", 0.55, 188.0),
        (sauter, "initial", "Sauter initial", 0.66, 82.0),
        (sauter, "final", "Sauter final", 0.76, 110.0),
    )
    for capture, state, label, probe, label_y in label_specs:
        curve = _curve(capture, "facade", state)
        target = float(np.interp(probe, curve.rho, curve.value))
        lines.append(
            f'<line x1="{x_position(probe):.2f}" '
            f'y1="{y_position(target):.2f}" x2="224" y2="{label_y:.2f}" '
            'class="leader" stroke-width="0.7"/>'
        )
        lines.append(
            f'<text x="226" y="{label_y + 3:.2f}" class="text" '
            f'font-size="8.1" font-weight="600">{html.escape(label)}</text>'
        )
    lines.extend(
        (
            '<text x="130" y="232" class="text" text-anchor="middle" '
            'font-size="8.8">normalised radius ρ</text>',
            '<text x="10.5" y="112.5" class="text" text-anchor="middle" '
            'font-size="8.5" transform="rotate(-90 10.5 112.5)">'
            "j bootstrap / j total</text>",
            '<text x="226" y="134" class="muted" font-size="7.5">'
            "circles · direct TORAX</text>",
            '<text x="226" y="146" class="muted" font-size="7.5">'
            "line · Nova facade</text>",
            '<text x="226" y="36" class="muted" font-size="7.5">'
            "authorised non-identical</text>",
            '<text x="226" y="48" class="muted" font-size="7.5">zeros → Sauter</text>',
            "</svg>",
        )
    )
    FIGURE_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _receipt(
    captures: Sequence[RunCapture], banked_contraction: float
) -> dict[str, Any]:
    implicit, zeros, sauter = captures
    changed_paths = _changed_leaves(zeros.config, sauter.config)
    expected_path = ".".join(MODEL_PATH)
    if changed_paths != [expected_path]:
        raise RuntimeError(f"unexpected configuration delta: {changed_paths}")
    implicit_contraction = float(implicit.result.convergence.contraction_estimate)
    explicit_contraction = float(zeros.result.convergence.contraction_estimate)
    selector_difference = abs(implicit_contraction - explicit_contraction)
    if selector_difference > SELECTOR_EQUIVALENCE_TOLERANCE:
        raise RuntimeError(
            f"selector-equivalence difference changed to {selector_difference}"
        )
    historical_difference = abs(explicit_contraction - banked_contraction)
    if historical_difference > HISTORICAL_REPRODUCTION_TOLERANCE:
        raise RuntimeError(
            f"historical reproduction changed to {historical_difference}"
        )
    implicit_min, implicit_max = _profile_extrema(implicit)
    zeros_min, zeros_max = _profile_extrema(zeros)
    sauter_min, sauter_max = _profile_extrema(sauter)
    if max(abs(implicit_min), abs(implicit_max)) > 1.0e-15:
        raise RuntimeError("the implicit baseline has a non-zero bootstrap fraction")
    if max(abs(zeros_min), abs(zeros_max)) > 1.0e-15:
        raise RuntimeError("the zeros baseline has a non-zero bootstrap fraction")
    if max(abs(sauter_min), abs(sauter_max)) <= 1.0e-8:
        raise RuntimeError("the Sauter inspection did not produce bootstrap current")
    implicit_model = _implicit_baseline_model()
    if implicit_model != BASELINE_MODEL:
        raise RuntimeError(
            f"the absent banked selector resolves to {implicit_model}, not zeros"
        )
    explicit_zeros_model = _validated_model_name(zeros.config)
    explicit_sauter_model = _validated_model_name(sauter.config)
    if (explicit_zeros_model, explicit_sauter_model) != (
        BASELINE_MODEL,
        INSPECTION_MODEL,
    ):
        raise RuntimeError(
            "explicit source dictionaries did not validate to their declared models"
        )
    return {
        "authority": {
            "ruling_date": OWNER_RULING_DATE,
            "classification": "owner-authorised non-identical configuration",
        },
        "demonstration": {
            "regime": "gentle",
            "window_seconds": WINDOW_SECONDS,
            "auxiliary_source_multiplier": AUXILIARY_MULTIPLIER,
        },
        "configuration_delta": {
            "changed_leaf_count": len(changed_paths),
            "changed_paths": changed_paths,
            "before": BASELINE_MODEL,
            "after": INSPECTION_MODEL,
            "all_other_fields_identical": True,
            "source_dictionary_shapes_identical": True,
            "banked_selector_present_in_raw_dictionary": False,
            "banked_implicit_validated_model": implicit_model,
            "explicit_zeros_validated_model": explicit_zeros_model,
            "explicit_sauter_validated_model": explicit_sauter_model,
        },
        "selector_equivalence": {
            "implicit_contraction": implicit_contraction,
            "explicit_zeros_contraction": explicit_contraction,
            "absolute_difference": selector_difference,
            "tolerance": SELECTOR_EQUIVALENCE_TOLERANCE,
            "passed": True,
            "same_process": True,
            "same_environment": True,
            "same_prepared_fixture": True,
        },
        "historical_reproduction": {
            "banked_contraction": banked_contraction,
            "explicit_zeros_contraction": explicit_contraction,
            "absolute_difference": historical_difference,
            "tolerance": HISTORICAL_REPRODUCTION_TOLERANCE,
            "passed": True,
            "comparison_class": "cross-environment reproduction note",
            "current_tree_same_config_contraction": (
                CURRENT_TREE_SAME_CONFIG_CONTRACTION
            ),
            "current_tree_same_config_drift": CURRENT_TREE_SAME_CONFIG_DRIFT,
            "measured_backend_drift": MEASURED_BACKEND_DRIFT,
        },
        "implicit_default": {
            "outcome_type": implicit.result.outcome_type,
            "iterations": implicit.result.convergence.iterations_used,
            "contraction": implicit_contraction,
            "bootstrap_fraction_minimum": implicit_min,
            "bootstrap_fraction_maximum": implicit_max,
            "facade_direct_maximum_separation": _route_gap(implicit),
            "wall_seconds": implicit.seconds,
        },
        "zeros": {
            "outcome_type": zeros.result.outcome_type,
            "iterations": zeros.result.convergence.iterations_used,
            "contraction": explicit_contraction,
            "banked_contraction": banked_contraction,
            "banked_contraction_absolute_difference": historical_difference,
            "bootstrap_fraction_minimum": zeros_min,
            "bootstrap_fraction_maximum": zeros_max,
            "facade_direct_maximum_separation": _route_gap(zeros),
            "wall_seconds": zeros.seconds,
        },
        "sauter": {
            "outcome_type": sauter.result.outcome_type,
            "iterations": sauter.result.convergence.iterations_used,
            "contraction": float(sauter.result.convergence.contraction_estimate),
            "exit_gating_norm": float(sauter.result.convergence.gating_norm),
            "exit_all_field_norm": float(sauter.result.convergence.all_field_norm),
            "bootstrap_fraction_minimum": sauter_min,
            "bootstrap_fraction_maximum": sauter_max,
            "facade_direct_maximum_separation": _route_gap(sauter),
            "wall_seconds": sauter.seconds,
        },
    }


def _write_report(receipt: Mapping[str, Any]) -> None:
    sauter = receipt["sauter"]
    delta = receipt["configuration_delta"]
    selector = receipt["selector_equivalence"]
    historical = receipt["historical_reproduction"]
    REPORT_PATH.write_text(
        "\n".join(
            (
                "# Sauter bootstrap-current inspection",
                "",
                (
                    "This is an owner-authorised non-identical configuration "
                    f"ruling dated {OWNER_RULING_DATE}. It is an inspection, not "
                    "an identity replay of the banked zeros configuration."
                ),
                "",
                "## Configuration identity",
                "",
                (
                    f"Exactly `{delta['changed_leaf_count']}` configuration leaf "
                    f"changed: `{delta['changed_paths'][0]}` from "
                    f"`{delta['before']}` to `{delta['after']}`. All other "
                    "configuration fields and the full source-dictionary shape "
                    "compare identical."
                ),
                (
                    "The banked raw dictionary omits this selector and TORAX "
                    "validation resolves it to "
                    f"`{delta['banked_implicit_validated_model']}`. "
                    "Both inspection dictionaries set the leaf explicitly before "
                    "validation."
                ),
                "",
                "## WindowReceipt evidence",
                "",
                (
                    "In one process, environment, and prepared fixture, the "
                    "implicit-default and explicit-zeros contractions were "
                    f"`{selector['implicit_contraction']:.17g}` and "
                    f"`{selector['explicit_zeros_contraction']:.17g}`. Their "
                    f"absolute difference `{selector['absolute_difference']:.17g}` "
                    f"passes the `{selector['tolerance']:.17g}` selector-equivalence "
                    "gate."
                ),
                (
                    "Against the untouched historical bank, explicit zeros differs "
                    f"by `{historical['absolute_difference']:.17g}`, passing the "
                    f"cross-environment `{historical['tolerance']:.17g}` note. "
                    "The independently measured same-configuration current-tree "
                    f"drift is `{historical['current_tree_same_config_drift']:.17g}` "
                    "and backend drift is "
                    f"`{historical['measured_backend_drift']:.17g}`."
                ),
                (
                    "The Sauter rerun returned `WindowReceipt` in "
                    f"`{sauter['iterations']}` iterations with contraction "
                    f"`{sauter['contraction']:.17g}`, exit gating norm "
                    f"`{sauter['exit_gating_norm']:.17g}`, and exit all-field "
                    f"norm `{sauter['exit_all_field_norm']:.17g}`."
                ),
                "",
                "## Bootstrap profile",
                "",
                (
                    "The zeros fraction remained exactly zero. The Sauter curves "
                    f"span `{sauter['bootstrap_fraction_minimum']:.17g}` to "
                    f"`{sauter['bootstrap_fraction_maximum']:.17g}`. The maximum "
                    "Nova-facade versus direct-TORAX separation is "
                    f"`{sauter['facade_direct_maximum_separation']:.17g}`."
                ),
                "",
                f"Figure: `{FIGURE_PATH}`",
                f"Profiles: `{PROFILE_PATH}`",
                f"Machine receipt: `{RECEIPT_PATH}`",
                "",
            )
        ),
        encoding="utf-8",
    )


def main() -> int:
    """Run implicit zeros, explicit zeros, and the one-leaf Sauter change."""
    raw_config = forward_module._thaw_mapping(
        demonstration._torax_model(WINDOW_SECONDS).torax_config
    )
    print(f"raw_demonstration_config_keys={tuple(raw_config.keys())}", flush=True)
    print(f"raw_neoclassical_present={'neoclassical' in raw_config}", flush=True)
    started = time.perf_counter()
    shared = demonstration_figures._prepare_fixture()
    preparation_seconds = time.perf_counter() - started
    implicit = _run_configuration(shared, None)
    zeros = _run_configuration(shared, BASELINE_MODEL)
    banked_contraction = _banked_contraction()
    _verify_selector_equivalence(implicit, zeros, banked_contraction)
    sauter = _run_configuration(shared, INSPECTION_MODEL)
    captures = (implicit, zeros, sauter)
    _write_profiles(captures)
    _write_figure(zeros, sauter)
    receipt = _receipt(captures, banked_contraction)
    receipt["fixture_preparation_seconds"] = preparation_seconds
    receipt["profile_rows"] = sum(
        curve.rho.size for capture in captures for curve in capture.curves
    )
    RECEIPT_PATH.write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_report(receipt)
    print(json.dumps(receipt, sort_keys=True))
    for path in (PROFILE_PATH, RECEIPT_PATH, REPORT_PATH, FIGURE_PATH):
        print(f"artifact={path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
