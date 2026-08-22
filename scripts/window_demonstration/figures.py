"""Reproduce the converged window and draw its evolved radial profiles."""

from __future__ import annotations

import csv
import html
import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from scripts.window_demonstration import run_window as demonstration


OUTPUT_DIRECTORY = Path(__file__).resolve().parent
PROFILE_PATH = OUTPUT_DIRECTORY / "profiles.tsv"
FIGURE_DIRECTORY = (
    OUTPUT_DIRECTORY.parent.parent
    / "docs"
    / "figures"
    / "flux-function-forward-transport"
)
TEMPERATURE_PATH = FIGURE_DIRECTORY / "evolved-temperatures.svg"
SAFETY_FACTOR_PATH = FIGURE_DIRECTORY / "evolved-safety-factor.svg"
BOOTSTRAP_PATH = FIGURE_DIRECTORY / "evolved-bootstrap.svg"

WINDOW_SECONDS = 0.0025
AUXILIARY_MULTIPLIER = 0.5


@dataclass(frozen=True)
class Curve:
    """One plotted radial profile and the semantics carried into the TSV."""

    quantity: str
    route: str
    state: str
    rho: np.ndarray
    value: np.ndarray
    unit: str


@dataclass(frozen=True)
class LabelGroup:
    """Curves described by one direct label."""

    text: str
    curves: tuple[Curve, ...]
    probe_rho: float = 0.78


def _prepare_fixture() -> dict[str, Any]:
    """Build the exact fixture inputs used by the committed window run."""
    profile, seed, _vacuum = demonstration._fixture_machine()
    extraction_lattice = demonstration._extraction_lattice(profile)
    fixture_sources = demonstration._fixture_sources(profile)
    equilibrium = profile.solve(
        seed,
        route="anderson",
        evaluations=demonstration.EVALUATIONS,
    )
    geometry, extraction = demonstration._geometry_from_equilibrium(
        equilibrium,
        profile.source,
        extraction_lattice,
        fixture_sources,
    )
    extraction.update(iteration=0, sample=0)
    return {
        "profile": profile,
        "baseline_equilibrium": equilibrium,
        "baseline_geometry": geometry,
        "baseline_extraction": extraction,
        "extraction_lattice": extraction_lattice,
        "fixture_sources": fixture_sources,
    }


def _run_facade_and_standalone():
    """Run one coupled window and replay its terminal TORAX interval directly."""
    from nova.jax.config import configure_dtypes
    from nova.transport import forward as forward_module
    from torax._src.orchestration.run_simulation import run_simulation

    configure_dtypes()
    executions: list[Any] = []
    configs: list[Any] = []
    window_receipts: list[Any] = []
    prepare_config = forward_module._prepare_torax_config
    run_steps = forward_module._run_torax_steps
    solve_window = demonstration.solve_window

    def record_config(inputs):
        config = prepare_config(inputs)
        configs.append(config)
        return config

    def record_execution(config, durations, multiplier):
        execution = run_steps(config, durations, multiplier)
        executions.append(execution)
        return execution

    def record_window(*args, **kwargs):
        receipt = solve_window(*args, **kwargs)
        window_receipts.append(receipt)
        return receipt

    forward_module._prepare_torax_config = record_config
    forward_module._run_torax_steps = record_execution
    demonstration.solve_window = record_window
    try:
        shared = _prepare_fixture()
        regime = demonstration.RegimeConfig(
            "gentle",
            WINDOW_SECONDS,
            AUXILIARY_MULTIPLIER,
            1,
        )
        result = demonstration._run_regime(regime, **shared)
    finally:
        forward_module._prepare_torax_config = prepare_config
        forward_module._run_torax_steps = run_steps
        demonstration.solve_window = solve_window

    if not result.converged:
        raise RuntimeError(
            "the committed gentle configuration did not converge: "
            f"{result.outcome_type}: {result.outcome}"
        )
    if len(window_receipts) != 1:
        raise RuntimeError(
            f"expected one coupled-window receipt, observed {len(window_receipts)}"
        )
    if not executions or len(executions) != len(configs):
        raise RuntimeError(
            "TORAX execution/config capture was incomplete: "
            f"executions={len(executions)}, configs={len(configs)}"
        )

    facade_execution = executions[-1]
    facade_config = configs[-1]
    direct_output, direct_history = run_simulation(
        facade_config,
        progress_bar=False,
    )
    if direct_history.sim_error.name != "NO_ERROR":
        raise RuntimeError(
            f"standalone TORAX replay failed with {direct_history.sim_error.name}"
        )
    return result, facade_execution, facade_config, direct_output


def _rho_with_boundaries(state) -> np.ndarray:
    """Return TORAX's cell-centred coordinate with both boundaries."""
    return np.concatenate(
        (
            np.zeros(1, dtype=np.float64),
            np.asarray(state.geometry.rho_norm, dtype=np.float64),
            np.ones(1, dtype=np.float64),
        )
    )


def _facade_bootstrap_fraction(state, min_rho_norm: float) -> np.ndarray:
    """Return local toroidal bootstrap density divided by total density."""
    from torax._src.output_tools import post_processing
    from torax._src.physics import psi_calculations

    bootstrap = state.core_sources.bootstrap_current
    bootstrap_cell = psi_calculations.j_parallel_to_j_toroidal(
        bootstrap.j_parallel_bootstrap,
        state.geometry,
        min_rho_norm,
    )
    bootstrap_face = post_processing._convert_j_parallel_face_to_j_toroidal_face(
        bootstrap.j_parallel_bootstrap_face,
        bootstrap.j_parallel_bootstrap,
        bootstrap_cell,
        state.geometry,
    )
    bootstrap_profile = np.concatenate(
        (
            np.asarray(bootstrap_face[:1], dtype=np.float64),
            np.asarray(bootstrap_cell, dtype=np.float64),
            np.asarray(bootstrap_face[-1:], dtype=np.float64),
        )
    )
    total_profile = np.concatenate(
        (
            np.asarray(state.core_profiles.j_total_face[:1], dtype=np.float64),
            np.asarray(state.core_profiles.j_total, dtype=np.float64),
            np.asarray(state.core_profiles.j_total_face[-1:], dtype=np.float64),
        )
    )
    scale = max(float(np.max(np.abs(total_profile))), 1.0)
    return np.divide(
        bootstrap_profile,
        total_profile,
        out=np.zeros_like(bootstrap_profile),
        where=np.abs(total_profile) > scale * 1.0e-14,
    )


def _dataset_profile(
    dataset,
    name: str,
    time_index: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Read one TORAX xarray profile and its matching radial coordinate."""
    profile = dataset[name].isel(time=time_index)
    radial_dimensions = [dimension for dimension in profile.dims if dimension != "time"]
    if len(radial_dimensions) != 1:
        raise RuntimeError(f"{name} has unexpected dimensions {profile.dims}")
    radial_dimension = radial_dimensions[0]
    return (
        np.asarray(dataset.coords[radial_dimension], dtype=np.float64),
        np.asarray(profile, dtype=np.float64),
    )


def _safe_fraction(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    """Divide current-density profiles while defining an empty-current edge as zero."""
    scale = max(float(np.max(np.abs(denominator))), 1.0)
    return np.divide(
        numerator,
        denominator,
        out=np.zeros_like(numerator),
        where=np.abs(denominator) > scale * 1.0e-14,
    )


def _collect_curves(facade_execution, facade_config, direct_output) -> list[Curve]:
    """Extract every curve plotted from the façade and standalone TORAX replay."""
    curves: list[Curve] = []
    states = (facade_execution.states[0], facade_execution.states[-1])
    for state_name, state in zip(("initial", "final"), states, strict=True):
        rho = _rho_with_boundaries(state)
        curves.extend(
            (
                Curve(
                    "ion_temperature",
                    "facade",
                    state_name,
                    rho,
                    np.asarray(state.core_profiles.T_i.cell_plus_boundaries()),
                    "keV",
                ),
                Curve(
                    "electron_temperature",
                    "facade",
                    state_name,
                    rho,
                    np.asarray(state.core_profiles.T_e.cell_plus_boundaries()),
                    "keV",
                ),
                Curve(
                    "safety_factor",
                    "facade",
                    state_name,
                    np.asarray(state.geometry.rho_face_norm),
                    np.asarray(state.core_profiles.q_face),
                    "1",
                ),
                Curve(
                    "bootstrap_fraction",
                    "facade",
                    state_name,
                    rho,
                    _facade_bootstrap_fraction(
                        state,
                        float(facade_config.numerics.min_rho_norm),
                    ),
                    "1",
                ),
            )
        )

    dataset = direct_output.children["profiles"].dataset
    for state_name, time_index in (("initial", 0), ("final", -1)):
        for quantity, variable, unit in (
            ("ion_temperature", "T_i", "keV"),
            ("electron_temperature", "T_e", "keV"),
            ("safety_factor", "q", "1"),
        ):
            rho, value = _dataset_profile(dataset, variable, time_index)
            curves.append(
                Curve(
                    quantity,
                    "torax-standalone",
                    state_name,
                    rho,
                    value,
                    unit,
                )
            )
        rho, bootstrap = _dataset_profile(dataset, "j_bootstrap", time_index)
        total_rho, total = _dataset_profile(dataset, "j_total", time_index)
        np.testing.assert_allclose(rho, total_rho, rtol=0.0, atol=0.0)
        curves.append(
            Curve(
                "bootstrap_fraction",
                "torax-standalone",
                state_name,
                rho,
                _safe_fraction(bootstrap, total),
                "1",
            )
        )
    return curves


def _write_profiles(curves: Sequence[Curve]) -> None:
    """Write the full plotted dataset with one row per curve point."""
    with PROFILE_PATH.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=(
                "window_seconds",
                "auxiliary_multiplier",
                "quantity",
                "route",
                "state",
                "rho",
                "value",
                "unit",
            ),
            delimiter="\t",
            lineterminator="\n",
        )
        writer.writeheader()
        for curve in curves:
            for rho, value in zip(curve.rho, curve.value, strict=True):
                writer.writerow(
                    {
                        "window_seconds": f"{WINDOW_SECONDS:.17g}",
                        "auxiliary_multiplier": f"{AUXILIARY_MULTIPLIER:.17g}",
                        "quantity": curve.quantity,
                        "route": curve.route,
                        "state": curve.state,
                        "rho": f"{float(rho):.17g}",
                        "value": f"{float(value):.17g}",
                        "unit": curve.unit,
                    }
                )


def _select(
    curves: Sequence[Curve],
    *,
    quantities: Iterable[str],
) -> list[Curve]:
    selected = set(quantities)
    return [curve for curve in curves if curve.quantity in selected]


def _svg_text(
    x: float,
    y: float,
    value: str,
    *,
    css_class: str = "text",
    anchor: str = "start",
    size: float = 9.0,
    weight: int = 400,
) -> str:
    return (
        f'<text x="{x:.2f}" y="{y:.2f}" class="{css_class}" '
        f'text-anchor="{anchor}" font-size="{size:.1f}" '
        f'font-weight="{weight}">{html.escape(value)}</text>'
    )


def _nice_ticks(lower: float, upper: float, count: int = 4) -> np.ndarray:
    """Return a compact set of readable linear tick values."""
    span = upper - lower
    if not np.isfinite(span) or span <= 0.0:
        return np.asarray((lower, upper))
    rough = span / max(count - 1, 1)
    magnitude = 10.0 ** math.floor(math.log10(rough))
    scaled = rough / magnitude
    step = next(
        candidate for candidate in (1.0, 2.0, 2.5, 5.0, 10.0) if scaled <= candidate
    )
    step *= magnitude
    start = math.ceil(lower / step) * step
    stop = math.floor(upper / step) * step
    ticks = np.arange(start, stop + 0.5 * step, step)
    return ticks if ticks.size >= 2 else np.linspace(lower, upper, count)


def _tick_text(value: float) -> str:
    if value == 0.0:
        return "0"
    if abs(value) >= 1000.0 or abs(value) < 0.01:
        return f"{value:.1e}"
    return f"{value:.3g}"


def _curve_style(curve: Curve) -> tuple[str, str, float]:
    css_class = "curve-final" if curve.state == "final" else "curve-initial"
    dash = "5 3" if curve.route == "torax-standalone" else "none"
    width = 2.0 if curve.quantity in {"ion_temperature", "safety_factor"} else 1.45
    return css_class, dash, width


def _path(curve: Curve, x_position, y_position) -> str:
    points = " ".join(
        f"{x_position(float(rho)):.2f},{y_position(float(value)):.2f}"
        for rho, value in zip(curve.rho, curve.value, strict=True)
    )
    css_class, dash, width = _curve_style(curve)
    return (
        f'<polyline points="{points}" class="{css_class}" fill="none" '
        f'stroke-width="{width:.2f}" stroke-dasharray="{dash}" '
        'stroke-linejoin="round" stroke-linecap="round" '
        'vector-effect="non-scaling-stroke"/>'
    )


def _standalone_markers(curve: Curve, x_position, y_position) -> list[str]:
    """Mark a coincident standalone curve without displacing either route."""
    if curve.route != "torax-standalone":
        return []
    css_class, _dash, _width = _curve_style(curve)
    indices = np.unique(np.linspace(0, curve.rho.size - 1, 5, dtype=int))
    return [
        (
            f'<circle cx="{x_position(float(curve.rho[index])):.2f}" '
            f'cy="{y_position(float(curve.value[index])):.2f}" r="1.65" '
            f'class="{css_class}" fill="none" stroke-width="0.8" '
            'vector-effect="non-scaling-stroke"/>'
        )
        for index in indices
    ]


def _spread_labels(
    desired: Sequence[float],
    *,
    lower: float,
    upper: float,
    spacing: float = 14.0,
) -> list[float]:
    """Keep direct labels readable without moving leaders outside the plot."""
    order = np.argsort(desired)
    placed = [0.0] * len(desired)
    cursor = lower
    for index in order:
        position = max(float(desired[index]), cursor)
        placed[index] = position
        cursor = position + spacing
    overflow = max(0.0, max(placed, default=upper) - upper)
    if overflow:
        placed = [position - overflow for position in placed]
    underflow = max(0.0, lower - min(placed, default=lower))
    if underflow:
        placed = [position + underflow for position in placed]
    return placed


def _write_svg(
    path: Path,
    curves: Sequence[Curve],
    labels: Sequence[LabelGroup],
    *,
    title: str,
    description: str,
    y_axis: str,
    y_limits: tuple[float, float] | None = None,
) -> None:
    """Write one transparent, direct-labelled SVG at its review width."""
    width, height = 320.0, 238.0
    left, right, top, bottom = 42.0, 218.0, 23.0, 202.0
    values = np.concatenate([np.asarray(curve.value) for curve in curves])
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        raise RuntimeError(f"{path.name} has no finite values")
    if y_limits is None:
        lower, upper = float(np.min(finite)), float(np.max(finite))
        padding = 0.08 * max(upper - lower, abs(upper), 1.0)
        lower -= padding
        upper += padding
    else:
        lower, upper = y_limits

    def x_position(rho: float) -> float:
        return left + rho * (right - left)

    def y_position(value: float) -> float:
        return bottom - (value - lower) / (upper - lower) * (bottom - top)

    identifier = path.stem.replace("-", "_")
    lines = [
        (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{int(width)}" '
            f'height="{int(height)}" viewBox="0 0 {int(width)} {int(height)}" '
            f'role="img" aria-labelledby="{identifier}_title {identifier}_desc">'
        ),
        f'<title id="{identifier}_title">{html.escape(title)}</title>',
        f'<desc id="{identifier}_desc">{html.escape(description)}</desc>',
        "<style>",
        (
            ".text{fill:#202124;font-family:system-ui,sans-serif}"
            ".muted-text{fill:#62666a;font-family:system-ui,sans-serif}"
            ".axis{stroke:#202124}.grid{stroke:#d4d6d8}"
            ".curve-final{stroke:#202124}.curve-initial{stroke:#71757a}"
            ".leader{stroke:#62666a}"
        ),
        (
            "@media (prefers-color-scheme:dark){"
            ".text{fill:#f1f3f4}.muted-text{fill:#c2c7cc}"
            ".axis{stroke:#f1f3f4}.grid{stroke:#4b5055}"
            ".curve-final{stroke:#f1f3f4}.curve-initial{stroke:#b4bac0}"
            ".leader{stroke:#c2c7cc}}"
        ),
        "</style>",
    ]
    for tick in _nice_ticks(lower, upper):
        y = y_position(float(tick))
        lines.append(
            f'<line x1="{left:.2f}" y1="{y:.2f}" x2="{right:.2f}" '
            f'y2="{y:.2f}" class="grid" stroke-width="0.65"/>'
        )
        lines.append(
            _svg_text(
                left - 5.0,
                y + 3.1,
                _tick_text(float(tick)),
                anchor="end",
                size=8.0,
            )
        )
    lines.extend(
        (
            f'<line x1="{left:.2f}" y1="{top:.2f}" x2="{left:.2f}" '
            f'y2="{bottom:.2f}" class="axis" stroke-width="0.9"/>',
            f'<line x1="{left:.2f}" y1="{bottom:.2f}" x2="{right:.2f}" '
            f'y2="{bottom:.2f}" class="axis" stroke-width="0.9"/>',
        )
    )
    for tick in (0.0, 0.25, 0.5, 0.75, 1.0):
        x = x_position(tick)
        lines.append(
            f'<line x1="{x:.2f}" y1="{bottom:.2f}" x2="{x:.2f}" '
            f'y2="{bottom + 3.5:.2f}" class="axis" stroke-width="0.75"/>'
        )
        lines.append(
            _svg_text(
                x,
                bottom + 13.5,
                f"{tick:g}",
                anchor="middle",
                size=8.0,
            )
        )

    for curve in curves:
        lines.append(_path(curve, x_position, y_position))
        lines.extend(_standalone_markers(curve, x_position, y_position))

    desired = []
    for group in labels:
        representative = group.curves[0]
        value = float(
            np.interp(group.probe_rho, representative.rho, representative.value)
        )
        desired.append(y_position(value))
    label_positions = _spread_labels(
        desired,
        lower=top + 8.0,
        upper=bottom - 5.0,
    )
    label_x = right + 8.0
    for group, target_y, label_y in zip(labels, desired, label_positions, strict=True):
        probe_x = x_position(group.probe_rho)
        lines.append(
            f'<line x1="{probe_x:.2f}" y1="{target_y:.2f}" '
            f'x2="{label_x - 2.0:.2f}" y2="{label_y:.2f}" '
            'class="leader" stroke-width="0.7"/>'
        )
        lines.append(
            _svg_text(
                label_x,
                label_y + 3.2,
                group.text,
                size=8.3,
                weight=600,
            )
        )

    lines.extend(
        (
            _svg_text(
                0.5 * (left + right),
                232.0,
                "normalised radius ρ",
                anchor="middle",
                size=8.8,
            ),
            _svg_text(
                10.5,
                0.5 * (top + bottom),
                y_axis,
                anchor="middle",
                size=8.5,
            )
            .replace(
                "</text>",
                "</text>",
            )
            .replace(
                f'x="10.50" y="{0.5 * (top + bottom):.2f}"',
                f'x="10.50" y="{0.5 * (top + bottom):.2f}" '
                f'transform="rotate(-90 10.50 {0.5 * (top + bottom):.2f})"',
            ),
            "</svg>",
        )
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _pair(
    curves: Sequence[Curve],
    quantity: str,
    state: str,
) -> tuple[Curve, ...]:
    pair = tuple(
        curve for curve in curves if curve.quantity == quantity and curve.state == state
    )
    if len(pair) != 2:
        raise RuntimeError(f"expected two {quantity}/{state} curves, found {len(pair)}")
    return pair


def _maximum_separation(curves: Sequence[Curve], quantities: Sequence[str]) -> float:
    """Return the largest façade-versus-standalone gap in plotted units."""
    maximum = 0.0
    for quantity in quantities:
        for state in ("initial", "final"):
            pair = _pair(curves, quantity, state)
            facade = next(curve for curve in pair if curve.route == "facade")
            standalone = next(
                curve for curve in pair if curve.route == "torax-standalone"
            )
            standalone_value = np.interp(
                facade.rho,
                standalone.rho,
                standalone.value,
            )
            maximum = max(
                maximum,
                float(np.max(np.abs(facade.value - standalone_value))),
            )
    return maximum


def _write_figures(curves: Sequence[Curve]) -> None:
    temperatures = _select(
        curves,
        quantities=("ion_temperature", "electron_temperature"),
    )
    _write_svg(
        TEMPERATURE_PATH,
        temperatures,
        (
            LabelGroup(
                "Tᵢ initial · both",
                _pair(curves, "ion_temperature", "initial"),
            ),
            LabelGroup(
                "Tₑ initial · both",
                _pair(curves, "electron_temperature", "initial"),
                0.68,
            ),
            LabelGroup(
                "Tᵢ final · both",
                _pair(curves, "ion_temperature", "final"),
                0.58,
            ),
            LabelGroup(
                "Tₑ final · both",
                _pair(curves, "electron_temperature", "final"),
                0.48,
            ),
        ),
        title="Ion and electron temperatures across the converged coupled window",
        description=(
            "Initial and final ion and electron temperature profiles from the "
            "Nova facade, overlaid with a direct TORAX replay of the identical "
            "terminal transport interval. Solid and dashed strokes coincide "
            "where the two routes agree."
        ),
        y_axis="temperature (keV)",
    )

    safety = _select(curves, quantities=("safety_factor",))
    _write_svg(
        SAFETY_FACTOR_PATH,
        safety,
        (
            LabelGroup(
                "q initial · both",
                _pair(curves, "safety_factor", "initial"),
                0.70,
            ),
            LabelGroup(
                "q final · both",
                _pair(curves, "safety_factor", "final"),
                0.55,
            ),
        ),
        title="Safety factor across the converged coupled window",
        description=(
            "Initial and final safety-factor profiles from the Nova facade, "
            "overlaid with a direct TORAX replay of the identical terminal "
            "transport interval."
        ),
        y_axis="safety factor q",
    )

    bootstrap = _select(curves, quantities=("bootstrap_fraction",))
    bootstrap_values = np.concatenate([curve.value for curve in bootstrap])
    if np.max(np.abs(bootstrap_values)) <= 1.0e-15:
        bootstrap_limits = (-0.02, 0.02)
        bootstrap_label = "all states/routes · zero"
    else:
        bootstrap_limits = None
        bootstrap_label = "bootstrap fraction · façade / TORAX"
    _write_svg(
        BOOTSTRAP_PATH,
        bootstrap,
        (
            LabelGroup(
                bootstrap_label,
                tuple(bootstrap),
                0.62,
            ),
        ),
        title="Local bootstrap-current fraction across the converged coupled window",
        description=(
            "Initial and final local toroidal bootstrap-current fractions from "
            "the Nova facade and the direct TORAX replay. The exact committed "
            "configuration selects TORAX's zero-bootstrap model, so every "
            "profile lies on zero."
        ),
        y_axis="j bootstrap / j total",
        y_limits=bootstrap_limits,
    )


def main() -> int:
    """Regenerate data and all three review-width figures from one window run."""
    result, facade_execution, facade_config, direct_output = (
        _run_facade_and_standalone()
    )
    curves = _collect_curves(facade_execution, facade_config, direct_output)
    _write_profiles(curves)
    _write_figures(curves)

    temperature_gap = _maximum_separation(
        curves,
        ("ion_temperature", "electron_temperature"),
    )
    safety_gap = _maximum_separation(curves, ("safety_factor",))
    bootstrap_gap = _maximum_separation(curves, ("bootstrap_fraction",))
    print(f"window_outcome={result.outcome_type}")
    print(f"window_iterations={result.convergence.iterations_used}")
    print(f"window_contraction={float(result.convergence.contraction_estimate):.17g}")
    print(f"window_maximum_residual={float(result.convergence.maximum_residual):.17g}")
    print(f"temperature_maximum_separation_kev={temperature_gap:.17g}")
    print(f"safety_factor_maximum_separation={safety_gap:.17g}")
    print(f"bootstrap_fraction_maximum_separation={bootstrap_gap:.17g}")
    print(f"bootstrap_model={facade_config.neoclassical.bootstrap_current.model_name}")
    print(f"profile_rows={sum(curve.rho.size for curve in curves)}")
    for path in (PROFILE_PATH, TEMPERATURE_PATH, SAFETY_FACTOR_PATH, BOOTSTRAP_PATH):
        print(f"artifact={path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
