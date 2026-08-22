"""Build validation time traces from the published ramp-up and coupled window."""

from __future__ import annotations

import argparse
import copy
import csv
import html
import math
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from scripts.window_demonstration import run_window as demonstration


OUTPUT_DIRECTORY = Path(__file__).resolve().parent
TRACE_PATH = OUTPUT_DIRECTORY / "time_traces.tsv"
FIGURE_DIRECTORY = (
    OUTPUT_DIRECTORY.parent.parent
    / "docs"
    / "figures"
    / "flux-function-forward-transport"
)
RAMPUP_PATH = FIGURE_DIRECTORY / "rampup-time-traces.svg"
WINDOW_PATH = FIGURE_DIRECTORY / "window-time-traces.svg"

RAMPUP_SECONDS = 20.0
FINAL_FIELD_RATIO = 0.98
WINDOW_SECONDS = 0.0025
AUXILIARY_MULTIPLIER = 0.5
WINDOW_TOLERANCE = 0.005
REPORTED_CONTRACTION = 0.53710396334179378

TRACE_FIELDS = (
    "case",
    "quantity",
    "route",
    "iteration",
    "time_s",
    "value",
    "unit",
)


@dataclass(frozen=True)
class Trace:
    """One scalar trajectory with its plotting semantics."""

    case: str
    quantity: str
    route: str
    iteration: np.ndarray
    time_s: np.ndarray
    value: np.ndarray
    unit: str


@dataclass(frozen=True)
class Panel:
    """Geometry and scale for one compact SVG panel."""

    top: float
    bottom: float
    lower: float
    upper: float
    logarithmic: bool = False


def _published_config() -> dict[str, Any]:
    """Return the landed moving-grid form of TORAX's published ITER ramp-up."""
    from torax.examples.iterhybrid_rampup import CONFIG

    config = copy.deepcopy(CONFIG)
    config["numerics"]["t_final"] = RAMPUP_SECONDS
    config["numerics"]["exact_t_final"] = True
    geometry = config["geometry"]
    initial_field = geometry.pop("B_0")
    geometry["calcphibdot"] = True
    geometry["geometry_configs"] = {
        0.0: {"B_0": initial_field},
        RAMPUP_SECONDS: {"B_0": initial_field * FINAL_FIELD_RATIO},
    }
    return config


def _initial_transport_state(config: Any):
    """Construct the façade state through the same TORAX initialisation route."""
    import nova.transport.forward as forward_module
    from nova.transport import TransportState
    from torax._src.orchestration.initial_state import (
        get_initial_state_and_post_processed_outputs,
    )

    step_function = forward_module._make_torax_step(config)
    initial, _post_processed = get_initial_state_and_post_processed_outputs(
        step_function
    )
    profiles = initial.core_profiles
    rho = np.concatenate(([0.0], np.asarray(initial.geometry.rho_norm), [1.0]))
    return TransportState(
        rho=rho,
        psi=np.asarray(profiles.psi.cell_plus_boundaries()),
        ion_temperature=np.asarray(profiles.T_i.cell_plus_boundaries()),
        electron_temperature=np.asarray(profiles.T_e.cell_plus_boundaries()),
        electron_density=np.asarray(profiles.n_e.cell_plus_boundaries()),
    )


def _rampup_request(config_data: dict[str, Any], config: Any):
    """Build the typed façade request for the published trajectory."""
    from nova.transport import (
        ForwardTransportInput,
        TransportGeometry,
        TransportModel,
        TransportRung,
        TransportWaveforms,
    )

    initial_current = 3.0e6
    final_current = initial_current + (10.5e6 - initial_current) * (
        RAMPUP_SECONDS / 80.0
    )
    return ForwardTransportInput(
        geometry=TransportGeometry({"valid": True}),
        initial_state=_initial_transport_state(config),
        waveforms=TransportWaveforms(
            time=np.asarray((0.0, RAMPUP_SECONDS)),
            plasma_current=np.asarray((initial_current, final_current)),
        ),
        model=TransportModel(
            TransportRung.TORAX_MULTI_CHANNEL,
            torax_config=config_data,
        ),
    )


def _collect_rampup() -> list[Trace]:
    """Run both published-case routes and retain their central saved states."""
    import nova.transport.forward as forward_module
    from nova.transport import ForwardTransport
    from torax._src.orchestration.run_simulation import run_simulation
    from torax._src.torax_pydantic.model_config import ToraxConfig

    config_data = _published_config()
    direct_config = ToraxConfig.from_dict(copy.deepcopy(config_data))
    request = _rampup_request(config_data, direct_config)
    executions: list[Any] = []
    run_steps = forward_module._run_torax_steps

    def capture_execution(*args, **kwargs):
        execution = run_steps(*args, **kwargs)
        executions.append(execution)
        return execution

    forward_module._run_torax_steps = capture_execution
    try:
        receipt = ForwardTransport().solve(request)
    finally:
        forward_module._run_torax_steps = run_steps
    if len(executions) != 1:
        raise RuntimeError(f"expected one façade execution, observed {len(executions)}")

    direct_output, direct_history = run_simulation(direct_config, progress_bar=False)
    if direct_history.sim_error.name != "NO_ERROR":
        raise RuntimeError(
            f"TORAX's run_simulation route returned {direct_history.sim_error.name}"
        )
    states = executions[0].states
    dataset = direct_output.children["profiles"].dataset
    time_s = np.asarray(dataset.coords["time"], dtype=np.float64)
    facade_time = np.asarray([state.t for state in states], dtype=np.float64)
    np.testing.assert_allclose(facade_time, time_s, rtol=0.0, atol=1.0e-12)
    if receipt.diagnostics.steps != 10 or time_s.size != 11:
        raise RuntimeError(
            "the landed published trajectory must save ten evolved steps; "
            f"observed steps={receipt.diagnostics.steps}, states={time_s.size}"
        )

    specifications = (
        ("central_ion_temperature", "T_i", "ion_temperature", 1.0, "keV"),
        (
            "central_electron_temperature",
            "T_e",
            "electron_temperature",
            1.0,
            "keV",
        ),
        (
            "central_electron_density",
            "n_e",
            "electron_density",
            1.0e-20,
            "1e20 m^-3",
        ),
        ("axis_psi", "psi", "psi", 1.0, "Wb"),
    )
    traces: list[Trace] = []
    iteration = np.arange(time_s.size, dtype=np.float64)
    for quantity, dataset_name, state_name, scale, unit in specifications:
        facade = (
            np.asarray(
                [
                    getattr(state.core_profiles, dataset_name).cell_plus_boundaries()[0]
                    for state in states
                ],
                dtype=np.float64,
            )
            * scale
        )
        direct = np.asarray(dataset[dataset_name], dtype=np.float64)[:, 0] * scale
        np.testing.assert_allclose(
            facade,
            direct,
            rtol=1.0e-10,
            atol=1.0e-12 if quantity != "central_electron_density" else 1.0e-12,
        )
        traces.extend(
            (
                Trace(
                    "rampup",
                    quantity,
                    "facade",
                    iteration,
                    time_s,
                    facade,
                    unit,
                ),
                Trace(
                    "rampup",
                    quantity,
                    "torax-run-simulation",
                    iteration,
                    time_s,
                    direct,
                    unit,
                ),
                Trace(
                    "rampup",
                    quantity,
                    "absolute-separation",
                    iteration,
                    time_s,
                    np.abs(facade - direct),
                    unit,
                ),
            )
        )
    return traces


def _prepare_window_fixture() -> dict[str, Any]:
    """Build the exact fixture inputs used by the committed gentle window."""
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


def _collect_window() -> list[Trace]:
    """Reproduce the gentle window while observing each exchanged candidate."""
    from nova.jax.config import configure_dtypes

    configure_dtypes()
    candidates: list[dict[str, float]] = []
    solve_window = demonstration.solve_window

    def observe_window(
        initial_geometry,
        initial_source,
        config,
        equilibrium_update: Callable,
        transport_update: Callable,
        *,
        damping: float = 1.0,
    ):
        pending: dict[str, float] = {}

        def observe_transport(geometry_waveform, sample_grid):
            transported = transport_update(geometry_waveform, sample_grid)
            pending["central_temperature"] = float(
                np.asarray(transported.receipt.state.ion_temperature)[0]
            )
            return transported

        def observe_equilibrium(source_waveform, sample_grid):
            equilibrated = equilibrium_update(source_waveform, sample_grid)
            final_sample = equilibrated.waveform.sample(float(sample_grid[-1]))
            geometry = final_sample.geometry()
            edge_current = float(np.asarray(geometry.record["ip_profile_face"])[-1])
            candidates.append(
                {
                    "central_temperature": pending["central_temperature"],
                    "edge_plasma_current": edge_current,
                }
            )
            return equilibrated

        return solve_window(
            initial_geometry,
            initial_source,
            config,
            observe_equilibrium,
            observe_transport,
            damping=damping,
        )

    demonstration.solve_window = observe_window
    try:
        result = demonstration._run_regime(
            demonstration.RegimeConfig(
                "gentle",
                WINDOW_SECONDS,
                AUXILIARY_MULTIPLIER,
                1,
            ),
            **_prepare_window_fixture(),
        )
    finally:
        demonstration.solve_window = solve_window
    if not result.converged:
        raise RuntimeError(
            "the landed gentle configuration did not converge: "
            f"{result.outcome_type}: {result.outcome}"
        )
    convergence = result.convergence
    if convergence.iterations_used != 10 or len(candidates) != 10:
        raise RuntimeError(
            "the landed gentle window must converge at iteration ten; "
            f"receipt={convergence.iterations_used}, observed={len(candidates)}"
        )
    if not np.isclose(
        convergence.contraction_estimate,
        REPORTED_CONTRACTION,
        rtol=0.0,
        atol=5.0e-10,
    ):
        raise RuntimeError(
            f"the reproduced contraction changed: {convergence.contraction_estimate}"
        )

    iteration = np.arange(1, 11, dtype=np.float64)
    residual = np.asarray(
        [max(row.values(), default=0.0) for row in convergence.residual_trace],
        dtype=np.float64,
    )
    temperature = np.asarray(
        [candidate["central_temperature"] for candidate in candidates],
        dtype=np.float64,
    )
    current = np.asarray(
        [candidate["edge_plasma_current"] for candidate in candidates],
        dtype=np.float64,
    )
    empty_time = np.full(iteration.shape, np.nan)
    return [
        Trace(
            "window",
            "maximum_exit_residual",
            "exchanged-state",
            iteration,
            empty_time,
            residual,
            "relative",
        ),
        Trace(
            "window",
            "central_ion_temperature",
            "exchanged-state",
            iteration,
            empty_time,
            temperature,
            "keV",
        ),
        Trace(
            "window",
            "edge_plasma_current",
            "exchanged-state",
            iteration,
            empty_time,
            current,
            "A",
        ),
    ]


def _write_traces(traces: Sequence[Trace]) -> None:
    """Write every plotted value, including route separations, to one TSV."""
    with TRACE_PATH.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=TRACE_FIELDS,
            delimiter="\t",
            lineterminator="\n",
        )
        writer.writeheader()
        for trace in traces:
            for iteration, time_s, value in zip(
                trace.iteration,
                trace.time_s,
                trace.value,
                strict=True,
            ):
                writer.writerow(
                    {
                        "case": trace.case,
                        "quantity": trace.quantity,
                        "route": trace.route,
                        "iteration": f"{int(iteration)}",
                        "time_s": ("" if np.isnan(time_s) else f"{float(time_s):.17g}"),
                        "value": f"{float(value):.17g}",
                        "unit": trace.unit,
                    }
                )


def _read_traces() -> list[Trace]:
    """Load the committed tidy trace data for render-only validation."""
    groups: dict[tuple[str, str, str, str], list[dict[str, str]]] = {}
    with TRACE_PATH.open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream, delimiter="\t"):
            key = (row["case"], row["quantity"], row["route"], row["unit"])
            groups.setdefault(key, []).append(row)
    traces = []
    for (case, quantity, route, unit), rows in groups.items():
        traces.append(
            Trace(
                case,
                quantity,
                route,
                np.asarray([float(row["iteration"]) for row in rows]),
                np.asarray(
                    [float(row["time_s"]) if row["time_s"] else np.nan for row in rows]
                ),
                np.asarray([float(row["value"]) for row in rows]),
                unit,
            )
        )
    return traces


def _select(
    traces: Sequence[Trace],
    case: str,
    quantity: str,
    route: str,
) -> Trace:
    """Return exactly one named trace."""
    selected = [
        trace
        for trace in traces
        if (trace.case, trace.quantity, trace.route) == (case, quantity, route)
    ]
    if len(selected) != 1:
        raise RuntimeError(
            f"expected one {case}/{quantity}/{route} trace, found {len(selected)}"
        )
    return selected[0]


def _svg_text(
    x: float,
    y: float,
    text: str,
    *,
    anchor: str = "start",
    css_class: str = "text",
    size: float = 8.5,
    weight: int = 400,
) -> str:
    return (
        f'<text x="{x:.2f}" y="{y:.2f}" text-anchor="{anchor}" '
        f'class="{css_class}" font-size="{size:.1f}" font-weight="{weight}">'
        f"{html.escape(text)}</text>"
    )


def _tick_text(value: float) -> str:
    """Format a compact axis label without discarding scale information."""
    if value == 0.0:
        return "0"
    if abs(value) >= 1.0e4 or abs(value) < 0.01:
        return f"{value:.1e}"
    return f"{value:.3g}"


def _linear_ticks(lower: float, upper: float, count: int = 3) -> np.ndarray:
    """Choose stable readable linear ticks for a narrow panel."""
    if not upper > lower:
        return np.asarray((lower, upper))
    rough = (upper - lower) / max(count - 1, 1)
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


def _limits(values: Iterable[np.ndarray]) -> tuple[float, float]:
    """Return padded finite limits for one scalar panel."""
    combined = np.concatenate(tuple(np.asarray(value) for value in values))
    finite = combined[np.isfinite(combined)]
    lower, upper = float(np.min(finite)), float(np.max(finite))
    span = upper - lower
    padding = 0.09 * max(span, abs(upper) * 0.04, 1.0e-12)
    return lower - padding, upper + padding


def _x_position(value: float, lower: float, upper: float) -> float:
    return 43.0 + (value - lower) / (upper - lower) * (230.0 - 43.0)


def _y_position(value: float, panel: Panel) -> float:
    if panel.logarithmic:
        value = math.log10(value)
        lower = math.log10(panel.lower)
        upper = math.log10(panel.upper)
    else:
        lower, upper = panel.lower, panel.upper
    return panel.bottom - (value - lower) / (upper - lower) * (panel.bottom - panel.top)


def _polyline(
    x: np.ndarray,
    y: np.ndarray,
    panel: Panel,
    *,
    x_lower: float,
    x_upper: float,
    css_class: str,
    dash: str = "none",
    width: float = 1.55,
) -> str:
    points = " ".join(
        f"{_x_position(float(x_value), x_lower, x_upper):.2f},"
        f"{_y_position(float(y_value), panel):.2f}"
        for x_value, y_value in zip(x, y, strict=True)
    )
    return (
        f'<polyline points="{points}" class="{css_class}" fill="none" '
        f'stroke-width="{width:.2f}" stroke-dasharray="{dash}" '
        'stroke-linecap="round" stroke-linejoin="round" '
        'vector-effect="non-scaling-stroke"/>'
    )


def _svg_header(
    height: int,
    title: str,
    description: str,
    identifier: str,
) -> list[str]:
    """Return a transparent, two-theme SVG preamble."""
    return [
        (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="320" '
            f'height="{height}" viewBox="0 0 320 {height}" role="img" '
            f'aria-labelledby="{identifier}_title {identifier}_desc">'
        ),
        f'<title id="{identifier}_title">{html.escape(title)}</title>',
        f'<desc id="{identifier}_desc">{html.escape(description)}</desc>',
        "<style>",
        (
            ".text{fill:#202124;font-family:system-ui,sans-serif}"
            ".muted{fill:#666b70;font-family:system-ui,sans-serif}"
            ".axis{stroke:#202124}.grid{stroke:#d6d8da}"
            ".facade{stroke:#202124}.torax{stroke:#71767b}"
            ".trace{stroke:#202124}.fit{stroke:#71767b}"
            ".marker{fill:none;stroke:#71767b}"
        ),
        (
            "@media (prefers-color-scheme:dark){"
            ".text{fill:#f1f3f4}.muted{fill:#c1c6ca}"
            ".axis{stroke:#f1f3f4}.grid{stroke:#4c5156}"
            ".facade{stroke:#f1f3f4}.torax{stroke:#b9bec3}"
            ".trace{stroke:#f1f3f4}.fit{stroke:#b9bec3}"
            ".marker{stroke:#b9bec3}}"
        ),
        "</style>",
    ]


def _axes(
    panel: Panel,
    *,
    x_ticks: Sequence[float],
    x_lower: float,
    x_upper: float,
    y_ticks: Sequence[float],
    show_x_labels: bool,
) -> list[str]:
    """Draw minimal left and bottom axes with unobtrusive horizontal guides."""
    lines: list[str] = []
    for tick in y_ticks:
        y = _y_position(float(tick), panel)
        lines.append(
            f'<line x1="43" y1="{y:.2f}" x2="230" y2="{y:.2f}" '
            'class="grid" stroke-width="0.65"/>'
        )
        lines.append(
            _svg_text(
                38.0,
                y + 2.8,
                _tick_text(float(tick)),
                anchor="end",
                size=7.5,
            )
        )
    lines.extend(
        (
            f'<line x1="43" y1="{panel.top:.2f}" x2="43" '
            f'y2="{panel.bottom:.2f}" class="axis" stroke-width="0.85"/>',
            f'<line x1="43" y1="{panel.bottom:.2f}" x2="230" '
            f'y2="{panel.bottom:.2f}" class="axis" stroke-width="0.85"/>',
        )
    )
    for tick in x_ticks:
        x = _x_position(float(tick), x_lower, x_upper)
        lines.append(
            f'<line x1="{x:.2f}" y1="{panel.bottom:.2f}" x2="{x:.2f}" '
            f'y2="{panel.bottom + 3:.2f}" class="axis" stroke-width="0.7"/>'
        )
        if show_x_labels:
            lines.append(
                _svg_text(
                    x,
                    panel.bottom + 12.0,
                    _tick_text(float(tick)),
                    anchor="middle",
                    size=7.5,
                )
            )
    return lines


def _write_rampup(traces: Sequence[Trace]) -> None:
    """Draw four direct-labelled published-ramp-up central trajectories."""
    quantities = (
        ("central_ion_temperature", "central Tᵢ · keV"),
        ("central_electron_temperature", "central Tₑ · keV"),
        ("central_electron_density", "central nₑ · 10²⁰ m⁻³"),
        ("axis_psi", "axis ψ · Wb"),
    )
    height = 414
    lines = _svg_header(
        height,
        "Published ITER ramp-up: Nova façade against TORAX run_simulation",
        (
            "Ten evolved saved states of central ion temperature, central electron "
            "temperature, central electron density and axis poloidal flux. The "
            "façade lines and TORAX markers overlay at every saved time."
        ),
        "rampup_time_traces",
    )
    panel_height = 68.0
    panel_gap = 28.0
    for index, (quantity, label) in enumerate(quantities):
        top = 22.0 + index * (panel_height + panel_gap)
        facade = _select(traces, "rampup", quantity, "facade")
        torax = _select(traces, "rampup", quantity, "torax-run-simulation")
        separation = _select(traces, "rampup", quantity, "absolute-separation")
        lower, upper = _limits((facade.value, torax.value))
        panel = Panel(top, top + panel_height, lower, upper)
        lines.append(_svg_text(43.0, top - 6.0, label, size=8.2, weight=600))
        lines.extend(
            _axes(
                panel,
                x_ticks=(0.0, 10.0, 20.0),
                x_lower=0.0,
                x_upper=RAMPUP_SECONDS,
                y_ticks=_linear_ticks(lower, upper),
                show_x_labels=index == len(quantities) - 1,
            )
        )
        lines.append(
            _polyline(
                facade.time_s,
                facade.value,
                panel,
                x_lower=0.0,
                x_upper=RAMPUP_SECONDS,
                css_class="facade",
                width=1.7,
            )
        )
        lines.append(
            _polyline(
                torax.time_s,
                torax.value,
                panel,
                x_lower=0.0,
                x_upper=RAMPUP_SECONDS,
                css_class="torax",
                dash="3 2",
                width=1.0,
            )
        )
        for time_s, value in zip(torax.time_s, torax.value, strict=True):
            lines.append(
                f'<circle cx="{_x_position(float(time_s), 0.0, RAMPUP_SECONDS):.2f}" '
                f'cy="{_y_position(float(value), panel):.2f}" r="1.5" '
                'class="marker" stroke-width="0.8"/>'
            )
        endpoint_y = _y_position(float(facade.value[-1]), panel)
        lines.append(
            f'<line x1="230" y1="{endpoint_y:.2f}" x2="238" '
            f'y2="{endpoint_y:.2f}" class="facade" stroke-width="0.7"/>'
        )
        lines.append(
            _svg_text(
                241.0,
                endpoint_y - 1.5,
                "façade —",
                size=7.7,
                weight=600,
            )
        )
        lines.append(
            _svg_text(
                241.0,
                endpoint_y + 8.0,
                "TORAX ○",
                css_class="muted",
                size=7.7,
                weight=600,
            )
        )
        lines.append(
            _svg_text(
                307.0,
                top + 9.0,
                f"max |Δ| {_tick_text(float(np.max(separation.value)))}",
                anchor="end",
                css_class="muted",
                size=7.2,
            )
        )
    lines.append(_svg_text(136.5, height - 5.0, "time · s", anchor="middle", size=8.3))
    lines.append("</svg>")
    RAMPUP_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_window(traces: Sequence[Trace]) -> None:
    """Draw convergence, temperature and edge-current traces for the window."""
    residual = _select(
        traces,
        "window",
        "maximum_exit_residual",
        "exchanged-state",
    )
    temperature = _select(
        traces,
        "window",
        "central_ion_temperature",
        "exchanged-state",
    )
    current = _select(
        traces,
        "window",
        "edge_plasma_current",
        "exchanged-state",
    )
    height = 326
    lines = _svg_header(
        height,
        "Gentle coupled window: convergence and exchanged state",
        (
            "The per-iteration maximum exchange residual converges on a logarithmic "
            "ordinate at iteration ten. Central ion temperature and edge plasma "
            "current show the physical state exchanged during the same iterations."
        ),
        "window_time_traces",
    )

    residual_panel = Panel(25.0, 125.0, 0.0035, 2.5, logarithmic=True)
    temperature_limits = _limits((temperature.value,))
    temperature_panel = Panel(163.0, 219.0, *temperature_limits)
    current_limits = _limits((current.value,))
    current_panel = Panel(257.0, 306.0, *current_limits)
    panels = (
        (
            residual,
            residual_panel,
            "maximum exit residual · relative (log)",
            (0.005, 0.02, 0.2, 2.0),
        ),
        (
            temperature,
            temperature_panel,
            "central Tᵢ of exchanged state · keV",
            _linear_ticks(*temperature_limits),
        ),
        (
            current,
            current_panel,
            "edge plasma current of exchanged state · A",
            _linear_ticks(*current_limits),
        ),
    )
    for index, (trace, panel, label, y_ticks) in enumerate(panels):
        lines.append(_svg_text(43.0, panel.top - 7.0, label, size=8.2, weight=600))
        lines.extend(
            _axes(
                panel,
                x_ticks=(1.0, 5.0, 10.0),
                x_lower=1.0,
                x_upper=10.0,
                y_ticks=y_ticks,
                show_x_labels=index == len(panels) - 1,
            )
        )
        lines.append(
            _polyline(
                trace.iteration,
                trace.value,
                panel,
                x_lower=1.0,
                x_upper=10.0,
                css_class="trace",
                width=1.75,
            )
        )

    tolerance_y = _y_position(WINDOW_TOLERANCE, residual_panel)
    lines.append(
        f'<line x1="43" y1="{tolerance_y:.2f}" x2="230" '
        f'y2="{tolerance_y:.2f}" class="fit" stroke-width="0.8" '
        'stroke-dasharray="2 2"/>'
    )
    lines.append(
        _svg_text(
            236.0,
            tolerance_y + 2.5,
            "tolerance 0.005",
            css_class="muted",
            size=7.5,
            weight=600,
        )
    )
    terminal_fit = residual.value[-1] * REPORTED_CONTRACTION ** (
        residual.iteration - residual.iteration[-1]
    )
    fit_indices = residual.iteration >= 7.0
    lines.append(
        _polyline(
            residual.iteration[fit_indices],
            terminal_fit[fit_indices],
            residual_panel,
            x_lower=1.0,
            x_upper=10.0,
            css_class="fit",
            dash="4 2",
            width=1.0,
        )
    )
    fit_y = _y_position(float(terminal_fit[7]), residual_panel)
    lines.append(
        _svg_text(
            236.0,
            fit_y + 2.5,
            "terminal fitted slope",
            css_class="muted",
            size=7.3,
            weight=600,
        )
    )
    lines.append(
        _svg_text(
            236.0,
            fit_y + 11.5,
            "×0.537 / iteration",
            css_class="muted",
            size=7.3,
            weight=600,
        )
    )
    lines.append(
        _svg_text(
            236.0,
            _y_position(float(residual.value[-1]), residual_panel) + 2.5,
            "iteration 10 · 0.004986",
            size=7.3,
            weight=600,
        )
    )

    for trace, panel, formatter in (
        (temperature, temperature_panel, lambda value: f"{value:.4f} keV"),
        (current, current_panel, lambda value: f"{value:.1f} A"),
    ):
        y = _y_position(float(trace.value[-1]), panel)
        lines.append(
            f'<line x1="230" y1="{y:.2f}" x2="238" y2="{y:.2f}" '
            'class="trace" stroke-width="0.7"/>'
        )
        lines.append(
            _svg_text(
                241.0,
                y + 2.5,
                formatter(float(trace.value[-1])),
                size=7.5,
                weight=600,
            )
        )
    lines.append(
        _svg_text(136.5, height - 5.0, "coupling iteration", anchor="middle", size=8.3)
    )
    lines.append("</svg>")
    WINDOW_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _print_summary(traces: Sequence[Trace]) -> None:
    """Emit compact manifest-ready headline metrics."""
    ramp_quantities = (
        "central_ion_temperature",
        "central_electron_temperature",
        "central_electron_density",
        "axis_psi",
    )
    print("rampup_saved_states=11")
    print("rampup_evolved_steps=10")
    for iteration in range(1, 11):
        metrics = []
        for quantity in ramp_quantities:
            trace = _select(
                traces,
                "rampup",
                quantity,
                "absolute-separation",
            )
            metrics.append(f"{quantity}={trace.value[iteration]:.17g}_{trace.unit}")
        print(f"rampup_step_{iteration}_separation " + " ".join(metrics))
    residual = _select(
        traces,
        "window",
        "maximum_exit_residual",
        "exchanged-state",
    )
    print(f"window_iterations={residual.iteration.size}")
    print(f"window_exit_residual={residual.value[-1]:.17g}")
    print(f"window_terminal_contraction={residual.value[-1] / residual.value[-2]:.17g}")
    print(f"trace_rows={sum(trace.value.size for trace in traces)}")
    for path in (TRACE_PATH, RAMPUP_PATH, WINDOW_PATH):
        print(f"artifact={path}")


def main() -> int:
    """Regenerate measurements or render the figures from committed trace data."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--render-only",
        action="store_true",
        help="render from the committed TSV without rerunning the physics",
    )
    arguments = parser.parse_args()
    if arguments.render_only:
        traces = _read_traces()
    else:
        from nova.jax.config import configure_dtypes

        configure_dtypes()
        traces = [*_collect_rampup(), *_collect_window()]
        _write_traces(traces)
    _write_rampup(traces)
    _write_window(traces)
    _print_summary(traces)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
