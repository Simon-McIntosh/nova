"""Measure and draw residual contraction for two coupled exchange windows."""

from __future__ import annotations

import csv
import dataclasses
import math
from pathlib import Path
from typing import Any

import numpy as np

from nova.transport.coupled_window import (
    ExchangeSweepResult,
    TransportSweepReceipt,
    Waveform,
    WindowConfig,
    WindowConvergenceError,
    solve_window,
)
from nova.transport.forward import (
    AchievedBoundaryValues,
    FluxConsumptionLedger,
    ForwardTransportReceipt,
    PlasmaCurrentLedger,
    SolverDiagnostics,
    TransportProvenance,
    TransportRung,
    TransportState,
)

WINDOW_LENGTH = 1.0
CONVERGENCE_TOLERANCE = 1.0e-10
ITERATION_CAP = 180
EQUILIBRIUM_GRID = np.array([0.0, 0.5, 1.0])
TRANSPORT_GRID = np.array([0.0, 0.25, 0.75, 1.0])
CASES = (
    ("weak coupling", 0.002, 1.0),
    ("strong coupling", 0.8, 0.8),
)


def _exchange_waveform(
    time: np.ndarray, radial_points: int, channel: str, value: float
) -> Waveform:
    radial_grid = np.broadcast_to(
        np.linspace(0.0, 1.0, radial_points), (time.size, radial_points)
    )
    return Waveform(
        time=time,
        radial_grid=radial_grid,
        phi_boundary=np.full(time.shape, 2.5),
        axis_reference=np.full(time.shape, -0.35),
        boundary_reference=np.full(time.shape, 0.02),
        values={channel: np.full(radial_grid.shape, value)},
    )


def _transport_receipt(time: np.ndarray) -> TransportSweepReceipt:
    """Return deterministic interval receipts whose ledgers close numerically."""
    rho = np.linspace(0.0, 1.0, 5)
    state = TransportState(
        rho=rho,
        psi=0.2 * rho**2,
        ion_temperature=8.0 - 7.0 * rho,
        electron_temperature=7.0 - 6.0 * rho,
        electron_density=1.2e20 - 2.0e19 * rho,
    )
    current = 1.0e6
    achieved_current = current + 2.0e-6
    receipts = []
    for index in range(time.size - 1):
        boundary_flux = 0.03 * (index + 1)
        resistive_flux = 0.4 * boundary_flux
        internal_flux = boundary_flux - resistive_flux + 5.0e-14
        duration = float(time[index + 1] - time[index])
        receipts.append(
            ForwardTransportReceipt(
                state=state,
                flux_consumption=FluxConsumptionLedger(
                    boundary=boundary_flux,
                    resistive=resistive_flux,
                    internal=internal_flux,
                    mean_axis_voltage=resistive_flux / duration,
                    mean_boundary_voltage=boundary_flux / duration,
                ),
                plasma_current=PlasmaCurrentLedger(
                    requested_initial=current,
                    requested_final=current,
                    achieved_initial=achieved_current,
                    achieved_final=achieved_current,
                ),
                boundary=AchievedBoundaryValues(
                    psi=float(state.psi[-1]),
                    plasma_current=achieved_current,
                    ion_temperature=float(state.ion_temperature[-1]),
                    electron_temperature=float(state.electron_temperature[-1]),
                    electron_density=float(state.electron_density[-1]),
                ),
                diagnostics=SolverDiagnostics(
                    engine_status="converged",
                    steps=1,
                    outer_iterations=1,
                    inner_iterations=1,
                ),
                provenance=TransportProvenance(
                    rung=TransportRung.NATIVE_PSI_DIFFUSION,
                    engine="analytic coupled exchange",
                    engine_version="measured fixture",
                ),
            )
        )
    return TransportSweepReceipt(
        time=time,
        geometry_time=0.5 * (time[:-1] + time[1:]),
        receipts=tuple(receipts),
    )


class AffineExchange:
    """Two-sided affine waveform map with a known combined coupling strength."""

    def __init__(self, coupling: float) -> None:
        self.coupling = coupling
        self.geometry_template = _exchange_waveform(
            EQUILIBRIUM_GRID, 5, "geometry", 0.0
        )
        self.source_template = _exchange_waveform(TRANSPORT_GRID, 7, "source", 0.0)

    @staticmethod
    def _map(
        input_waveform: Waveform,
        template: Waveform,
        input_channel: str,
        output_channel: str,
        gain: float,
        offset: float,
    ) -> Waveform:
        values = np.stack(
            [
                gain
                * input_waveform.sample(
                    float(time), radial_grid=template.radial_grid[index]
                ).values[input_channel]
                + offset
                for index, time in enumerate(template.time)
            ]
        )
        return dataclasses.replace(template, values={output_channel: values})

    def transport(
        self, geometry: Waveform, sample_grid: np.ndarray
    ) -> ExchangeSweepResult:
        source = self._map(
            geometry,
            self.source_template,
            "geometry",
            "source",
            gain=1.0,
            offset=1.0,
        )
        return ExchangeSweepResult(source, _transport_receipt(sample_grid))

    def equilibrium(
        self, source: Waveform, _sample_grid: np.ndarray
    ) -> ExchangeSweepResult:
        geometry = self._map(
            source,
            self.geometry_template,
            "source",
            "geometry",
            gain=self.coupling,
            offset=0.0,
        )
        return ExchangeSweepResult(geometry, {"finite_conservation_receipts": True})


def _measure_case(label: str, coupling: float, damping: float) -> dict[str, Any]:
    exchange = AffineExchange(coupling)
    config = WindowConfig(
        length=WINDOW_LENGTH,
        equilibrium_grid=EQUILIBRIUM_GRID,
        transport_grid=TRANSPORT_GRID,
        iteration_cap=ITERATION_CAP,
        tolerance=CONVERGENCE_TOLERANCE,
    )
    receipt = solve_window(
        exchange.geometry_template,
        exchange.source_template,
        config,
        exchange.equilibrium,
        exchange.transport,
        damping=damping,
    )
    cap_one_config = dataclasses.replace(config, iteration_cap=1)
    try:
        solve_window(
            exchange.geometry_template,
            exchange.source_template,
            cap_one_config,
            exchange.equilibrium,
            exchange.transport,
            damping=damping,
        )
    except WindowConvergenceError as error:
        cap_one = error.convergence
    else:
        raise AssertionError("the cap-one diagnostic unexpectedly converged")

    trace = tuple(
        max(row.values(), default=0.0) for row in receipt.convergence.residual_trace
    )
    if cap_one.maximum_residual != trace[0]:
        raise AssertionError("the cap-one residual differs from the converged trace")
    return {
        "label": label,
        "coupling": coupling,
        "damping": damping,
        "iterations": receipt.convergence.iterations_used,
        "contraction": receipt.convergence.contraction_estimate,
        "trace": trace,
        "cap_one": cap_one.maximum_residual,
    }


def _write_tsv(path: Path, cases: tuple[dict[str, Any], ...]) -> None:
    fields = (
        "window",
        "coupling",
        "damping",
        "tolerance",
        "iteration",
        "maximum_residual",
        "iteration_cap_one",
        "iterations_used",
        "contraction_estimate",
    )
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=fields,
            delimiter="\t",
            lineterminator="\n",
        )
        writer.writeheader()
        for case in cases:
            for iteration, residual in enumerate(case["trace"], start=1):
                writer.writerow(
                    {
                        "window": case["label"],
                        "coupling": f"{case['coupling']:.17g}",
                        "damping": f"{case['damping']:.17g}",
                        "tolerance": f"{CONVERGENCE_TOLERANCE:.17e}",
                        "iteration": iteration,
                        "maximum_residual": f"{residual:.17e}",
                        "iteration_cap_one": str(iteration == 1).lower(),
                        "iterations_used": case["iterations"],
                        "contraction_estimate": f"{case['contraction']:.17e}",
                    }
                )


def _svg_text(
    x: float,
    y: float,
    text: str,
    *,
    size: float = 10.0,
    anchor: str = "start",
    fill: str = "#202020",
    weight: int = 400,
    rotate: int | None = None,
) -> str:
    transform = f' transform="rotate({rotate} {x:.2f} {y:.2f})"' if rotate else ""
    return (
        f'<text x="{x:.2f}" y="{y:.2f}" text-anchor="{anchor}" '
        f'font-family="system-ui, sans-serif" font-size="{size:.1f}" '
        f'font-weight="{weight}" fill="{fill}"{transform}>{text}</text>'
    )


def _svg_line(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    *,
    stroke: str,
    width: float,
) -> str:
    return (
        f'<line x1="{x1:.2f}" y1="{y1:.2f}" '
        f'x2="{x2:.2f}" y2="{y2:.2f}" '
        f'stroke="{stroke}" stroke-width="{width:.1f}"/>'
    )


def _svg_circle(
    x: float,
    y: float,
    radius: float,
    *,
    fill: str,
    stroke: str | None = None,
    width: float | None = None,
) -> str:
    outline = ""
    if stroke is not None and width is not None:
        outline = f' stroke="{stroke}" stroke-width="{width:.1f}"'
    return (
        f'<circle cx="{x:.2f}" cy="{y:.2f}" r="{radius:.1f}" fill="{fill}"{outline}/>'
    )


def _write_svg(path: Path, cases: tuple[dict[str, Any], ...]) -> None:
    height = 220
    left, right, top, bottom = 49.0, 308.0, 31.0, 181.0
    maximum_iteration = max(case["iterations"] for case in cases)
    minimum_residual = 1.0e-11
    residual_decades = -math.log10(minimum_residual)

    def x_position(iteration: int | float) -> float:
        return left + math.log10(iteration) / math.log10(maximum_iteration) * (
            right - left
        )

    def y_position(residual: float) -> float:
        clipped = max(residual, minimum_residual)
        decades = -math.log10(clipped)
        return top + decades / residual_decades * (bottom - top)

    lines = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="320" height="220" '
        'viewBox="0 0 320 220" role="img" '
        'aria-labelledby="window-contraction-title window-contraction-desc">',
        (
            '<title id="window-contraction-title">Residual decay of weakly and '
            "strongly coupled windows</title>"
        ),
        (
            '<desc id="window-contraction-desc">Maximum exchanged-field residual '
            "by iteration on a logarithmic ordinate. The weak window converges in "
            "five iterations; the strongly coupled damped window converges in one "
            "hundred twenty-four. The shared first point marks coupling truncated "
            "at one iteration.</desc>"
        ),
    ]
    for exponent in range(0, 11, 2):
        y = y_position(10.0 ** (-exponent))
        lines.append(_svg_line(left, y, right, y, stroke="#d5d5d5", width=0.7))
        superscript = str(exponent).translate(str.maketrans("0123456789", "⁰¹²³⁴⁵⁶⁷⁸⁹"))
        tick = "1" if exponent == 0 else f"10⁻{superscript}"
        lines.append(_svg_text(left - 6.0, y + 3.2, tick, size=8.8, anchor="end"))

    lines.extend(
        [
            _svg_line(left, top, left, bottom, stroke="#202020", width=1.0),
            _svg_line(left, bottom, right, bottom, stroke="#202020", width=1.0),
        ]
    )
    for iteration in (1, 2, 5, 10, 20, 50, 100):
        x = x_position(iteration)
        lines.append(_svg_line(x, bottom, x, bottom + 4.0, stroke="#202020", width=0.8))
        lines.append(
            _svg_text(x, bottom + 14.0, str(iteration), size=8.8, anchor="middle")
        )

    styles = {
        "weak coupling": ("#151515", "4 3"),
        "strong coupling": ("#666666", "none"),
    }
    for case in reversed(cases):
        stroke, dash = styles[case["label"]]
        points = " ".join(
            f"{x_position(iteration):.2f},{y_position(residual):.2f}"
            for iteration, residual in enumerate(case["trace"], start=1)
        )
        lines.append(
            f'<polyline points="{points}" fill="none" stroke="{stroke}" '
            f'stroke-width="1.8" stroke-dasharray="{dash}" stroke-linejoin="round" '
            'stroke-linecap="round" vector-effect="non-scaling-stroke"/>'
        )

    cap_x = x_position(1)
    cap_y = y_position(cases[0]["cap_one"])
    lines.extend(
        [
            _svg_circle(
                cap_x,
                cap_y,
                5.2,
                fill="none",
                stroke="#666666",
                width=1.4,
            ),
            _svg_circle(cap_x, cap_y, 2.4, fill="#151515"),
            _svg_line(
                cap_x + 3.0,
                cap_y - 2.0,
                cap_x + 10.0,
                cap_y - 10.0,
                stroke="#333333",
                width=0.8,
            ),
            _svg_text(cap_x + 12.0, cap_y - 11.0, "iteration cap = 1", size=9.5),
        ]
    )

    weak = next(case for case in cases if case["label"] == "weak coupling")
    weak_end_x = x_position(weak["iterations"])
    weak_end_y = y_position(weak["trace"][-1])
    weak_metrics = (
        f"{weak['iterations']} iterations · contraction {weak['contraction']:.4f}"
    )
    lines.extend(
        [
            _svg_line(
                weak_end_x,
                weak_end_y,
                weak_end_x + 11.0,
                weak_end_y - 12.0,
                stroke="#151515",
                width=0.8,
            ),
            _svg_text(
                weak_end_x + 13.0,
                weak_end_y - 16.0,
                "weak coupling",
                size=10.2,
                weight=600,
            ),
            _svg_text(
                weak_end_x + 13.0,
                weak_end_y - 5.0,
                weak_metrics,
                size=8.7,
            ),
        ]
    )

    strong = next(case for case in cases if case["label"] == "strong coupling")
    label_iteration = 34
    strong_y = y_position(strong["trace"][label_iteration - 1])
    strong_metrics = (
        f"{strong['iterations']} iterations · contraction {strong['contraction']:.3f}"
    )
    lines.extend(
        [
            _svg_text(
                right - 3.0,
                strong_y - 14.0,
                "strong coupling",
                size=10.2,
                anchor="end",
                fill="#555555",
                weight=600,
            ),
            _svg_text(
                right - 3.0,
                strong_y - 3.0,
                strong_metrics,
                size=8.7,
                anchor="end",
                fill="#555555",
            ),
        ]
    )

    lines.extend(
        [
            _svg_text(
                (left + right) / 2.0,
                height - 7.0,
                "exchange iteration (log spacing)",
                size=9.8,
                anchor="middle",
            ),
            _svg_text(
                11.5,
                (top + bottom) / 2.0,
                "maximum exchanged-field residual",
                size=9.2,
                anchor="middle",
                rotate=-90,
            ),
            "</svg>",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    output_dir = Path(__file__).resolve().parent
    figure = (
        output_dir.parents[1]
        / "docs"
        / "figures"
        / "flux-function-forward-transport"
        / "window-contraction.svg"
    )
    cases = tuple(_measure_case(*case) for case in CASES)
    _write_tsv(output_dir / "residuals.tsv", cases)
    _write_svg(figure, cases)
    for case in cases:
        print(
            f"{case['label']}: iterations={case['iterations']}, "
            f"contraction={case['contraction']:.6g}, "
            f"exit_residual={case['trace'][-1]:.6g}, "
            f"cap_one_residual={case['cap_one']:.6g}"
        )


if __name__ == "__main__":
    main()
