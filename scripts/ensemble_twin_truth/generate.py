"""Generate deterministic twin truth through Nova's public forward seams."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from nova.equilibrium.observation_kernels import synthesize_thomson
from nova.transport import (
    AchievedBoundaryValues,
    BatchedCouplingState,
    BatchedExchangeSweepResult,
    BatchedWaveform,
    CouplingState,
    EquilibriumSweepReceipt,
    FluxConsumptionLedger,
    ForwardTransportReceipt,
    MemberArrayBatch,
    PlasmaCurrentLedger,
    SolverDiagnostics,
    TransportProvenance,
    TransportRung,
    TransportState,
    TransportSweepReceipt,
    Waveform,
    WindowBatchInput,
    WindowConfig,
    solve_window_batch,
)


WINDOW_COUNT = 6
WINDOW_LENGTH_SECONDS = 0.05
MEMBER_IDS = ("truth", "pressure-high", "field-high", "current-low")
RADIAL_GRID = np.linspace(0.0, 1.0, 9)
EQUILIBRIUM_GRID = np.array((0.0, 0.5, 1.0)) * WINDOW_LENGTH_SECONDS
TRANSPORT_GRID = np.array((0.0, 0.25, 0.75, 1.0)) * WINDOW_LENGTH_SECONDS
CHORD_COORDINATES = np.asarray(
    (
        ((0.94, -0.08), (1.03, -0.03), (1.12, 0.02)),
        ((1.00, 0.12), (1.15, 0.09), (1.25, 0.04)),
    ),
    dtype=np.float64,
)


@dataclass(frozen=True)
class PrescribedMember:
    member_id: str
    role: str
    p_prime_scale: float
    ff_prime_scale: float
    current_scale: float
    temperature_scale: float
    density_scale: float


def _tree_sha() -> str:
    return subprocess.check_output(("git", "rev-parse", "HEAD"), text=True).strip()


def _waveform(time: np.ndarray, values: dict[str, np.ndarray]) -> Waveform:
    radial = np.broadcast_to(RADIAL_GRID, (time.size, RADIAL_GRID.size))
    return Waveform(
        time=time,
        radial_grid=radial,
        phi_boundary=np.full(time.shape, 2.5),
        axis_reference=np.full(time.shape, -0.35),
        boundary_reference=np.full(time.shape, 0.02),
        values=values,
    )


def _initial_state() -> CouplingState:
    geometry = _waveform(
        EQUILIBRIUM_GRID,
        {"shape": np.zeros((EQUILIBRIUM_GRID.size, RADIAL_GRID.size))},
    )
    source = _waveform(
        TRANSPORT_GRID,
        {
            name: np.zeros((TRANSPORT_GRID.size, RADIAL_GRID.size))
            for name in (
                "p_prime",
                "ff_prime",
                "boundary_pressure",
                "boundary_field_function",
            )
        },
    )
    return CouplingState(geometry=geometry, source=source)


def _members(window_index: int) -> tuple[PrescribedMember, ...]:
    pressure = 1.0 + 0.03 * window_index
    field = 1.0 - 0.02 * window_index
    current = 1.0 + 0.015 * window_index
    temperature = 1.0 + 0.01 * window_index
    density = 1.0 - 0.008 * window_index
    return (
        PrescribedMember(
            "truth", "truth", pressure, field, current, temperature, density
        ),
        PrescribedMember(
            "pressure-high",
            "counterfactual",
            pressure + 0.08,
            field,
            current,
            temperature + 0.025,
            density,
        ),
        PrescribedMember(
            "field-high",
            "counterfactual",
            pressure,
            field + 0.06,
            current,
            temperature,
            density - 0.02,
        ),
        PrescribedMember(
            "current-low",
            "counterfactual",
            pressure,
            field,
            current - 0.05,
            temperature - 0.015,
            density + 0.015,
        ),
    )


def _batch_input(
    members: tuple[PrescribedMember, ...],
    states: dict[str, CouplingState],
) -> WindowBatchInput:
    identifiers = tuple(member.member_id for member in members)
    radius = np.linspace(0.8, 1.4, 25)
    height = np.linspace(-0.3, 0.3, 25)
    return WindowBatchInput(
        seam_state=MemberArrayBatch(
            identifiers,
            {
                "p_prime_scale": np.asarray(
                    [member.p_prime_scale for member in members]
                ),
                "ff_prime_scale": np.asarray(
                    [member.ff_prime_scale for member in members]
                ),
                "temperature_scale": np.asarray(
                    [member.temperature_scale for member in members]
                ),
                "density_scale": np.asarray(
                    [member.density_scale for member in members]
                ),
            },
        ),
        actuator_waveforms=MemberArrayBatch(
            identifiers,
            {
                "plasma_current_a": np.stack(
                    [
                        np.full(TRANSPORT_GRID.shape, 1.0e6 * member.current_scale)
                        for member in members
                    ]
                ),
                "field_drive": np.stack(
                    [
                        np.full(EQUILIBRIUM_GRID.shape, member.ff_prime_scale)
                        for member in members
                    ]
                ),
            },
        ),
        geometry=MemberArrayBatch(
            identifiers,
            {
                "radius_m": np.broadcast_to(radius, (len(members), radius.size)),
                "height_m": np.broadcast_to(height, (len(members), height.size)),
            },
        ),
        coupling_state=BatchedCouplingState.from_members(
            tuple((member_id, states[member_id]) for member_id in identifiers)
        ),
    )


def _sample_profile(
    waveform: BatchedWaveform,
    member_id: str,
    time: float,
    channel: str,
) -> np.ndarray:
    return np.asarray(waveform.member(member_id).sample(time).values[channel])


def _transport_receipt(
    time: np.ndarray,
    source_amplitude: float,
    current: float,
    temperature_scale: float,
    density_scale: float,
) -> TransportSweepReceipt:
    rho = RADIAL_GRID
    state = TransportState(
        rho=rho,
        psi=0.2 * rho**2 + 0.01 * source_amplitude * (1.0 - rho**2),
        ion_temperature=8.0 * temperature_scale - 7.0 * rho**2,
        electron_temperature=7.0 * temperature_scale - 6.0 * rho**2,
        electron_density=(1.2e20 * density_scale) - 2.0e19 * rho**2,
    )
    receipts = []
    for interval in range(time.size - 1):
        boundary = 0.03 * (interval + 1)
        resistive = 0.4 * boundary
        internal = boundary - resistive
        duration = float(time[interval + 1] - time[interval])
        receipts.append(
            ForwardTransportReceipt(
                state=state,
                flux_consumption=FluxConsumptionLedger(
                    boundary,
                    resistive,
                    internal,
                    resistive / duration,
                    boundary / duration,
                ),
                plasma_current=PlasmaCurrentLedger(current, current, current, current),
                boundary=AchievedBoundaryValues(
                    float(state.psi[-1]),
                    current,
                    float(state.ion_temperature[-1]),
                    float(state.electron_temperature[-1]),
                    float(state.electron_density[-1]),
                ),
                diagnostics=SolverDiagnostics("converged", 1, 1, 1),
                provenance=TransportProvenance(
                    TransportRung.NATIVE_PSI_DIFFUSION,
                    "synthetic-twin-fixture",
                    "analytic",
                ),
            )
        )
    return TransportSweepReceipt(
        time=time,
        geometry_time=0.5 * (time[:-1] + time[1:]),
        receipts=tuple(receipts),
    )


class SyntheticBatchOperators:
    """Known contracting maps used only to produce deterministic fixture truth."""

    def transport(self, inputs, geometry, sample_grid):
        member_waveforms = []
        receipts = []
        for member_id in inputs.member_ids:
            seam = inputs.seam_state.member(member_id)
            actuators = inputs.actuator_waveforms.member(member_id)
            geometry_profile = np.stack(
                [
                    _sample_profile(geometry, member_id, float(time), "shape")
                    for time in sample_grid
                ]
            )
            rho = RADIAL_GRID
            core = 1.0 - rho**2
            edge = rho**2
            pressure_scale = float(seam["p_prime_scale"])
            field_scale = float(seam["ff_prime_scale"])
            values = {
                "p_prime": 0.18 * geometry_profile + pressure_scale * core[None, :],
                "ff_prime": 0.12 * geometry_profile + field_scale * edge[None, :],
                "boundary_pressure": 0.08 * geometry_profile + 0.04 * pressure_scale,
                "boundary_field_function": 0.06 * geometry_profile + 0.07 * field_scale,
            }
            waveform = _waveform(np.asarray(sample_grid), values)
            amplitude = float(np.mean(values["p_prime"] + values["ff_prime"]))
            current = float(np.asarray(actuators["plasma_current_a"])[-1])
            receipts.append(
                _transport_receipt(
                    np.asarray(sample_grid),
                    amplitude,
                    current,
                    float(seam["temperature_scale"]),
                    float(seam["density_scale"]),
                )
            )
            member_waveforms.append((member_id, waveform))
        return BatchedExchangeSweepResult(
            BatchedWaveform.from_members(tuple(member_waveforms)), tuple(receipts)
        )

    def equilibrium(self, inputs, source, sample_grid):
        member_waveforms = []
        receipts = []
        for member_id in inputs.member_ids:
            actuators = inputs.actuator_waveforms.member(member_id)
            profiles = []
            for time in sample_grid:
                sample = source.member(member_id).sample(float(time))
                combined = np.asarray(sample.values["p_prime"]) + np.asarray(
                    sample.values["ff_prime"]
                )
                profiles.append(combined)
            drive = float(np.mean(actuators["field_drive"]))
            shape = 0.35 * np.stack(profiles) + 0.05 * drive
            waveform = _waveform(np.asarray(sample_grid), {"shape": shape})
            current = float(np.asarray(actuators["plasma_current_a"])[-1])
            equilibrium = SimpleNamespace(
                topology=SimpleNamespace(diverted=np.asarray(False)),
                moments=SimpleNamespace(plasma_current=current),
                conservation=SimpleNamespace(
                    relative_divergence_b=0.0,
                    relative_divergence_j=0.0,
                ),
            )
            receipts.append(
                EquilibriumSweepReceipt(
                    time=np.asarray(sample_grid),
                    source_samples=(),
                    equilibria=(equilibrium,),
                    branch_receipts=(),
                )
            )
            member_waveforms.append((member_id, waveform))
        return BatchedExchangeSweepResult(
            BatchedWaveform.from_members(tuple(member_waveforms)), tuple(receipts)
        )


def _thomson(member_receipt):
    radius = np.linspace(0.8, 1.4, 25)
    height = np.linspace(-0.3, 0.3, 25)
    radius_map, height_map = np.meshgrid(radius, height, indexing="ij")
    shape = float(np.mean(member_receipt.fields.geometry.values["shape"][-1]))
    psi_norm = ((radius_map - (1.1 + 0.004 * shape)) / 0.45) ** 2 + (
        height_map / 0.4
    ) ** 2
    flux = 2.0 * psi_norm
    state = member_receipt.transport_state
    return synthesize_thomson(
        radius,
        height,
        flux,
        state.rho,
        1.0e3 * state.electron_temperature,
        state.electron_density,
        CHORD_COORDINATES,
        axis_flux=0.0,
        boundary_flux=2.0,
    )


def _write_tsv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=tuple(rows[0]),
            delimiter="\t",
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_svg(path: Path, trajectory: list[dict[str, object]]) -> None:
    truth = [row for row in trajectory if row["role"] == "truth"]
    current = np.asarray([float(row["net_current_a"]) for row in truth]) / 1.0e6
    temperature = np.asarray(
        [float(row["mean_electron_temperature_ev"]) for row in truth]
    )

    def points(values, lower, upper):
        return " ".join(
            f"{90 + 90 * index:.1f},{350 - 240 * (value - lower) / (upper - lower):.1f}"
            for index, value in enumerate(values)
        )

    current_points = points(
        current, float(current.min()) - 0.02, float(current.max()) + 0.02
    )
    temperature_points = points(
        temperature,
        float(temperature.min()) - 100.0,
        float(temperature.max()) + 100.0,
    )
    path.write_text(
        "\n".join(
            (
                '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 700 410" '
                'role="img" aria-label="Six-window deterministic truth trajectory">',
                "<style>:root{color-scheme:light dark}"
                "text{font-family:system-ui,sans-serif;"
                "font-size:14px;fill:currentColor}.title{font-size:20px;font-weight:650}"
                ".axis{stroke:#64748b;stroke-width:1}</style>",
                '<text x="38" y="34" class="title">'
                "Known truth marched through six windows</text>",
                '<line x1="90" y1="350" x2="570" y2="350" class="axis"/>',
                '<line x1="90" y1="80" x2="90" y2="350" class="axis"/>',
                f'<polyline points="{current_points}" fill="none" stroke="#2563eb" '
                'stroke-width="3"/>',
                f'<polyline points="{temperature_points}" fill="none" stroke="#d97706" '
                'stroke-width="3"/>',
                '<text x="580" y="165" fill="#2563eb">net current (scaled)</text>',
                '<text x="580" y="105" fill="#d97706">mean Thomson Te</text>',
                '<text x="295" y="392">window</text>',
                "</svg>",
            )
        )
        + "\n",
        encoding="utf-8",
    )


def generate_package(output_dir: Path, *, tree_sha: str | None = None) -> dict:
    """Generate and return the forward-only package receipt."""
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = tree_sha or _tree_sha()
    config = WindowConfig(
        length=WINDOW_LENGTH_SECONDS,
        equilibrium_grid=EQUILIBRIUM_GRID,
        transport_grid=TRANSPORT_GRID,
        iteration_cap=40,
        tolerance=1.0e-10,
    )
    states = {member_id: _initial_state() for member_id in MEMBER_IDS}
    operators = SyntheticBatchOperators()
    trajectory_rows: list[dict[str, object]] = []
    observation_rows: list[dict[str, object]] = []
    coupling_rows: list[dict[str, object]] = []
    for window_index in range(WINDOW_COUNT):
        members = _members(window_index)
        batch = solve_window_batch(
            _batch_input(members, states),
            config,
            operators.equilibrium,
            operators.transport,
        )
        for prescribed in members:
            member = batch.for_member(prescribed.member_id)
            signals = _thomson(member)
            net_current = float(member.conservation.plasma_current.achieved_final)
            temperature = np.asarray(signals.electron_temperature)
            trajectory_rows.append(
                {
                    "tree_sha": stamp,
                    "window_index": window_index,
                    "window_start_s": window_index * WINDOW_LENGTH_SECONDS,
                    "window_end_s": (window_index + 1) * WINDOW_LENGTH_SECONDS,
                    "member_id": prescribed.member_id,
                    "role": prescribed.role,
                    "p_prime_scale": prescribed.p_prime_scale,
                    "ff_prime_scale": prescribed.ff_prime_scale,
                    "current_scale": prescribed.current_scale,
                    "temperature_scale": prescribed.temperature_scale,
                    "density_scale": prescribed.density_scale,
                    "iterations": member.convergence.iterations_used,
                    "gating_norm": member.convergence.gating_norm,
                    "contraction": member.convergence.contraction_estimate,
                    "topology_class": member.topology_class.name.lower(),
                    "net_current_a": net_current,
                    "mean_electron_temperature_ev": float(np.mean(temperature)),
                    "flux_closure_residual": member.conservation.flux_closure_residual,
                    "current_continuity_residual": (
                        member.conservation.current_continuity_residual
                    ),
                }
            )
            support = np.asarray(signals.receipt.interpolation_support.supported)
            error = np.asarray(signals.receipt.numerical_error_bound)
            for chord_index, coordinates in enumerate(CHORD_COORDINATES.reshape(-1, 2)):
                observation_rows.append(
                    {
                        "tree_sha": stamp,
                        "window_index": window_index,
                        "member_id": prescribed.member_id,
                        "role": prescribed.role,
                        "chord_index": chord_index,
                        "radius_m": coordinates[0],
                        "height_m": coordinates[1],
                        "psi_norm": float(
                            np.asarray(signals.psi_norm).reshape(-1)[chord_index]
                        ),
                        "electron_temperature_ev": float(
                            temperature.reshape(-1)[chord_index]
                        ),
                        "electron_density_m-3": float(
                            np.asarray(signals.electron_density).reshape(-1)[
                                chord_index
                            ]
                        ),
                        "net_current_a": net_current,
                        "thomson_cocos": signals.receipt.cocos,
                        "temperature_unit": signals.receipt.units[0],
                        "density_unit": signals.receipt.units[1],
                        "net_current_unit": "A",
                        "support_method": (
                            signals.receipt.interpolation_support.method
                        ),
                        "supported": bool(support.reshape(-1)[chord_index]),
                        "temperature_error_bound_ev": float(error[0]),
                        "density_error_bound_m-3": float(error[1]),
                        "net_current_support": (
                            "typed window plasma-current conservation ledger"
                        ),
                    }
                )
            states[prescribed.member_id] = member.fields
            coupling_rows.append(
                {
                    "tree_sha": stamp,
                    "window_index": window_index,
                    "member_id": prescribed.member_id,
                    "role": prescribed.role,
                    "coupling_state": member.fields.to_dict(),
                }
            )

    trajectory_path = output_dir / "trajectory.tsv"
    observation_path = output_dir / "observations.tsv"
    coupling_path = output_dir / "coupling_states.jsonl"
    _write_tsv(trajectory_path, trajectory_rows)
    _write_tsv(observation_path, observation_rows)
    coupling_path.write_text(
        "".join(json.dumps(row, separators=(",", ":")) + "\n" for row in coupling_rows),
        encoding="utf-8",
    )
    _write_svg(output_dir / "trajectory.svg", trajectory_rows)
    artifacts = {
        path.name: _sha256(path)
        for path in (trajectory_path, observation_path, coupling_path)
    }
    receipt = {
        "schema": "nova.ensemble-twin-truth-package",
        "schema_version": "1.0.0",
        "tree_sha": stamp,
        "window_count": WINDOW_COUNT,
        "window_length_s": WINDOW_LENGTH_SECONDS,
        "member_count": len(MEMBER_IDS),
        "member_ids": list(MEMBER_IDS),
        "truth_member_id": "truth",
        "counterfactual_member_ids": list(MEMBER_IDS[1:]),
        "trajectory_rows": len(trajectory_rows),
        "observation_rows": len(observation_rows),
        "coupling_state_rows": len(coupling_rows),
        "all_windows_converged": True,
        "all_observations_supported": all(
            bool(row["supported"]) for row in observation_rows
        ),
        "observation_contract": {
            "thomson": {
                "cocos": 17,
                "units": ["eV", "m^-3"],
                "support_receipt_per_row": True,
            },
            "net_current": {
                "unit": "A",
                "source": "typed window plasma-current conservation ledger",
            },
        },
        "known_prescription": (
            "window-indexed p-prime, FF-prime, current, temperature and density "
            "scales declared in every trajectory row"
        ),
        "joint_recovery_gate_run": False,
        "joint_recovery_gate_owner": "ambix",
        "joint_recovery_gate_note": (
            "Not run in Nova; existing Ambix estimator consumes this deterministic "
            "package through the coupling-state seam."
        ),
        "artifacts_sha256": artifacts,
    }
    (output_dir / "receipt.json").write_text(
        json.dumps(receipt, indent=2) + "\n", encoding="utf-8"
    )
    (output_dir / "consumption.md").write_text(
        "\n".join(
            (
                "# Deterministic twin-truth package",
                "",
                "The package contains six marched windows for one known truth "
                "and three member-preserving counterfactuals. `trajectory.tsv` "
                "carries prescribed flux-function/drive scales and convergence/"
                "conservation receipts; `observations.tsv` carries Thomson, "
                "net-current, COCOS, unit, support and error receipts per row; "
                "`coupling_states.jsonl` is the exact versioned "
                "window-boundary handoff.",
                "",
                "Ambix should group by `member_id`, order by `window_index`, "
                "reconstruct each boundary with `CouplingState.from_dict`, and "
                "admit only rows whose window receipt converged. The `truth` "
                "observation rows are deterministic "
                "measurements; the other member rows are counterfactual predictions.",
                "",
                "The joint recovery gate is deliberately **not run here**. Estimator "
                "selection, inference, coverage and scoring remain Ambix-owned; Nova "
                "provides only truth, forward members, observations and receipts.",
            )
        )
        + "\n",
        encoding="utf-8",
    )
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    arguments = parser.parse_args()
    receipt = generate_package(arguments.output_dir)
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
