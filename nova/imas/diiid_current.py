"""Bind DIII-D conductor currents to the measured circuit and response matrix."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Mapping, Sequence

import jax.numpy as jnp
import numpy as np

from nova.equilibrium.conductor_current import (
    ConductorCurrentDeclaration,
    CurrentResolution,
    CurrentTier,
    StaticCurrentRelation,
    UnknownCurrentPrior,
    resolve_conductor_currents,
)
from nova.database.filepath import diiid_machine_source
from nova.equilibrium.forward import ForwardProfile

from .diiid_description import (
    CIRCUIT_DRIVEN_CONDUCTORS,
    DiiidDescription,
    PF_ACTIVE_CIRCUIT,
    PF_ACTIVE_SUPPLY,
    POLOIDAL_CONDUCTORS,
    active_coil_response_from_imas,
)


CIRCUIT_BYPASS_PRIOR_CONDUCTORS = ("ECOILB", "E567DN", "E89UP")
CIRCUIT_BYPASS_RELATIONS = {
    "E567UP": StaticCurrentRelation(
        source="ECOILA",
        scale=0.9929,
        relative_residual=0.00570,
        provenance=(
            "fit once over 480256 independent samples from the measured ohmic trace"
        ),
        transfer_caveat=(
            "calibration and inference use different pulses; the measured relation "
            "residual is transferred without a same-pulse refit"
        ),
    ),
    "E89DN": StaticCurrentRelation(
        source="ECOILA",
        scale=1.0165,
        relative_residual=0.00706,
        provenance=(
            "fit once over 480256 independent samples from the measured ohmic trace"
        ),
        transfer_caveat=(
            "calibration and inference use different pulses; the measured relation "
            "residual is transferred without a same-pulse refit"
        ),
    ),
}
CIRCUIT_RELATIONS = {
    drive.conductor: StaticCurrentRelation(
        source=PF_ACTIVE_CIRCUIT.source_conductor,
        scale=drive.gain,
        relative_residual=drive.uncertainty.residual_rms_fraction,
        provenance=(
            f"{PF_ACTIVE_CIRCUIT.provenance}; leave-one-shot-out R-squared "
            f"{drive.uncertainty.leave_one_shot_out_r_squared:.6f}; residual RMS "
            f"{drive.uncertainty.residual_rms_a_turn:.6f} A.turn and sample "
            "standard deviation "
            f"{drive.uncertainty.residual_sample_standard_deviation_a_turn:.6f} "
            "A.turn"
        ),
        transfer_caveat="; ".join(PF_ACTIVE_CIRCUIT.caveats),
    )
    for drive in PF_ACTIVE_CIRCUIT.drives
}
DEFAULT_ACTIVE_COIL_ENTRY = diiid_machine_source()
DEFAULT_ACTIVE_COIL_DD_VERSION = "3.41.0"


@dataclass(frozen=True)
class DiiidCurrentAdapter:
    """Complete DIII-D profile plus its current parameterization and provenance."""

    profile: ForwardProfile
    resolution: CurrentResolution
    response_receipt: dict[str, Any]


def current_declarations(
    shipped_names: Sequence[str],
    circuit_bypass_priors: Mapping[str, UnknownCurrentPrior] | None = None,
    *,
    use_circuit: bool = True,
) -> tuple[ConductorCurrentDeclaration, ...]:
    """Return the measured circuit declarations or an explicit diagnostic bypass."""

    names = tuple(str(name) for name in shipped_names)
    if set(names) != set(POLOIDAL_CONDUCTORS):
        raise ValueError(
            "DIII-D shipped response must contain its 19 poloidal channels"
        )
    if not use_circuit and (
        circuit_bypass_priors is None
        or set(circuit_bypass_priors) != set(CIRCUIT_BYPASS_PRIOR_CONDUCTORS)
    ):
        raise ValueError(
            "DIII-D circuit-bypass priors must name ECOILB, E567DN, and E89UP exactly"
        )
    declared = [
        ConductorCurrentDeclaration(
            name=name,
            tier=CurrentTier.KNOWN,
            provenance=f"same-frame shipped magnetics_{name} channel",
        )
        for name in names
    ]
    for name in CIRCUIT_DRIVEN_CONDUCTORS:
        if use_circuit:
            declared.append(
                ConductorCurrentDeclaration(
                    name=name,
                    tier=CurrentTier.KNOWABLE,
                    relation=CIRCUIT_RELATIONS[name],
                    provenance="fit-once machine-description circuit drive",
                )
            )
        elif name in CIRCUIT_BYPASS_RELATIONS:
            declared.append(
                ConductorCurrentDeclaration(
                    name=name,
                    tier=CurrentTier.KNOWABLE,
                    relation=CIRCUIT_BYPASS_RELATIONS[name],
                    provenance="independent fit-once ohmic relation",
                )
            )
        else:
            assert circuit_bypass_priors is not None
            declared.append(
                ConductorCurrentDeclaration(
                    name=name,
                    tier=CurrentTier.UNKNOWN,
                    prior=circuit_bypass_priors[name],
                    provenance=circuit_bypass_priors[name].provenance,
                )
            )
    return tuple(declared)


def circuit_current_map(shipped_current_a: Mapping[str, float]) -> dict[str, float]:
    """Drive every circuit-represented conductor from shipped ECOILA ampere-turns."""

    try:
        source = float(shipped_current_a[PF_ACTIVE_CIRCUIT.source_conductor])
    except KeyError as error:
        raise ValueError("shipped current map is missing ECOILA") from error
    return PF_ACTIVE_CIRCUIT.currents(source)


def shipped_current_at(
    row: Mapping[str, Any],
    description: DiiidDescription,
    names: Sequence[str],
    time_ms: float,
) -> dict[str, float]:
    """Interpolate only shipped current channels in response-column order."""

    source_time = np.asarray(row["magnetics_time"], dtype=float)
    by_name = {item.name: item for item in description.conductors}
    result = {}
    for name in names:
        conductor = by_name[name]
        values = np.asarray(row[conductor.input_column], dtype=float)
        valid = np.isfinite(source_time) & np.isfinite(values)
        if np.count_nonzero(valid) < 2:
            raise ValueError(f"{conductor.input_column} has fewer than two samples")
        current_ka = np.interp(time_ms, source_time[valid], values[valid])
        result[name] = 1000.0 * float(current_ka) * conductor.turns.applied_multiplier
    return result


def resolve_diiid_currents(
    shipped_names: Sequence[str],
    shipped_current_a: Mapping[str, float],
    circuit_bypass_priors: Mapping[str, UnknownCurrentPrior] | None = None,
    *,
    use_circuit: bool = True,
) -> CurrentResolution:
    """Resolve 24 currents with the measured circuit as the default authority."""

    names = tuple(str(name) for name in shipped_names) + CIRCUIT_DRIVEN_CONDUCTORS
    return resolve_conductor_currents(
        names,
        current_declarations(
            shipped_names,
            circuit_bypass_priors,
            use_circuit=use_circuit,
        ),
        shipped_current_a,
    )


def _completed_target(
    target,
    names: tuple[str, ...],
    *,
    label: str,
    active_coil_entry: str | Path,
    active_coil_dd_version: str,
):
    """Append the circuit-driven columns to one target's conductor response.

    Every target the solve contracts against the conductor vector carries the
    same columns: a target left at the shipped order is a shorter operand than
    the current vector, which the contraction refuses.
    """

    if target.source_target.shape[1] != len(names):
        raise ValueError(
            f"{label} response columns do not match shipped conductor order"
        )
    circuit_names, response, receipt = active_coil_response_from_imas(
        active_coil_entry,
        active_coil_dd_version,
        CIRCUIT_DRIVEN_CONDUCTORS,
        np.asarray(target.coordinate)[:, 0],
        np.asarray(target.coordinate)[:, 1],
    )
    if circuit_names != CIRCUIT_DRIVEN_CONDUCTORS:
        raise RuntimeError("active-coil response did not preserve requested order")
    return (
        replace(
            target,
            source_target=jnp.column_stack(
                (target.source_target, jnp.asarray(response.T))
            ),
        ),
        receipt,
    )


def complete_profile_current_adapter(
    profile: ForwardProfile,
    *,
    shipped_names: Sequence[str],
    shipped_current_a: Mapping[str, float],
    circuit_bypass_priors: Mapping[str, UnknownCurrentPrior] | None = None,
    use_circuit: bool = True,
    active_coil_entry: str | Path = DEFAULT_ACTIVE_COIL_ENTRY,
    active_coil_dd_version: str = DEFAULT_ACTIVE_COIL_DD_VERSION,
) -> DiiidCurrentAdapter:
    """Append circuit-driven columns and bind the ordered current resolution.

    The grid, the wall and the sampling nodes are all solved targets, so all
    three carry the completed conductor order. The sampling nodes exist only
    when the booked current route is attached, and completing them here keeps
    the order in one place rather than at each call site that builds a booked
    operator.
    """

    names = tuple(str(name) for name in shipped_names)
    resolution = resolve_diiid_currents(
        names,
        shipped_current_a,
        circuit_bypass_priors,
        use_circuit=use_circuit,
    )
    targets = {}
    receipts = {}
    for label, target in (
        ("grid", profile.operator.grid),
        ("wall", profile.operator.wall),
        ("sample", profile.operator.sample),
    ):
        if target is None:
            continue
        targets[label], receipts[label] = _completed_target(
            target,
            names,
            label=label,
            active_coil_entry=active_coil_entry,
            active_coil_dd_version=active_coil_dd_version,
        )
    operator = replace(
        profile.operator,
        external_current=jnp.asarray(resolution.current(resolution.prior_mean_a)),
        **targets,
    )
    return DiiidCurrentAdapter(
        profile=replace(profile, operator=operator),
        resolution=resolution,
        response_receipt={
            "response_order": list(resolution.names),
            "shipped_count": len(names),
            "complete_count": len(resolution.names),
            "current_authority": (
                "pf_active circuit"
                if use_circuit
                else "diagnostic prior-driven circuit bypass"
            ),
            "pf_active": {
                "supply": PF_ACTIVE_SUPPLY.as_record(),
                "circuit": PF_ACTIVE_CIRCUIT.as_record(),
            },
            "grid": receipts["grid"],
            "wall": receipts["wall"],
            "sample": receipts.get("sample"),
            "sample_rows": (
                None
                if profile.operator.sample is None
                else int(profile.operator.sample.node_number)
            ),
        },
    )
