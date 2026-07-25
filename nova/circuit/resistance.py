"""Data-led calibration of passive-circuit resistance and conductor topology.

The passive linkage matrix is exact from geometry; the resistances are not.  A
machine-description passive element is a bounding box of a welded 3-D structure,
so the nominal ring resistances misstate the true conducting paths, and the
isolated-ring model assumes a topology the structure may not have.

A coil-only (vacuum) interval is where both can be falsified: there is no plasma,
so the measured magnetics minus the drives' static prediction is PURE eddy
signal, the drives are measured, and the only unknowns are a few bounded
parameters -- classical parameter estimation, done standalone so every downstream
consumer inherits a calibrated system instead of re-learning it.

Held-back measured circuits (binding contract): a channel that measures a passive
circuit's current is NEVER an input here.  Its circuit is moved into the passive
set (:func:`~nova.circuit.passive.build_passive_circuit_system` with
``measured_circuits``) so the current is PREDICTED from the remaining drives
through the mutual couplings, and the measurement serves purely as a held-back
fitting and validation target -- the strongest per-circuit test of both L and R
available.

Identifiability ladder (never per-shot, never per-slice): a global scale, then
structure versus cases, then structure regions plus case pairs, then per-case --
each extra tier of bounded positive multipliers accepted only if held-out vacuum
shots improve.  Region assignment uses a machine-agnostic rule on NORMALISED
centroid coordinates of the passive set (fractions of the set's own radial span
and vertical extent), so it transfers across machines with no metre-level
thresholds.

Structure discovery keeps the linkage untouched and tests three structured
hypothesis families against the same data:

* **galvanic case wiring** -- when a measured identity meters a case current
  inside its coil's supply circuit, the case loop sees the winding's terminal
  voltage.  The candidate drive is ``V = g_v dLambda_w/dt + r_w i_winding`` with
  ``Lambda_w`` the winding's geometric flux from the measured drives (a static
  edit of the case row's drive-linkage columns) and the resistive term a
  voltage-type drive.  Recorded approximation: the winding's linkage of PASSIVE
  currents (including the case's own) is dropped from ``V``, because keeping it
  makes the generalised eigenproblem asymmetric; the fitted ``g_v`` / ``r_w`` and
  the per-case resistance multiplier absorb the diagonal part.
* **pair wiring as constraint reductions** -- series (``I_i = I_j``) or
  anti-series (``I_i = -I_j``) merges of measured circuit pairs that move as one,
  expressed as a reduction map ``C`` with ``L -> C' L C``, ``R -> C' R C``,
  drives ``-> C' u``; plus common/differential drive-gain corrections for coil
  pairs the measurements cannot separate.
* **adjacency-restricted galvanic couplings** -- a real vessel is a continuous
  welded shell, so ADJACENT elements share conductor.  Candidate off-diagonal
  resistance stamps ``rho (e_i - e_j)(e_i - e_j)'`` (positive-semidefinite: a
  shared branch of resistance ``rho``) restricted to a nearest-neighbour graph
  whose threshold is normalised by the circuits' own section scales -- a
  dimensionless rule, never a free dense interaction fit.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from nova.circuit.passive import PassiveCircuitSystem, PassiveEigenbasis
from nova.circuit.propagate import zoh_mode_response

#: normalised-radius thresholds of the region rule: below INBOARD is the centre
#: column, above OUTBOARD the outer cylinder; of the remainder, circuits with
#: |z|/max|z| above ENDS are the end plates and the rest mid structures
REGION_INBOARD_RFRAC = 0.25
REGION_OUTBOARD_RFRAC = 0.60
REGION_ENDS_ZFRAC = 0.70

#: the identifiability ladder, fewest to most degrees of freedom
LADDER_LEVELS = ("global", "vessel-case", "regions-casepairs", "regions-percase")

#: bounded positive multipliers (log-space optimisation bounds)
MULTIPLIER_BOUNDS = (0.2, 64.0)


def resistance_group_labels(
    circuits: np.ndarray,
    centroid_r: np.ndarray,
    centroid_z: np.ndarray,
    level: str,
    *,
    case_of: dict[int, str] | None = None,
) -> list[str]:
    """Group label of each passive circuit at one ladder level.

    ``case_of`` maps a coil-case circuit id to the coil it encloses (from the
    machine description; empty when no circuit is a coil case).  Region labels
    use normalised centroid coordinates over the GIVEN circuit set, so the rule
    transfers across machines unchanged.
    """
    if level not in LADDER_LEVELS:
        raise ValueError(f"unknown ladder level {level!r} (want {LADDER_LEVELS})")
    case_of = dict(case_of or {})
    radius = np.asarray(centroid_r, dtype=np.float64)
    height = np.asarray(centroid_z, dtype=np.float64)
    span = max(float(radius.max() - radius.min()), 1e-9)
    r_norm = (radius - radius.min()) / span
    z_frac = np.abs(height) / max(float(np.abs(height).max()), 1e-9)

    labels: list[str] = []
    for index, circuit in enumerate(np.asarray(circuits, dtype=np.int64)):
        coil = case_of.get(int(circuit))
        if level == "global":
            labels.append("all")
        elif coil is not None:
            if level == "vessel-case":
                labels.append("case")
            elif level == "regions-casepairs":
                labels.append(f"case:{coil[:-1]}")  # up/down cases share a build
            else:
                labels.append(f"case:{coil}")
        elif level == "vessel-case":
            labels.append("vessel")
        elif r_norm[index] < REGION_INBOARD_RFRAC:
            labels.append("vessel:inboard")
        elif r_norm[index] > REGION_OUTBOARD_RFRAC:
            labels.append("vessel:outboard")
        elif z_frac[index] > REGION_ENDS_ZFRAC:
            labels.append("vessel:ends")
        else:
            labels.append("vessel:mid")
    return labels


@dataclass
class ResistanceCalibration:
    """A fitted resistance model: one multiplier per group at one ladder level."""

    level: str
    group_multipliers: dict[str, float]
    provenance: dict

    def per_circuit(
        self,
        circuits: np.ndarray,
        centroid_r: np.ndarray,
        centroid_z: np.ndarray,
        *,
        case_of: dict[int, str] | None = None,
    ) -> np.ndarray:
        """Per-circuit multipliers for an arbitrary passive circuit set.

        Fails loud on an unknown group -- a silent 1.0 would un-calibrate a
        region without anyone noticing.  Groups the calibration carries but the
        set does not use are simply unused.
        """
        labels = resistance_group_labels(
            circuits, centroid_r, centroid_z, self.level, case_of=case_of
        )
        missing = sorted({lb for lb in labels if lb not in self.group_multipliers})
        if missing:
            raise KeyError(
                f"calibration level {self.level!r} has no multiplier for "
                f"group(s) {missing}"
            )
        return np.array([self.group_multipliers[label] for label in labels])


def save_calibration(path: Path | str, calibration: ResistanceCalibration) -> None:
    """Write a resistance calibration artifact."""
    Path(path).write_text(
        json.dumps(
            {
                "kind": "vacuum-passive-resistance-calibration",
                "level": calibration.level,
                "group_multipliers": calibration.group_multipliers,
                "region_rule": {
                    "inboard_rfrac": REGION_INBOARD_RFRAC,
                    "outboard_rfrac": REGION_OUTBOARD_RFRAC,
                    "ends_zfrac": REGION_ENDS_ZFRAC,
                },
                "provenance": calibration.provenance,
            },
            indent=2,
        )
    )


def load_calibration(path: Path | str) -> ResistanceCalibration:
    """Read a resistance calibration artifact."""
    stored = json.loads(Path(path).read_text())
    if stored.get("kind") != "vacuum-passive-resistance-calibration":
        raise ValueError(f"{path}: not a resistance calibration artifact")
    return ResistanceCalibration(
        level=str(stored["level"]),
        group_multipliers={
            key: float(value) for key, value in stored["group_multipliers"].items()
        },
        provenance=dict(stored.get("provenance", {})),
    )


# ---------------------------------------------------------------------------
# fit data + per-shot loss terms
# ---------------------------------------------------------------------------
@dataclass
class VacuumShotData:
    """One shot's prepared coil-only arrays (the leading coil-only run only).

    ``psi_circuit`` ``(n_t, n_circuits)`` is the drive flux each passive circuit
    links [Wb]; ``residual`` ``(n_t, n_sensors)`` the measured magnetics minus the
    non-held-back static drive prediction; ``measured`` ``(n_t, n_measured)`` the
    held-back circuit currents [A], NaN where absent.  ``i_drive`` ``(n_t,
    n_channels)`` in the system's channel order is required by the structure
    discovery, whose drive-column edits and galvanic voltage terms are functions
    of the raw drives rather than of the precomputed ``psi_circuit``.
    """

    shot: int
    campaign: str
    stratum: str
    interval: float
    psi_circuit: np.ndarray
    residual: np.ndarray
    sigma: np.ndarray
    measured: np.ndarray
    i_drive: np.ndarray | None = None

    @property
    def n_samples(self) -> int:
        """Return the number of raw samples in the interval."""
        return int(self.psi_circuit.shape[0])


@dataclass
class ModeMaps:
    """Resistance-dependent maps of one campaign, shared by all its shots."""

    tau: np.ndarray
    v: np.ndarray
    a_sensor_modes: np.ndarray
    measured_v: np.ndarray


def campaign_mode_maps(
    system: PassiveCircuitSystem, multipliers: np.ndarray
) -> ModeMaps:
    """Solve the generalised eigenproblem for one candidate resistance model."""
    tau, vectors = system.mode_system(np.asarray(multipliers, dtype=np.float64))
    rows = measured_rows(system)
    return ModeMaps(
        tau=tau,
        v=vectors,
        a_sensor_modes=system.a_circuit @ vectors,
        measured_v=(vectors[rows] if rows.size else np.zeros((0, vectors.shape[0]))),
    )


def measured_rows(system: PassiveCircuitSystem) -> np.ndarray:
    """Circuit rows of the held-back measured channels, sorted by channel name."""
    return np.array(
        [
            system.measured_channel_row[channel]
            for channel in sorted(system.measured_channel_row)
        ],
        dtype=np.int64,
    )


def _whitened_terms(residual: np.ndarray, sigma: np.ndarray) -> tuple[float, int]:
    """Sum of whitened squares with a per-channel offset nuisance removed."""
    with np.errstate(invalid="ignore"):
        residual = residual - np.nanmean(residual, axis=0, keepdims=True)
    white = residual / sigma
    finite = np.isfinite(white)
    return float(np.nansum(np.where(finite, white, 0.0) ** 2)), int(finite.sum())


def shot_loss_terms(
    data: VacuumShotData,
    maps: ModeMaps,
    sigma_sensor: np.ndarray,
    sigma_measured: np.ndarray,
) -> tuple[float, int, float, int]:
    """Whitened sum-of-squares terms of one shot under one resistance model.

    Magnetics: the eddy-signal residual ``residual - a_sensor . a(t)``, with each
    channel's mean removed over the interval (an offset nuisance, as the static
    vacuum audit fits) and whitened by the pooled channel scale.  Held-back
    circuits: measured minus predicted, mean removed (instrumental zero offset),
    whitened by the pooled measured scale.  Returns
    ``(ss_sensor, n_sensor, ss_measured, n_measured)``.
    """
    psi_mode = data.psi_circuit @ maps.v
    state = zoh_mode_response(maps.tau, data.interval, psi_mode)
    ss_sensor, n_sensor = _whitened_terms(
        data.residual - state @ maps.a_sensor_modes.T, sigma_sensor
    )
    ss_measured, n_measured = 0.0, 0
    if data.measured.size and maps.measured_v.size:
        ss_measured, n_measured = _whitened_terms(
            data.measured - state @ maps.measured_v.T, sigma_measured
        )
    return ss_sensor, n_sensor, ss_measured, n_measured


def pooled_loss(
    theta: np.ndarray,
    group_index: dict[str, np.ndarray],
    systems: dict[str, PassiveCircuitSystem],
    shots: list[VacuumShotData],
    sigma_sensor: dict[str, np.ndarray],
    sigma_measured: dict[str, np.ndarray],
    *,
    measured_weight: float = 1.0,
) -> dict[str, float]:
    """Combined mean whitened square over a shot pool for one parameter vector.

    ``theta`` holds the per-GROUP multipliers; ``group_index[campaign]`` maps it
    onto that campaign's circuits.  Returns the combined loss and its magnetics /
    held-back components, all means so pools of different size compare directly.
    """
    maps = {
        key: campaign_mode_maps(systems[key], theta[group_index[key]])
        for key in systems
    }
    total_sensor = total_measured = 0.0
    n_sensor = n_measured = 0
    for shot in shots:
        terms = shot_loss_terms(
            shot,
            maps[shot.campaign],
            sigma_sensor[shot.campaign],
            sigma_measured[shot.campaign],
        )
        total_sensor += terms[0]
        n_sensor += terms[1]
        total_measured += terms[2]
        n_measured += terms[3]
    sensor = total_sensor / max(n_sensor, 1)
    measured = total_measured / max(n_measured, 1)
    return {
        "combined": sensor + measured_weight * measured,
        "sensor": sensor,
        "measured": measured,
        "n_sensor": float(n_sensor),
        "n_measured": float(n_measured),
    }


# ---------------------------------------------------------------------------
# structure discovery: wiring, constraint reductions, adjacency couplings
# ---------------------------------------------------------------------------
def case_parent_channels(case_channel: str, channels: list[str]) -> list[str]:
    """Winding channels galvanically tied to a measured case channel.

    Label-convention rule: a case channel ``<family><position>_case_current``
    matches every ``*_coil_current`` whose coil label starts with the family and
    ends with the position -- both windings of a doubly-wound coil, or exactly one
    for a singly-wound one.  The measured sibling identity
    ``plain = sum(coils) + case`` is what grounds the mapping.
    """
    base = case_channel.split("_")[0]
    family, position = base[:-1], base[-1]
    return sorted(
        channel
        for channel in channels
        if channel.endswith("_coil_current")
        and (label := channel.split("_")[0]).startswith(family)
        and label.endswith(position)
    )


def updown_pair_channels(channels: list[str]) -> list[tuple[str, str]]:
    """Up/down partner pairs among the drive channels (label-convention rule).

    A channel pairs with the one whose coil label differs only in the final
    ``u``/``l`` position -- the mirror pairs a coil audit typically cannot
    separate.  Returned as ``(upper, lower)``, sorted by label.
    """
    label_of = {channel.split("_")[0]: channel for channel in channels}
    return [
        (label_of[label], label_of[label[:-1] + "l"])
        for label in sorted(label_of)
        if label.endswith("u") and label[:-1] + "l" in label_of
    ]


def neighbour_edges(
    centroid_r: np.ndarray,
    centroid_z: np.ndarray,
    section_scale: np.ndarray,
    *,
    factor: float = 1.5,
    exclude_rows: set[int] | frozenset[int] = frozenset(),
) -> list[tuple[int, int]]:
    """Nearest-neighbour candidate galvanic couplings of a passive set.

    An edge is a candidate when the centroid distance is within ``factor`` times
    the pair's mean section scale -- touching or nearly touching cross-sections
    under each pair's OWN size, a dimensionless rule that transfers across
    machines.  ``exclude_rows`` keeps circuits with a dedicated wiring hypothesis
    (the instrumented cases) out of the graph.
    """
    radius = np.asarray(centroid_r, dtype=np.float64)
    height = np.asarray(centroid_z, dtype=np.float64)
    scale = np.asarray(section_scale, dtype=np.float64)
    edges: list[tuple[int, int]] = []
    for i in range(radius.size):
        if i in exclude_rows:
            continue
        for j in range(i + 1, radius.size):
            if j in exclude_rows:
                continue
            distance = np.hypot(radius[i] - radius[j], height[i] - height[j])
            if distance <= factor * 0.5 * (scale[i] + scale[j]):
                edges.append((i, j))
    return edges


def series_reduction(n_circuits: int, pairs: list[tuple[int, int, int]]) -> np.ndarray:
    """Constraint-reduction map for wired circuit pairs.

    Each ``(i, j, sign)`` imposes ``I_j = sign I_i``, so one merged state carries
    both circuits.  With ``I = C q`` the reduced system is
    ``C'LC q' + C'RC q = C'u``, and the congruence reproduces the classical series
    result ``L_eff = L_ii + L_jj +- 2M``, ``R_eff = R_i + R_j``.  Pairs must be
    disjoint.
    """
    used: set[int] = set()
    for i, j, _sign in pairs:
        if i == j or i in used or j in used:
            raise ValueError(f"pairs must be disjoint, got {pairs}")
        used.update((i, j))
    partner = {i: (j, sign) for i, j, sign in pairs}
    keep = [
        row for row in range(n_circuits) if row not in {j for _i, j, _sign in pairs}
    ]
    reduction = np.zeros((n_circuits, len(keep)))
    for column, row in enumerate(keep):
        reduction[row, column] = 1.0
        if row in partner:
            merged, sign = partner[row]
            reduction[merged, column] = float(sign)
    return reduction


@dataclass
class PassiveStructure:
    """Discovered conductor topology plus its fitted bounded parameters.

    ``case_series_pairs``: held-back channels wired as one circuit (sign +1
    series, -1 anti-series).  ``case_wiring``: per case channel the galvanic
    drive ``{parents, g_v, r_w}``.  ``pair_drive_gains``: per coil pair
    ``{channels, common, differential}`` corrections on the drive columns.
    ``adjacency``: per campaign the accepted neighbour couplings
    ``{i, j, r_couple}`` keyed by circuit id.  ``r_level`` /
    ``r_group_multipliers``: the jointly refit resistance calibration.
    """

    case_series_pairs: list[dict]
    case_wiring: dict[str, dict]
    pair_drive_gains: list[dict]
    adjacency: dict[str, list[dict]]
    neighbour_rule: dict
    r_level: str
    r_group_multipliers: dict[str, float]
    provenance: dict


def save_structure(path: Path | str, structure: PassiveStructure) -> None:
    """Write a passive-structure artifact."""
    Path(path).write_text(
        json.dumps(
            {
                "kind": "vacuum-passive-structure-calibration",
                "case_series_pairs": structure.case_series_pairs,
                "case_wiring": structure.case_wiring,
                "pair_drive_gains": structure.pair_drive_gains,
                "adjacency": structure.adjacency,
                "neighbour_rule": structure.neighbour_rule,
                "r_level": structure.r_level,
                "r_group_multipliers": structure.r_group_multipliers,
                "provenance": structure.provenance,
            },
            indent=2,
        )
    )


def load_structure(path: Path | str) -> PassiveStructure:
    """Read a passive-structure artifact."""
    stored = json.loads(Path(path).read_text())
    if stored.get("kind") != "vacuum-passive-structure-calibration":
        raise ValueError(f"{path}: not a passive-structure artifact")
    return PassiveStructure(
        case_series_pairs=list(stored["case_series_pairs"]),
        case_wiring={key: dict(value) for key, value in stored["case_wiring"].items()},
        pair_drive_gains=list(stored["pair_drive_gains"]),
        adjacency={key: list(value) for key, value in stored["adjacency"].items()},
        neighbour_rule=dict(stored.get("neighbour_rule", {})),
        r_level=str(stored["r_level"]),
        r_group_multipliers={
            key: float(value) for key, value in stored["r_group_multipliers"].items()
        },
        provenance=dict(stored.get("provenance", {})),
    )


@dataclass
class StructuredModeMaps:
    """Maps of one campaign under one structure hypothesis set.

    ``v_physical`` ``(n_circuits, n_modes)`` maps mode amplitudes to PHYSICAL
    circuit currents with the constraint reduction folded in, so sensors,
    held-back targets and downstream consumers never see reduced coordinates.
    Mode drives assemble from the raw drive currents:
    ``psi_mode = i_drive @ drive_flux.T``, ``volt_mode = i_drive @ drive_volt.T``.
    """

    tau: np.ndarray
    v_physical: np.ndarray
    a_sensor_modes: np.ndarray
    measured_map: np.ndarray
    drive_flux: np.ndarray
    drive_volt: np.ndarray | None


@dataclass
class StructureHypothesis:
    """One campaign's frozen hypothesis set plus its geometry-only arrays.

    Everything independent of the fitted parameters lives here, so a candidate
    costs one cheap eigensolve: the reduction map, the adjacency edge rows, the
    wiring rows (case row, parent drive columns, parent winding-flux rows of the
    drive linkage), the coil-pair drive columns, and the multiplier group index.
    """

    system: PassiveCircuitSystem
    reduction: np.ndarray
    group_index: np.ndarray
    edges: list[tuple[int, int]]
    wiring_rows: np.ndarray
    wiring_lam: np.ndarray
    wiring_select: np.ndarray
    pair_columns: list[tuple[int, int]]


def build_structure_hypothesis(
    system: PassiveCircuitSystem,
    group_index: np.ndarray,
    *,
    case_series: list[tuple[str, str, int]] | None = None,
    wiring_cases: list[str] | None = None,
    drive_linkage: tuple[list[str], np.ndarray] | None = None,
    pair_channels: list[tuple[str, str]] | None = None,
    edges: list[tuple[int, int]] | None = None,
) -> StructureHypothesis:
    """Freeze one hypothesis set into parameter-independent arrays.

    ``case_series``: ``(channel_i, channel_j, sign)`` wired pairs.
    ``wiring_cases``: held-back channels given the galvanic drive (parameter
    slots in this order); requires ``drive_linkage`` -- the ``(channels, lam)``
    pair from :func:`~nova.circuit.linkage.drive_linkage`.  ``pair_channels``:
    coil pairs given common/differential drive gains.  ``edges``: adjacency
    candidate row pairs, parameter slots in list order.
    """
    n_circuits = system.n_circuits
    row_pairs = [
        (
            system.measured_channel_row[channel_i],
            system.measured_channel_row[channel_j],
            int(sign),
        )
        for channel_i, channel_j, sign in case_series or []
    ]
    reduction = (
        series_reduction(n_circuits, row_pairs) if row_pairs else np.eye(n_circuits)
    )

    wiring_cases = list(wiring_cases or [])
    n_channels = len(system.channels)
    wiring_rows = np.array(
        [system.measured_channel_row[channel] for channel in wiring_cases],
        dtype=np.int64,
    )
    wiring_lam = np.zeros((len(wiring_cases), n_channels))
    wiring_select = np.zeros((len(wiring_cases), n_channels))
    column_of = {channel: i for i, channel in enumerate(system.channels)}
    if wiring_cases:
        if drive_linkage is None:
            raise ValueError("wiring_cases needs the drive_linkage (channels, lam)")
        lam_channels, lam = drive_linkage
        lam_row = {channel: i for i, channel in enumerate(lam_channels)}
        for slot, case_channel in enumerate(wiring_cases):
            parents = case_parent_channels(case_channel, system.channels)
            if not parents:
                raise ValueError(f"no parent winding channels for {case_channel}")
            for parent in parents:
                wiring_select[slot, column_of[parent]] = 1.0
                # the winding's flux from EVERY measured drive, its own self term
                # included, restricted to the system's drive columns
                for channel, column in column_of.items():
                    wiring_lam[slot, column] += lam[lam_row[parent], lam_row[channel]]

    return StructureHypothesis(
        system=system,
        reduction=reduction,
        group_index=np.asarray(group_index, dtype=np.int64),
        edges=list(edges or []),
        wiring_rows=wiring_rows,
        wiring_lam=wiring_lam,
        wiring_select=wiring_select,
        pair_columns=[
            (column_of[upper], column_of[lower]) for upper, lower in pair_channels or []
        ],
    )


def structured_mode_maps(
    hypothesis: StructureHypothesis,
    multipliers: np.ndarray,
    *,
    edge_r: np.ndarray | None = None,
    g_v: np.ndarray | None = None,
    r_w: np.ndarray | None = None,
    pair_gains: np.ndarray | None = None,
) -> StructuredModeMaps:
    """Solve one campaign's structured eigenproblem for one parameter set.

    ``multipliers`` scale the diagonal ring resistances; ``edge_r`` (per
    ``hypothesis.edges``) adds the positive-semidefinite adjacency stamps;
    ``g_v`` / ``r_w`` (per ``hypothesis.wiring_rows``) apply the galvanic case
    wiring; ``pair_gains`` ``(n_pairs, 2)`` of common and differential terms
    corrects the pair drive columns.
    """
    from scipy.linalg import eigh

    system = hypothesis.system
    n_circuits = system.n_circuits
    r_physical = np.diag(system.r_diag * np.asarray(multipliers, dtype=np.float64))
    if edge_r is not None and len(hypothesis.edges):
        for (i, j), rho in zip(hypothesis.edges, np.asarray(edge_r), strict=True):
            r_physical[i, i] += rho
            r_physical[j, j] += rho
            r_physical[i, j] -= rho
            r_physical[j, i] -= rho
    reduction = hypothesis.reduction
    rate, v_reduced = eigh(
        reduction.T @ r_physical @ reduction,
        reduction.T @ system.lmat @ reduction,
    )
    tau = 1.0 / np.clip(rate, 1e-12, None)
    v_physical = reduction @ v_reduced

    m_effective = system.m_channel
    volt_columns = None
    if g_v is not None and hypothesis.wiring_rows.size:
        m_effective = m_effective.copy()
        # +g_v dLambda_w/dt on the case row is -g_v Lambda_w in the linked-flux
        # columns, because the mode drive is -dPsi/dt
        m_effective[hypothesis.wiring_rows] -= (
            np.asarray(g_v)[:, np.newaxis] * hypothesis.wiring_lam
        )
    if r_w is not None and hypothesis.wiring_rows.size:
        volt_columns = np.zeros((n_circuits, len(system.channels)))
        volt_columns[hypothesis.wiring_rows] = (
            np.asarray(r_w)[:, np.newaxis] * hypothesis.wiring_select
        )
    if pair_gains is not None and hypothesis.pair_columns:
        if m_effective is system.m_channel:
            m_effective = m_effective.copy()
        for (upper, lower), (common_gain, differential_gain) in zip(
            hypothesis.pair_columns,
            np.asarray(pair_gains).reshape(-1, 2),
            strict=True,
        ):
            column_u = system.m_channel[:, upper]
            column_l = system.m_channel[:, lower]
            common = 0.5 * (column_u + column_l)
            differential = 0.5 * (column_u - column_l)
            m_effective[:, upper] += (
                common_gain * common + differential_gain * differential
            )
            m_effective[:, lower] += (
                common_gain * common - differential_gain * differential
            )

    rows = measured_rows(system)
    return StructuredModeMaps(
        tau=tau,
        v_physical=v_physical,
        a_sensor_modes=system.a_circuit @ v_physical,
        measured_map=(
            v_physical[rows] if rows.size else np.zeros((0, v_physical.shape[1]))
        ),
        drive_flux=v_physical.T @ m_effective,
        drive_volt=None if volt_columns is None else v_physical.T @ volt_columns,
    )


def structured_shot_loss(
    data: VacuumShotData,
    maps: StructuredModeMaps,
    sigma_sensor: np.ndarray,
    sigma_measured: np.ndarray,
) -> tuple[float, int, float, int]:
    """Whitened loss terms of one shot under one structured model.

    Same contract as :func:`shot_loss_terms` -- offset-nuisance magnetics plus
    held-back targets -- with the drives assembled from the RAW drive currents so
    the structure's drive-column edits and galvanic voltage terms take effect.
    """
    if data.i_drive is None:
        raise ValueError("the structured loss needs VacuumShotData.i_drive")
    psi_mode = data.i_drive @ maps.drive_flux.T
    volt_mode = None if maps.drive_volt is None else data.i_drive @ maps.drive_volt.T
    state = zoh_mode_response(maps.tau, data.interval, psi_mode, voltage_mode=volt_mode)
    ss_sensor, n_sensor = _whitened_terms(
        data.residual - state @ maps.a_sensor_modes.T, sigma_sensor
    )
    ss_measured, n_measured = 0.0, 0
    if data.measured.size and maps.measured_map.size:
        ss_measured, n_measured = _whitened_terms(
            data.measured - state @ maps.measured_map.T, sigma_measured
        )
    return ss_sensor, n_sensor, ss_measured, n_measured


def structure_hypothesis_parts(
    system: PassiveCircuitSystem,
    structure: PassiveStructure,
    *,
    campaign: str | None = None,
    drive_linkage: tuple[list[str], np.ndarray] | None = None,
    case_of: dict[int, str] | None = None,
) -> tuple[StructureHypothesis, dict]:
    """Instantiate a saved structure artifact on one circuit system.

    Elements that do not apply are dropped automatically: a system built with the
    measured circuits kept as DRIVES has no held-back rows, so the case wiring and
    series pairs apply only to the holdback form, while the adjacency couplings,
    pair drive gains and resistance multipliers apply to both.  Returns the frozen
    hypothesis plus the fitted parameter arrays ready for
    :func:`structured_mode_maps`.
    """
    case_series = [
        (pair["channels"][0], pair["channels"][1], int(pair["sign"]))
        for pair in structure.case_series_pairs
        if all(channel in system.measured_channel_row for channel in pair["channels"])
    ]
    wiring_cases = sorted(
        channel
        for channel in structure.case_wiring
        if channel in system.measured_channel_row
    )
    if wiring_cases and drive_linkage is None:
        raise ValueError(
            "the structure carries case wiring for this system -- pass drive_linkage"
        )
    pair_channels = [
        (pair["channels"][0], pair["channels"][1])
        for pair in structure.pair_drive_gains
        if all(channel in system.channels for channel in pair["channels"])
    ]
    row_of = {int(circuit): row for row, circuit in enumerate(system.circuits)}
    edges: list[tuple[int, int]] = []
    edge_r: list[float] = []
    for record in structure.adjacency.get(campaign or "", []):
        if int(record["i"]) in row_of and int(record["j"]) in row_of:
            edges.append((row_of[int(record["i"])], row_of[int(record["j"])]))
            edge_r.append(float(record["r_couple"]))

    calibration = ResistanceCalibration(
        level=structure.r_level,
        group_multipliers=structure.r_group_multipliers,
        provenance={},
    )
    hypothesis = build_structure_hypothesis(
        system,
        np.zeros(system.n_circuits, dtype=np.int64),
        case_series=case_series,
        wiring_cases=wiring_cases,
        drive_linkage=drive_linkage if wiring_cases else None,
        pair_channels=pair_channels,
        edges=edges,
    )
    parts: dict = {
        "multipliers": calibration.per_circuit(
            system.circuits,
            system.centroid_r,
            system.centroid_z,
            case_of=case_of,
        )
    }
    if wiring_cases:
        parts["g_v"] = np.array(
            [structure.case_wiring[channel]["g_v"] for channel in wiring_cases]
        )
        parts["r_w"] = np.array(
            [structure.case_wiring[channel]["r_w"] for channel in wiring_cases]
        )
    if pair_channels:
        parts["pair_gains"] = np.array(
            [
                (pair["common"], pair["differential"])
                for pair in structure.pair_drive_gains
                if all(channel in system.channels for channel in pair["channels"])
            ]
        ).reshape(-1, 2)
    if edges:
        parts["edge_r"] = np.array(edge_r)
    return hypothesis, parts


def structured_reduced_basis(
    system: PassiveCircuitSystem,
    structure: PassiveStructure,
    *,
    sensor_scale: np.ndarray,
    n_modes: int,
    cell_index: np.ndarray,
    campaign: str | None = None,
    drive_linkage: tuple[list[str], np.ndarray] | None = None,
    case_of: dict[int, str] | None = None,
) -> PassiveEigenbasis:
    """Mode-reduced eigenbasis of a structured system.

    The same history-relevance selection as
    :func:`~nova.circuit.passive.reduce_passive_system`, computed over the
    STRUCTURED eigenmodes; the drive couplings carry the wiring flux edits and
    pair gains, and ``volt_channel`` carries the galvanic terms the raw-cadence
    trajectory needs.
    """
    from nova.circuit.passive import select_modes

    hypothesis, parts = structure_hypothesis_parts(
        system,
        structure,
        campaign=campaign,
        drive_linkage=drive_linkage,
        case_of=case_of,
    )
    maps = structured_mode_maps(hypothesis, **parts)
    keep = select_modes(maps.tau, maps.a_sensor_modes, sensor_scale, n_modes)
    g_modes = system.g_grid @ maps.v_physical[:, keep]
    volt = None
    if maps.drive_volt is not None and np.any(maps.drive_volt):
        volt = maps.drive_volt[keep]
    return PassiveEigenbasis(
        tau=maps.tau[keep],
        v=maps.v_physical[:, keep],
        a_sensor=maps.a_sensor_modes[:, keep],
        g_grid=g_modes,
        m_channel=maps.drive_flux[keep],
        m_cell=g_modes[np.asarray(cell_index)].T,
        resistivity=float(system.resistivity),
        volt_channel=volt,
    )


__all__ = [
    "LADDER_LEVELS",
    "MULTIPLIER_BOUNDS",
    "REGION_ENDS_ZFRAC",
    "REGION_INBOARD_RFRAC",
    "REGION_OUTBOARD_RFRAC",
    "ModeMaps",
    "PassiveStructure",
    "ResistanceCalibration",
    "StructureHypothesis",
    "StructuredModeMaps",
    "VacuumShotData",
    "build_structure_hypothesis",
    "campaign_mode_maps",
    "case_parent_channels",
    "load_calibration",
    "load_structure",
    "measured_rows",
    "neighbour_edges",
    "pooled_loss",
    "resistance_group_labels",
    "save_calibration",
    "save_structure",
    "series_reduction",
    "shot_loss_terms",
    "structure_hypothesis_parts",
    "structured_mode_maps",
    "structured_reduced_basis",
    "structured_shot_loss",
    "updown_pair_channels",
    "zoh_mode_response",
]
