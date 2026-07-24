"""Circuit-coupling generator: classify conductor circuits, merge drive channels.

Magnetostatic reconstruction couples sensors (and grids) to *driven current
sources*, but a facility's conductor model enumerates *circuits* — filament
groups that may redundantly discretise one physical coil (a fine interior
grid plus a coarse corner set, each normalised to the full coil current), may
be a coil's separately-supplied structural case sitting centimetres from the
winding it encloses, or may be unpowered structure. Building one coupling
column per circuit therefore double-counts redundantly-discretised coils, and
geometry alone cannot tell a coil from its co-located case.

This module owns that bookkeeping as a machine-agnostic mechanism. The
machine description arrives as data (:class:`CircuitTable`: coil centroids
with drive-channel preference orders, plus the authoritative circuit-id ↔
case correspondence); classification is verify-and-flag — a circuit is KNOWN
only when its weighted filament centroid matches a coil within the table's
radius AND its drive channel is actually present; a case circuit is
recognised BY ID, never by distance, and is driven only by its own measured
channel. Everything else is inferred passive, with the reason flagged rather
than a value fabricated.

:func:`couple_circuits` then turns a classification into a coupling plan:
one column per drive channel, redundant same-channel circuits merged by
averaging (the coil current is applied exactly once), case circuits always
in their own column, passive circuits listed for the eddy/structure block.
:meth:`CouplingPlan.weight_matrix` expresses the plan as per-source column
weights so any per-source coupling matrix from the biot Solve/Matrix tier
collapses to merged columns with a single matmul.
"""

from dataclasses import dataclass

import numpy as np

__all__ = [
    "CaseChannel",
    "CircuitClass",
    "CircuitTable",
    "CoilChannel",
    "CouplingColumn",
    "CouplingPlan",
    "classify_circuits",
    "couple_circuits",
]


@dataclass(frozen=True)
class CoilChannel:
    """One driven coil: geometry anchor + drive-channel preference order."""

    label: str
    centroid: tuple[float, float]
    channels: tuple[str, ...]

    def channel(self, available: frozenset[str]) -> str:
        """First preferred drive channel present in ``available`` ('' if none)."""
        for name in self.channels:
            if name in available:
                return name
        return ""


@dataclass(frozen=True)
class CaseChannel:
    """One coil-case (structural, separately-supplied) circuit.

    ``circuit`` is the authoritative machine-description circuit id;
    ``coil_label`` names the active coil whose centroid the case is
    geometrically confusable with. ``constrained_zero`` marks a case the
    machine description pins to zero current (no measurement exists) — such
    a circuit is never driven, even if a channel of the right name appears.
    """

    circuit: int
    coil_label: str
    channel: str | None
    constrained_zero: bool


@dataclass(frozen=True)
class CircuitTable:
    """Machine circuit description: coils, cases, and the match radius [m]."""

    coils: tuple[CoilChannel, ...]
    cases: tuple[CaseChannel, ...]
    match_radius: float = 0.08

    def nearest_coil(self, radius: float, height: float) -> tuple[str, float]:
        """Label and distance of the nearest coil centroid."""
        best_label, best_distance = "", np.inf
        for coil in self.coils:
            distance = float(
                np.hypot(radius - coil.centroid[0], height - coil.centroid[1])
            )
            if distance < best_distance:
                best_label, best_distance = coil.label, distance
        return best_label, best_distance

    def coil(self, label: str) -> CoilChannel:
        for entry in self.coils:
            if entry.label == label:
                return entry
        raise KeyError(
            f"unknown coil label {label!r}; known labels: "
            f"{sorted(coil.label for coil in self.coils)}"
        )

    def case(self, circuit: int) -> CaseChannel | None:
        for entry in self.cases:
            if entry.circuit == circuit:
                return entry
        return None


@dataclass(frozen=True)
class CircuitClass:
    """Classification of one circuit: known coil, known case, or inferred.

    ``flag`` records the reason a geometry-matched circuit was demoted to
    inferred (verify-and-flag — never a silently fabricated drive).
    """

    circuit: int
    centroid_radius: float
    centroid_height: float
    filament_count: int
    weight_sum: float
    role: str  # "known_pf" | "known_case" | "inferred_passive"
    coil_label: str  # "" when inferred
    channel: str  # "" when inferred
    flag: str  # "" when confident


_KNOWN_ROLES = ("known_pf", "known_case")


def classify_circuits(
    circuit: np.ndarray,
    radius: np.ndarray,
    height: np.ndarray,
    weight: np.ndarray,
    channels: "list[str] | tuple[str, ...]",
    table: CircuitTable,
) -> list[CircuitClass]:
    """Classify each circuit present in the per-filament ``circuit`` labels.

    ``radius`` / ``height`` / ``weight`` are per-filament coordinates [m] and
    current-share weights. A circuit is KNOWN active when its weighted
    centroid sits within ``table.match_radius`` of a coil AND one of that
    coil's drive channels is present in ``channels``. The nearest-centroid
    match is only the first pass: the table's circuit-id ↔ case
    correspondence then decides whether this specific circuit is the matched
    coil's case — driven by its own measured channel or, when that channel is
    absent or the case is constrained to zero, demoted to inferred passive.
    """
    available = frozenset(channels)
    circuit = np.asarray(circuit, dtype=int)
    radius = np.asarray(radius, dtype=np.float64)
    height = np.asarray(height, dtype=np.float64)
    weight = np.asarray(weight, dtype=np.float64)

    classes: list[CircuitClass] = []
    for label in sorted(np.unique(circuit).tolist()):
        select = circuit == label
        weights = weight[select]
        weight_sum = float(weights.sum())
        if weight_sum:
            centroid_radius = float((weights * radius[select]).sum() / weight_sum)
            centroid_height = float((weights * height[select]).sum() / weight_sum)
        else:
            centroid_radius = float(radius[select].mean())
            centroid_height = float(height[select].mean())

        coil_label, distance = table.nearest_coil(centroid_radius, centroid_height)
        role, matched_label, channel, flag = "inferred_passive", "", "", ""
        if distance <= table.match_radius:
            case = table.case(label)
            if case is not None and case.coil_label == coil_label:
                # this circuit IS the coil's dedicated case — never drive it
                # by the active coil's current, even though geometry alone
                # would confuse the two.
                if not case.constrained_zero and case.channel in available:
                    role = "known_case"
                    matched_label = f"{coil_label}_case"
                    channel = case.channel or ""
                elif case.constrained_zero:
                    flag = (
                        f"case circuit {label} constrained to zero by the "
                        "machine description (no measured channel) → inferred"
                    )
                else:
                    flag = (
                        f"case circuit {label} channel {case.channel!r} "
                        "absent from this campaign → inferred"
                    )
            else:
                channel = table.coil(coil_label).channel(available)
                if channel:
                    role, matched_label = "known_pf", coil_label
                else:
                    flag = (
                        f"coil {coil_label!r} matched by geometry "
                        f"(d={distance * 1e3:.0f}mm) but no drive channel "
                        "present → inferred"
                    )
        classes.append(
            CircuitClass(
                circuit=label,
                centroid_radius=centroid_radius,
                centroid_height=centroid_height,
                filament_count=int(select.sum()),
                weight_sum=weight_sum,
                role=role,
                coil_label=matched_label,
                channel=channel,
                flag=flag,
            )
        )
    return classes


@dataclass(frozen=True)
class CouplingColumn:
    """One driven coupling column: a drive channel and its member circuits."""

    channel: str
    circuits: tuple[int, ...]
    kind: str  # "coil" | "case"


@dataclass(frozen=True)
class CouplingPlan:
    """Column grouping for a classified circuit set.

    ``columns`` hold the driven sources (one per drive channel, member
    circuits merged); ``passive`` lists the inferred circuits for the
    eddy/structure block.
    """

    columns: tuple[CouplingColumn, ...]
    passive: tuple[int, ...]
    classes: tuple[CircuitClass, ...]

    def weight_matrix(
        self,
        circuit: np.ndarray,
        weight: np.ndarray,
        include_passive: bool = False,
    ) -> np.ndarray:
        """Per-source column weights: ``couplings @ weight_matrix`` merges.

        Each filament of a member circuit carries ``weight / n_members`` in
        its column, so redundant same-channel discretisations average and the
        drive current is applied exactly once. Passive columns (appended when
        ``include_passive``) carry the raw filament weight.
        """
        circuit = np.asarray(circuit, dtype=int)
        weight = np.asarray(weight, dtype=np.float64)
        column_count = len(self.columns) + (len(self.passive) if include_passive else 0)
        weights = np.zeros((circuit.size, column_count), dtype=np.float64)
        for index, column in enumerate(self.columns):
            share = 1.0 / len(column.circuits)
            for member in column.circuits:
                select = circuit == member
                weights[select, index] = weight[select] * share
        if include_passive:
            for index, member in enumerate(self.passive, start=len(self.columns)):
                select = circuit == member
                weights[select, index] = weight[select]
        return weights


def couple_circuits(classes: "list[CircuitClass]") -> CouplingPlan:
    """Group classified circuits into driven columns plus a passive block.

    Known circuits sharing a drive channel merge into one column (redundant
    discretisations of the same coil); a case circuit's dedicated channel is
    never shared with its coil, so cases always land in their own column.
    Columns are ordered by channel name; member circuits ascend.
    """
    by_channel: dict[str, list[CircuitClass]] = {}
    passive: list[int] = []
    for entry in classes:
        if entry.role in _KNOWN_ROLES:
            by_channel.setdefault(entry.channel, []).append(entry)
        else:
            passive.append(entry.circuit)
    columns = tuple(
        CouplingColumn(
            channel=channel,
            circuits=tuple(sorted(entry.circuit for entry in members)),
            kind="case" if members[0].role == "known_case" else "coil",
        )
        for channel, members in sorted(by_channel.items())
    )
    return CouplingPlan(
        columns=columns, passive=tuple(sorted(passive)), classes=tuple(classes)
    )
