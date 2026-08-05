"""Calibrate the MAST passive resistance against the dedicated decay experiments.

Four stages, each writing what the next one reads, because the first is expensive
and the geometry it produces never changes while a fit is being iterated.

``inductance`` builds the flux-linkage matrix of the passive circuits on the
authored configuration and measures its own convergence.  ``transients`` selects
the decay experiments, applies every screen, and caches the windowed waveforms.
``fit`` fits the resistivity classes with the linkage in the loop, profiles each
class for identifiability, and scores the models on held-out shots and a held-out
excitation family.  ``figures`` draws what the fit established.

The held-out excitation family is declared here, before any fit runs, and it is
the outboard coil nearest the two probes that carry the description's residual
excess: predicting an unseen coil's case decay is the hardest challenge the
archive offers, and it is the one whose answer matters to the residual.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from nova.catalog.mast_geometry import MachineGeometryRegistry
from nova.imas.mast_error_field_screen import (
    ChannelCoupling,
    ErrorFieldScreen,
    read_error_field_drive,
)
from nova.imas.mast_fitted_parameters import MIS_SCALED_SHOTS, RADIAL_PROBE_FAMILY
from nova.imas.mast_passive_decay_modes import (
    FASTEST_RESOLVABLE_TIME,
    MAXIMUM_DRIVE_SHARE,
    RESOLVED_MODE_COUNT,
    SLOWEST_RESOLVABLE_TIME,
    DecayTransient,
    channel_rows,
    class_names,
    fit_resistivity,
    held_out_score,
    leave_one_out,
    mode_count_sensitivity,
    mode_set,
    profile_class,
    read_transient,
    reconstruct,
    resistivity_class,
    visible_modes,
)
from nova.imas.mast_passive_inductance import (
    Linkage,
    linkage_convergence,
    linkage_matrix,
    linkage_provenance,
    nominal_resistance,
    passive_turns,
    probe_coupling,
)
from nova.imas.mast_seed_parameters import STAINLESS_STEEL, passive_material
from nova.imas.mast_vacuum_cohort import (
    SHOT_STORE,
    probe_channels,
    read_shot_waveforms,
)
from nova.imas.mast_vacuum_response import ResponseModel

REPRESENTATIVE_SHOT = 11766
"""Registry selection whose configuration the calibration is authored against."""

DECAY_EXPERIMENT_CLASSES = (
    "sustained_single_coil",
    "sustained_symmetric_pair",
    "pulsed_excitation",
)
"""The experiment classes whose switch-off leaves a readable free decay.

A sustained pulse holds long enough that the passive currents reach the drive
before it stops, so what follows the switch-off is a decay from a known state
rather than a mixture of rise and fall.  A pulsed shot is admitted because its
switch-off is a decay too, and its shorter hold excites the fast circuits more
strongly.  Coil-group and toroidal-field-only shots are left out: the first drives
everything at once and separates nothing, and the second produces no poloidal
transient to decay.
"""

HELD_OUT_COIL = "p5"
"""The coil whose experiments are withheld from every fit.

Its case is the outboard case nearest the two probes carrying the description's
unexplained residual, so predicting its decay from a model that never saw it is
both the hardest test available and the one whose outcome bears on the residual.
Declared before fitting and never revisited.
"""


def load_screen(path: Path) -> ErrorFieldScreen:
    """Rebuild the committed error-field screen from its record.

    The screen is measured once on isolated shots and read here rather than
    re-derived, so the thresholds a decay shot is judged against are the same
    numbers every other fit in the ladder was judged against.
    """

    payload = json.loads(path.read_text())["screen"]
    return ErrorFieldScreen(
        couplings=tuple(
            ChannelCoupling(
                channel=row["channel"],
                driver=row["driver"],
                shot_count=int(row["shot_count"]),
                response=float(row["response"]),
                scatter=float(row["scatter"]),
                noise_floor=float(row["noise_floor"]),
                neighbour_response=float(row["neighbour_response"]),
            )
            for row in payload["couplings"]
        )
    )


def build_geometry():
    """Return the authored configuration, its probe model and its passive circuits."""

    registry = MachineGeometryRegistry.default()
    selection = registry.select(REPRESENTATIVE_SHOT)
    configuration = selection.configuration
    geometry = configuration.geometry
    probes = geometry["magnetics"]["poloidal_probes"]
    model = ResponseModel.build(
        geometry,
        probes,
        probe_channels(probes),
        radial_families=frozenset({RADIAL_PROBE_FAMILY}),
    )
    return (configuration, geometry, model, passive_turns(geometry))


def inductance(arguments: argparse.Namespace) -> None:
    """Build and cache the passive circuits' flux linkage."""

    configuration, _, model, turns = build_geometry()
    linkage = linkage_matrix(turns)
    provenance = linkage_provenance(
        turns,
        linkage,
        physical_digest=configuration.physical_digest,
        shot=REPRESENTATIVE_SHOT,
    )
    coupling = probe_coupling(turns, model.targets)
    resistance = nominal_resistance(turns)
    print(
        f"linkage over {len(turns)} circuits from "
        f"{provenance['quadrature_points']} quadrature points; "
        f"reciprocity residual {linkage.reciprocity_residual:.2e}"
    )
    modes = mode_set(linkage, resistance, coupling)
    band = modes.tau[(modes.tau >= 1.0e-3) & (modes.tau <= 0.5)]
    print(
        f"nominal resistance gives {modes.tau.max() * 1e3:.1f} ms slowest mode and "
        f"{band.size} modes between 1 ms and 500 ms"
    )
    if arguments.convergence:
        report = linkage_convergence(turns)
        provenance["convergence"] = report
        print(
            "convergence: self terms move "
            f"{report['self_term_shift_median'] * 100:.2f}% median / "
            f"{report['self_term_shift_max'] * 100:.2f}% worst, eigenvalues "
            f"{report['eigenvalue_shift_max'] * 100:.2f}% worst"
        )
    arguments.linkage.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        arguments.linkage,
        names=np.asarray(linkage.names),
        matrix=linkage.matrix,
        resistance=resistance,
        coupling=coupling,
        channels=np.asarray([target.channel for target in model.targets]),
        quadrature_points=linkage.quadrature_points,
        reciprocity_residual=np.asarray(linkage.reciprocity_residual),
    )
    arguments.report.parent.mkdir(parents=True, exist_ok=True)
    arguments.report.write_text(json.dumps(provenance, indent=1, sort_keys=True) + "\n")
    print(f"wrote {arguments.linkage} and {arguments.report}")


def load_linkage(path: Path) -> tuple[Linkage, np.ndarray, np.ndarray, list[str]]:
    """Return the cached linkage, nominal resistance, probe coupling and channels."""

    cached = np.load(path, allow_pickle=False)
    linkage = Linkage(
        names=tuple(str(name) for name in cached["names"]),
        matrix=cached["matrix"],
        reciprocity_residual=float(cached["reciprocity_residual"]),
        quadrature_points=cached["quadrature_points"],
    )
    return (
        linkage,
        cached["resistance"],
        cached["coupling"],
        [str(channel) for channel in cached["channels"]],
    )


def candidate_shots(cohort: dict[str, Any]) -> dict[int, str]:
    """Return the decay-experiment shots, each with the class that selected it."""

    by_class = cohort["cohort"]["by_class"]
    selected: dict[int, str] = {}
    for name in DECAY_EXPERIMENT_CLASSES:
        for shot in by_class.get(name, []):
            selected.setdefault(int(shot), name)
    return selected


def excitation_coil(families: Sequence[str]) -> str:
    """Return the coil set a shot's drive belongs to, or an empty string if mixed."""

    stems = {family.split("_")[0] for family in families}
    return stems.pop() if len(stems) == 1 else ""


def coil_field_response(model: ResponseModel) -> dict[str, dict[str, float]]:
    """Return the field each probe reads per unit drive of each coil family.

    The same geometry response the turn fits are built on, read here so the
    residual drive still falling through a decay window can be carried as a
    modelled column rather than screened against.  Its amplitude is fitted per
    shot, so no turn count enters: geometry supplies the spatial pattern and the
    archive supplies the waveform.
    """

    return {
        family: {
            target.channel: float(model.response[row, column])
            for row, target in enumerate(model.targets)
        }
        for column, family in enumerate(model.families)
    }


def transients(arguments: argparse.Namespace) -> None:
    """Select, screen and cache the decay experiments' windowed waveforms."""

    cohort = json.loads(arguments.cohort.read_text())
    binding = {
        "training": {int(shot) for shot in cohort["cohort"]["training"]},
        "held_out": {int(shot) for shot in cohort["cohort"]["held_out"]},
    }
    _, _, model, _ = build_geometry()
    screen = load_screen(arguments.screen)
    refused_amplitude = set(MIS_SCALED_SHOTS)
    drive_response = coil_field_response(model)

    rows: list[dict[str, Any]] = []
    kept: list[DecayTransient] = []
    for shot, experiment in sorted(candidate_shots(cohort).items()):
        record: dict[str, Any] = {"experiment": experiment, "shot": shot}
        if shot in refused_amplitude:
            record["refusal"] = "amplitude refused by the acquisition sweep"
            rows.append(record)
            continue
        split = next(
            (name for name, shots in binding.items() if shot in shots), "unsplit"
        )
        record["split"] = split
        try:
            drive = read_error_field_drive(shot, store=arguments.store)
        except Exception as error:  # noqa: BLE001 - the store's failures are data
            record["refusal"] = f"error-field drive unreadable: {type(error).__name__}"
            rows.append(record)
            continue
        record["error_field_disposition"] = (
            "unmeasured" if drive.unmeasured else "measured"
        )
        record["error_field_peak"] = drive.peak
        removed = screen.refused(drive)
        record["error_field_refused_channels"] = len(removed)
        if drive.unmeasured:
            record["refusal"] = "error-field channels unmeasured, so unvouched"
            rows.append(record)
            continue
        try:
            waveforms = read_shot_waveforms(shot, store=arguments.store)
            transient = read_transient(
                waveforms,
                model.targets,
                excitation_family="",
                refused_channels=removed,
                drive_response=drive_response,
            )
        except Exception as error:  # noqa: BLE001 - the store's failures are data
            record["refusal"] = f"{type(error).__name__}: {error}"
            rows.append(record)
            continue
        coil = excitation_coil(transient.driven_families)
        transient = replace(
            transient,
            excitation_family=coil or "+".join(transient.driven_families),
        )
        if transient.drive_share > arguments.maximum_drive_share:
            record["refusal"] = (
                f"the measured residual drive alone explains "
                f"{transient.drive_share:.3f} of the transient, so the window "
                "watches the coil switching off rather than the structure"
            )
            rows.append(record)
            continue
        record.update(transient.as_dict())
        record["excitation_coil"] = coil
        rows.append(record)
        kept.append(transient)

    if not kept:
        raise SystemExit("no decay experiment survived the screens")
    payload = {
        "decay_experiment_classes": list(DECAY_EXPERIMENT_CLASSES),
        "held_out_coil": HELD_OUT_COIL,
        "kept": len(kept),
        "mis_scaled_shots_refused": sorted(refused_amplitude),
        "shots": rows,
        "store": str(arguments.store),
    }
    arguments.transients.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        arguments.transients,
        **{
            f"{key}_{transient.shot}": value
            for transient in kept
            for key, value in (
                ("time", transient.time),
                ("signal", transient.signal),
                ("noise", transient.noise),
                ("channels", np.asarray(transient.channels)),
                (
                    "drive_patterns",
                    transient.drive_patterns
                    if transient.drive_patterns is not None
                    else np.zeros((len(transient.channels), 0)),
                ),
                (
                    "drive_waveforms",
                    transient.drive_waveforms
                    if transient.drive_waveforms is not None
                    else np.zeros((0, transient.time.size)),
                ),
                ("drive_names", np.asarray(transient.drive_names, dtype="<U32")),
                ("family", np.asarray([transient.excitation_family])),
                ("driven", np.asarray(transient.driven_families)),
                ("refused", np.asarray(transient.refused_channels)),
                (
                    "activity",
                    np.asarray([transient.peak_drive, transient.residual_drive]),
                ),
            )
        },
        shots=np.asarray([transient.shot for transient in kept]),
    )
    arguments.report.parent.mkdir(parents=True, exist_ok=True)
    arguments.report.write_text(json.dumps(payload, indent=1, sort_keys=True) + "\n")
    refusals = sum(1 for row in rows if "refusal" in row)
    print(
        f"{len(kept)} decay transients kept of {len(rows)} candidates "
        f"({refusals} refused)"
    )
    families: dict[str, int] = {}
    for transient in kept:
        families[transient.excitation_family] = (
            families.get(transient.excitation_family, 0) + 1
        )
    print(
        "excitation families: "
        + ", ".join(f"{name}={count}" for name, count in sorted(families.items()))
    )
    print(f"wrote {arguments.transients} and {arguments.report}")


def load_transients(path: Path) -> tuple[DecayTransient, ...]:
    """Return the cached decay transients."""

    cached = np.load(path, allow_pickle=False)
    rows = []
    for shot in cached["shots"]:
        shot = int(shot)
        activity = cached[f"activity_{shot}"]
        rows.append(
            DecayTransient(
                shot=shot,
                channels=tuple(str(name) for name in cached[f"channels_{shot}"]),
                time=cached[f"time_{shot}"],
                signal=cached[f"signal_{shot}"],
                noise=cached[f"noise_{shot}"],
                excitation_family=str(cached[f"family_{shot}"][0]),
                driven_families=tuple(str(name) for name in cached[f"driven_{shot}"]),
                peak_drive=float(activity[0]),
                residual_drive=float(activity[1]),
                refused_channels=tuple(str(name) for name in cached[f"refused_{shot}"]),
            )
        )
    return tuple(rows)


def split_transients(
    rows: Sequence[DecayTransient], held_out_shots: set[int]
) -> dict[str, tuple[DecayTransient, ...]]:
    """Split the transients into the training set and the two held-out challenges."""

    training: list[DecayTransient] = []
    held_shots: list[DecayTransient] = []
    held_coil: list[DecayTransient] = []
    for row in rows:
        if row.excitation_family.startswith(HELD_OUT_COIL):
            held_coil.append(row)
        elif row.shot in held_out_shots:
            held_shots.append(row)
        else:
            training.append(row)
    return {
        "held_out_coil": tuple(held_coil),
        "held_out_shots": tuple(held_shots),
        "training": tuple(training),
    }


def promotion_verdict(
    name: str,
    fitted: float,
    interval: tuple[float, float],
    stability: dict[str, Any],
    identified: bool,
    improvement: float,
) -> dict[str, Any]:
    """Apply the standing promotion contract to one resistivity class.

    Four tests, all of which have to pass.  The value has to be identified by the
    data rather than merely returned by the optimiser; it has to be stable when
    shots are dropped; it has to improve prediction on decays the fit never saw;
    and its implied resistivity has to be physically admissible.  A value outside
    the material interval is not automatically refused -- an axisymmetric ring
    standing in for a welded shell is expected to be more resistive than the bulk
    metal -- but it is then recorded as an effective ring resistance rather than as
    a measurement of the metal's resistivity.

    **The promoted interval is the union of the two things that can widen it.**  A
    profile interval answers how far the multiplier can move before the misfit
    rises, and on a clean transient that band can be narrower than the estimator's
    own bias -- so quoting it alone would be tightest exactly where the data is
    cleanest, which is backwards.  The leave-one-out range answers how far the
    answer moves when the shot set changes.  Taking the wider of the two on each
    side gives an interval that no available test contradicts.
    """

    material = _class_material(name)
    resistivity = material.resistivity * fitted
    spread = stability.get("relative_spread", float("nan"))
    stable = bool(np.isfinite(spread) and spread <= 0.5)
    lower = min(interval[0], stability.get("minimum", interval[0]) or interval[0])
    upper = max(interval[1], stability.get("maximum", interval[1]) or interval[1])
    inside = material.resistivity_lower <= resistivity <= material.resistivity_upper
    reasons = []
    if not identified:
        reasons.append("the profile does not close inside the search bounds")
    if not stable:
        reasons.append(f"leave-one-out spread is {spread:.2f} of the median")
    if improvement <= 0.0:
        reasons.append("held-out prediction does not improve")
    return {
        "held_out_improvement": improvement,
        "identified": identified,
        "interval": [lower, upper],
        "leave_one_out_range": [
            stability.get("minimum", float("nan")),
            stability.get("maximum", float("nan")),
        ],
        "leave_one_out_spread": spread,
        "material": material.name,
        "multiplier": fitted,
        "nominal_resistivity": material.resistivity,
        "profile_interval": list(interval),
        "promoted": not reasons,
        "refusals": reasons,
        "resistivity": resistivity,
        "resistivity_interval": [
            material.resistivity * lower,
            material.resistivity * upper,
        ],
        "resistivity_inside_material_interval": inside,
        "stable": stable,
    }


def _class_material(name: str):
    """Return the material whose resistivity a class scales."""

    for family in ("coil_cases", "incon", "rodgr"):
        if resistivity_class(family) == name:
            return passive_material(family) or STAINLESS_STEEL
    return STAINLESS_STEEL


def fit(arguments: argparse.Namespace) -> None:
    """Fit the resistivity classes, profile them, and score the held-out sets."""

    linkage, resistance, coupling, channels = load_linkage(arguments.linkage)
    _, _, _, turns = build_geometry()
    rows = load_transients(arguments.transients)
    cohort = json.loads(arguments.cohort.read_text())
    held_out_shots = {int(shot) for shot in cohort["cohort"]["held_out"]}
    split = split_transients(rows, held_out_shots)
    training = split["training"]
    if len(training) < 3:
        raise SystemExit(f"only {len(training)} training transients survived")
    names = class_names(turns)
    print(
        f"training on {len(training)} transients, holding out "
        f"{len(split['held_out_shots'])} shots and "
        f"{len(split['held_out_coil'])} {HELD_OUT_COIL} experiments; "
        f"classes {names}"
    )

    fitted = fit_resistivity(
        training,
        linkage,
        resistance,
        coupling,
        channels,
        turns,
        names=names,
        mode_count=arguments.modes,
    )
    print(
        f"misfit {fitted.nominal_misfit:.4f} -> {fitted.misfit:.4f} "
        f"({fitted.improvement * 100:.1f}%), variance explained "
        f"{fitted.variance_explained:.4f}"
    )
    for name, value in zip(fitted.names, fitted.multipliers, strict=True):
        print(f"  {name:16s} multiplier {value:.3f}")

    profiles = {
        name: profile_class(
            name,
            fitted,
            training,
            linkage,
            resistance,
            coupling,
            channels,
            turns,
            points=arguments.profile_points,
            mode_count=arguments.modes,
        )
        for name in names
    }
    for name, profile in profiles.items():
        print(
            f"  {name:16s} profile [{profile.lower:.3f}, {profile.upper:.3f}] "
            f"identified={profile.identified}"
        )

    stability = leave_one_out(
        training,
        linkage,
        resistance,
        coupling,
        channels,
        turns,
        names=names,
        mode_count=arguments.modes,
        start=fitted.multipliers,
    )
    scores = {
        label: held_out_score(
            fitted,
            split[label],
            linkage,
            resistance,
            coupling,
            channels,
            turns,
            mode_count=arguments.modes,
        )
        for label in ("held_out_shots", "held_out_coil")
        if split[label]
    }
    for label, score in scores.items():
        print(
            f"  {label:16s} misfit {score['nominal_misfit']:.4f} -> "
            f"{score['fitted_misfit']:.4f} ({score['improvement'] * 100:+.1f}%)"
        )

    improvement = min((score["improvement"] for score in scores.values()), default=0.0)
    verdicts = {
        name: promotion_verdict(
            name,
            fitted.multiplier(name),
            (profiles[name].lower, profiles[name].upper),
            stability.get(name, {}),
            profiles[name].identified,
            improvement,
        )
        for name in names
    }
    for name, verdict in verdicts.items():
        state = "PROMOTED" if verdict["promoted"] else "refused"
        print(
            f"  {name:16s} {state}: {'; '.join(verdict['refusals']) or 'contract met'}"
        )

    modes = mode_set(
        linkage,
        resistance,
        coupling,
        multipliers=np.asarray(
            [fitted.multiplier(resistivity_class(turn.family)) for turn in turns]
        ),
    )
    payload = {
        "circuit_names": list(linkage.names),
        "fit": fitted.as_dict(),
        "held_out_coil": HELD_OUT_COIL,
        "held_out_scores": scores,
        "mode_count": arguments.modes,
        "mode_count_sensitivity": (
            mode_count_sensitivity(
                fitted,
                training,
                linkage,
                resistance,
                coupling,
                channels,
                turns,
            )
            if arguments.sensitivity
            else {}
        ),
        "nominal_time_constants": sorted(
            float(value)
            for value in mode_set(linkage, resistance, coupling).tau
            if value <= 1.0
        )[-8:],
        "profiles": {name: profile.as_dict() for name, profile in profiles.items()},
        "reconstructions": [row.as_dict() for row in fitted.reconstructions],
        "resistivity_classes": {
            turn.name: resistivity_class(turn.family) for turn in turns
        },
        "split": {label: [row.shot for row in rows_] for label, rows_ in split.items()},
        "stability": stability,
        "time_constants": sorted(float(value) for value in modes.tau if value <= 1.0)[
            -8:
        ],
        "verdicts": verdicts,
    }
    arguments.report.parent.mkdir(parents=True, exist_ok=True)
    arguments.report.write_text(json.dumps(payload, indent=1, sort_keys=True) + "\n")
    print(f"wrote {arguments.report}")


def figures(arguments: argparse.Namespace) -> None:
    """Draw what the calibration established."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 9, "svg.fonttype": "none", "figure.dpi": 110})
    linkage, resistance, coupling, channels = load_linkage(arguments.linkage)
    _, _, _, turns = build_geometry()
    report = json.loads(arguments.report.read_text())
    rows = load_transients(arguments.transients)
    out = arguments.figures
    out.mkdir(parents=True, exist_ok=True)

    _linkage_figure(plt, linkage, resistance, turns, out)
    _spectrum_figure(plt, linkage, resistance, coupling, turns, report, out)
    _resistance_figure(plt, linkage, resistance, turns, report, out)
    _reconstruction_figure(
        plt, linkage, resistance, coupling, channels, turns, report, rows, out
    )
    print(f"wrote four figures to {out}")


def _fitted_multipliers(report: dict[str, Any], turns) -> np.ndarray:
    """Return the fitted multiplier of every circuit from a fit record."""

    values = report["fit"]["multipliers"]
    return np.asarray([values[resistivity_class(turn.family)] for turn in turns])


def _linkage_figure(plt, linkage, resistance, turns, out: Path) -> None:
    """Show the linkage's block structure and what it implies per circuit."""

    figure, axes = plt.subplots(1, 3, figsize=(12.4, 4.0))
    scale = np.sqrt(np.outer(np.diag(linkage.matrix), np.diag(linkage.matrix)))
    normalised = linkage.matrix / scale
    image = axes[0].imshow(normalised, cmap="RdBu_r", vmin=-1.0, vmax=1.0)
    axes[0].set_title("coupling coefficient  $L_{ij}/\\sqrt{L_{ii}L_{jj}}$")
    axes[0].set_xlabel("circuit")
    axes[0].set_ylabel("circuit")
    cases = [index for index, turn in enumerate(turns) if turn.enclosed_coil]
    if cases:
        axes[0].axhline(min(cases) - 0.5, color="#333", lw=0.6)
        axes[0].axhline(max(cases) + 0.5, color="#333", lw=0.6)
        axes[0].text(
            1.0,
            float(np.mean(cases)),
            "coil cases",
            fontsize=7,
            rotation=90,
            va="center",
        )
    figure.colorbar(image, ax=axes[0], fraction=0.046)

    self_terms = np.diag(linkage.matrix)
    radii = np.asarray([turn.centroid[0] for turn in turns])
    is_case = np.asarray([bool(turn.enclosed_coil) for turn in turns])
    for mask, colour, label in (
        (~is_case, "#1f5f9c", "vessel and structure"),
        (is_case, "#b0353a", "coil cases"),
    ):
        axes[1].scatter(
            radii[mask], self_terms[mask] * 1e6, s=18, c=colour, label=label
        )
    axes[1].set_xlabel("centroid major radius [m]")
    axes[1].set_ylabel("self inductance [$\\mu$H]")
    axes[1].set_title("self inductance grows with the ring it encloses")
    axes[1].legend(fontsize=7)

    tau = self_terms / resistance
    order = np.argsort(-tau)
    axes[2].barh(
        np.arange(min(14, len(turns))),
        tau[order][:14] * 1e3,
        color=["#b0353a" if is_case[index] else "#1f5f9c" for index in order[:14]],
    )
    axes[2].set_yticks(np.arange(min(14, len(turns))))
    axes[2].set_yticklabels([turns[index].name for index in order[:14]], fontsize=6)
    axes[2].invert_yaxis()
    axes[2].set_xlabel("isolated $L/R$ at nominal resistivity [ms]")
    axes[2].set_title("the slowest circuits taken alone")
    figure.suptitle(
        "Passive flux linkage on the corrected case geometry: "
        f"{len(turns)} circuits, {int(linkage.quadrature_points.sum())} quadrature "
        f"points, reciprocity residual {linkage.reciprocity_residual:.1e}",
        fontsize=9.5,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.94))
    figure.savefig(out / "passive_linkage_structure.svg")
    plt.close(figure)


def _spectrum_figure(plt, linkage, resistance, coupling, turns, report, out) -> None:
    """Compare the nominal and fitted mode spectra and their probe visibility."""

    nominal = mode_set(linkage, resistance, coupling)
    fitted = mode_set(
        linkage,
        resistance,
        coupling,
        multipliers=_fitted_multipliers(report, turns),
    )
    figure, axes = plt.subplots(1, 3, figsize=(12.4, 4.0))
    for modes, colour, label in (
        (nominal, "#1f5f9c", "nominal resistivity"),
        (fitted, "#b0353a", "fitted resistivity"),
    ):
        keep = modes.tau <= 1.0
        axes[0].semilogy(
            np.arange(int(keep.sum())),
            modes.tau[keep] * 1e3,
            "o-",
            ms=3,
            color=colour,
            label=label,
            lw=0.9,
        )
    axes[0].axhspan(
        FASTEST_RESOLVABLE_TIME * 1e3,
        SLOWEST_RESOLVABLE_TIME * 1e3,
        color="#dfe8ef",
        zorder=0,
        label="window can resolve",
    )
    axes[0].set_xlabel("mode, slowest first")
    axes[0].set_ylabel("decay time [ms]")
    axes[0].set_title("mode spectrum")
    axes[0].legend(fontsize=7)

    for modes, colour, label in (
        (nominal, "#1f5f9c", "nominal"),
        (fitted, "#b0353a", "fitted"),
    ):
        strength = np.linalg.norm(modes.signature, axis=0)
        keep = modes.tau <= 1.0
        axes[1].loglog(
            modes.tau[keep] * 1e3,
            strength[keep] * 1e6,
            "o",
            ms=4,
            color=colour,
            label=label,
        )
    axes[1].set_xlabel("decay time [ms]")
    axes[1].set_ylabel("probe signature $\\|Cv\\|$ [$\\mu$T per unit amplitude]")
    axes[1].set_title("how strongly each mode shows up")
    axes[1].legend(fontsize=7)

    weight = np.abs(fitted.vectors[:, np.argsort(-fitted.tau)[:3]])
    weight = weight / np.maximum(weight.max(axis=0, keepdims=True), 1e-30)
    top = np.argsort(-weight.max(axis=1))[:12]
    positions = np.arange(len(top))
    for column, (colour, label) in enumerate(
        (("#b0353a", "slowest"), ("#d18a2c", "second"), ("#3d8b5f", "third"))
    ):
        axes[2].barh(
            positions + 0.26 * (column - 1),
            weight[top, column],
            height=0.24,
            color=colour,
            label=label,
        )
    axes[2].set_yticks(positions)
    axes[2].set_yticklabels([turns[index].name for index in top], fontsize=6)
    axes[2].invert_yaxis()
    axes[2].set_xlabel("normalised circuit participation")
    axes[2].set_title("the three slowest fitted modes are mixtures")
    axes[2].legend(fontsize=7)
    figure.suptitle(
        "A decay mode is a mixture of circuits, and resistance reshapes it",
        fontsize=9.5,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.94))
    figure.savefig(out / "passive_mode_spectra.svg")
    plt.close(figure)


def _resistance_figure(plt, linkage, resistance, turns, report, out) -> None:
    """Show each class's fitted multiplier, its interval and its verdict."""

    verdicts = report["verdicts"]
    names = sorted(verdicts)
    figure, axes = plt.subplots(1, 3, figsize=(12.4, 4.0))
    positions = np.arange(len(names))
    for position, name in zip(positions, names, strict=True):
        row = verdicts[name]
        lower, upper = row["interval"]
        colour = "#3d8b5f" if row["promoted"] else "#b0353a"
        axes[0].plot([lower, upper], [position, position], color=colour, lw=2.4)
        axes[0].plot(row["multiplier"], position, "o", color=colour, ms=6)
        profile = row.get("profile_interval", [lower, upper])
        axes[0].plot(profile, [position - 0.18] * 2, color="#666", lw=1.0)
    axes[0].axvline(1.0, color="#333", ls="--", lw=0.8)
    axes[0].set_xscale("log")
    axes[0].set_yticks(positions)
    axes[0].set_yticklabels(names, fontsize=7)
    axes[0].set_xlabel("resistivity multiplier on nominal")
    axes[0].set_title("thick: promoted interval, thin: profile alone")

    for position, name in zip(positions, names, strict=True):
        row = verdicts[name]
        lower, upper = row["resistivity_interval"]
        axes[1].plot([lower, upper], [position, position], color="#1f5f9c", lw=2.4)
        axes[1].plot(row["resistivity"], position, "o", color="#1f5f9c", ms=6)
        material_lower = row["nominal_resistivity"] * 0.93
        material_upper = row["nominal_resistivity"] * 1.22
        axes[1].plot(
            [material_lower, material_upper],
            [position + 0.2] * 2,
            color="#3d8b5f",
            lw=1.4,
        )
    axes[1].set_xscale("log")
    axes[1].set_yticks(positions)
    axes[1].set_yticklabels(names, fontsize=7)
    axes[1].set_xlabel("effective resistivity [$\\Omega$ m]")
    axes[1].set_title("blue: fitted, green: bulk-material interval")

    multipliers = _fitted_multipliers(report, turns)
    axes[2].scatter(
        resistance,
        resistance * multipliers,
        s=16,
        c=["#b0353a" if turn.enclosed_coil else "#1f5f9c" for turn in turns],
    )
    limits = [float(resistance.min()) * 0.7, float(resistance.max()) * 1.4]
    axes[2].plot(limits, limits, color="#333", ls="--", lw=0.8)
    axes[2].set_xscale("log")
    axes[2].set_yscale("log")
    axes[2].set_xlabel("nominal ring resistance [$\\Omega$]")
    axes[2].set_ylabel("fitted ring resistance [$\\Omega$]")
    axes[2].set_title("red: coil cases, blue: vessel and structure")
    figure.suptitle(
        "Per-class resistivity: the promoted interval is the union of the profile "
        "and the leave-one-out range",
        fontsize=9.5,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.94))
    figure.savefig(out / "passive_group_resistance.svg")
    plt.close(figure)


def _reconstruction_figure(
    plt, linkage, resistance, coupling, channels, turns, report, rows, out
) -> None:
    """Reconstruct a held-out transient under both resistance models."""

    held = set(report["split"].get("held_out_coil", [])) | set(
        report["split"].get("held_out_shots", [])
    )
    candidates = [row for row in rows if row.shot in held] or list(rows)
    transient = max(candidates, key=lambda row: row.signal_to_noise)
    models = {
        "nominal": mode_set(linkage, resistance, coupling),
        "fitted": mode_set(
            linkage,
            resistance,
            coupling,
            multipliers=_fitted_multipliers(report, turns),
        ),
    }
    figure, axes = plt.subplots(1, 3, figsize=(12.4, 4.0))
    channel_index = channel_rows(transient, channels)
    loudest = np.argsort(-np.abs(transient.signal).max(axis=1))[:3]
    for offset, row in enumerate(loudest):
        axes[0].plot(
            transient.time * 1e3,
            transient.signal[row] * 1e3,
            color="#333",
            lw=1.4,
            label="measured" if offset == 0 else None,
        )
    residuals = {}
    for label, colour in (("nominal", "#1f5f9c"), ("fitted", "#b0353a")):
        modes = models[label]
        selection = visible_modes(modes, transient, channel_index)
        outcome = reconstruct(transient, modes, channel_index, selection)
        envelope = np.exp(-transient.time[None, :] / modes.tau[selection][:, None])
        signature = modes.signature[np.ix_(channel_index, selection)]
        predicted = (signature * np.asarray(outcome.amplitudes)) @ envelope
        residuals[label] = (
            np.abs(transient.signal - predicted).max(axis=1) / transient.noise
        )
        for offset, row in enumerate(loudest):
            axes[0].plot(
                transient.time * 1e3,
                predicted[row] * 1e3,
                color=colour,
                ls="--",
                lw=1.1,
                label=label if offset == 0 else None,
            )
        axes[1].plot(
            np.sort(residuals[label]),
            np.linspace(0.0, 1.0, residuals[label].size),
            color=colour,
            label=f"{label}: {outcome.whitened_residual:.2f}",
        )
    axes[0].set_xlabel("time since switch-off [ms]")
    axes[0].set_ylabel("probe field [mT]")
    axes[0].set_title(f"shot {transient.shot}, three loudest channels")
    axes[0].legend(fontsize=7)
    axes[1].set_xscale("log")
    axes[1].set_xlabel("worst-sample residual, in units of the channel's own noise")
    axes[1].set_ylabel("fraction of channels")
    axes[1].set_title("whitened residual, pooled misfit in the legend")
    axes[1].legend(fontsize=7)

    scores = report["held_out_scores"]
    labels = sorted(scores)
    width = 0.36
    for offset, (key, colour) in enumerate(
        (("nominal_misfit", "#1f5f9c"), ("fitted_misfit", "#b0353a"))
    ):
        axes[2].bar(
            np.arange(len(labels)) + width * (offset - 0.5),
            [scores[label][key] for label in labels],
            width=width,
            color=colour,
            label=key.split("_")[0],
        )
    axes[2].set_xticks(np.arange(len(labels)))
    axes[2].set_xticklabels([label.replace("_", "\n") for label in labels], fontsize=7)
    axes[2].set_ylabel("pooled whitened misfit")
    axes[2].set_title("held-out challenges")
    axes[2].legend(fontsize=7)
    figure.suptitle("A held-out decay under both resistance models", fontsize=9.5)
    figure.tight_layout(rect=(0, 0, 1, 0.94))
    figure.savefig(out / "passive_held_out_reconstruction.svg")
    plt.close(figure)


def main(argv: Sequence[str] | None = None) -> None:
    """Run one stage of the passive calibration."""

    cache = Path.home() / ".cache" / "nova-mast"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--store", type=Path, default=SHOT_STORE)
    parser.add_argument(
        "--linkage", type=Path, default=cache / "mast_passive_linkage.npz"
    )
    parser.add_argument(
        "--transients", type=Path, default=cache / "mast_passive_transients.npz"
    )
    parser.add_argument(
        "--cohort", type=Path, default=cache / "mast_calibration_cohort.json"
    )
    parser.add_argument(
        "--screen", type=Path, default=cache / "mast_error_field_screen.json"
    )
    subparsers = parser.add_subparsers(dest="stage", required=True)

    linkage_parser = subparsers.add_parser("inductance")
    linkage_parser.add_argument(
        "--report", type=Path, default=cache / "mast_passive_linkage.json"
    )
    linkage_parser.add_argument("--convergence", action="store_true")
    linkage_parser.set_defaults(handler=inductance)

    transient_parser = subparsers.add_parser("transients")
    transient_parser.add_argument(
        "--report", type=Path, default=cache / "mast_passive_transients.json"
    )
    transient_parser.add_argument(
        "--maximum-drive-share", type=float, default=MAXIMUM_DRIVE_SHARE
    )
    transient_parser.set_defaults(handler=transients)

    fit_parser = subparsers.add_parser("fit")
    fit_parser.add_argument(
        "--report", type=Path, default=cache / "mast_passive_calibration.json"
    )
    fit_parser.add_argument("--modes", type=int, default=RESOLVED_MODE_COUNT)
    fit_parser.add_argument("--profile-points", type=int, default=13)
    fit_parser.add_argument("--sensitivity", action="store_true")
    fit_parser.set_defaults(handler=fit)

    figure_parser = subparsers.add_parser("figures")
    figure_parser.add_argument(
        "--report", type=Path, default=cache / "mast_passive_calibration.json"
    )
    figure_parser.add_argument(
        "--figures",
        type=Path,
        default=Path(__file__).resolve().parents[2]
        / "docs"
        / "figures"
        / "mast-vacuum-floor",
    )
    figure_parser.set_defaults(handler=figures)

    arguments = parser.parse_args(argv)
    arguments.handler(arguments)


if __name__ == "__main__":
    main()
