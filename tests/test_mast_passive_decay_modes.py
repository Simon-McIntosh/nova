"""The mode-resolved decay fit, on decays whose resistance is known in advance.

A fit is only worth trusting if it recovers an answer somebody planted, so the
tests here synthesise decays from a chosen resistivity model and ask the fit to
find it back.  That makes every claim checkable: the recovered multiplier, the
profile that should close around it, the profile that should stay open when two
classes are made indistinguishable, and the held-out score that should improve
only when the model is right.

The screens are tested against the case each one exists to refuse rather than
against a shot that happens to pass.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from nova.imas.mast_passive_decay_modes import (
    DEFAULT_RESISTIVITY_CLASS,
    FASTEST_RESOLVABLE_TIME,
    MULTIPLIER_BOUNDS,
    RESOLVED_MODE_COUNT,
    SLOWEST_RESOLVABLE_TIME,
    DecayModeError,
    DecayTransient,
    channel_rows,
    circuit_multipliers,
    class_names,
    decay_misfit,
    fit_resistivity,
    held_out_score,
    leave_one_out,
    mode_set,
    profile_class,
    read_transient,
    reconstruct,
    resistivity_class,
    resolvable_modes,
    visible_modes,
)
from nova.imas.mast_passive_response import PassiveError
from nova.imas.mast_passive_inductance import (
    PassiveTurn,
    linkage_matrix,
    nominal_resistance,
    probe_coupling,
)
from nova.imas.mast_vacuum_cohort import EXCITATION_CURRENT, ShotWaveforms
from nova.imas.mast_vacuum_response import ProbeTarget


def rectangle(r: float, z: float, width: float, height: float) -> np.ndarray:
    """Return an axis-aligned rectangular section centred on ``(r, z)``."""

    return np.array(
        [
            [r - 0.5 * width, z - 0.5 * height],
            [r + 0.5 * width, z - 0.5 * height],
            [r + 0.5 * width, z + 0.5 * height],
            [r - 0.5 * width, z + 0.5 * height],
        ]
    )


CIRCUITS = (
    PassiveTurn("wall", "vertw", "", rectangle(1.9, 0.0, 0.02, 2.0)),
    PassiveTurn("crown", "endcrown_u", "", rectangle(0.6, 0.9, 0.08, 0.20)),
    PassiveTurn(
        "case_upper", "coil_cases", "p4_upper", rectangle(1.5, 1.1, 0.19, 0.19)
    ),
    PassiveTurn(
        "case_lower", "coil_cases", "p4_lower", rectangle(1.5, -1.1, 0.19, 0.19)
    ),
    PassiveTurn("column", "incon", "", rectangle(0.2, 0.0, 0.02, 2.4)),
)
"""Five circuits spanning three resistivity classes, far enough apart to separate.

The pair of coil cases sits symmetrically about the midplane, which is what gives
the probe array an up-down pattern to see and makes the case class reachable at
all; the wall and the centre column give the other two classes their own patterns.
"""

TARGETS = tuple(
    ProbeTarget(
        channel=f"obv{index + 1:02d}",
        family="obv",
        registry_index=index,
        r=1.75,
        z=height,
        radial_cosine=0.0,
        axial_sine=1.0,
    )
    for index, height in enumerate(np.linspace(-1.4, 1.4, 15))
) + tuple(
    ProbeTarget(
        channel=f"obr{index + 1:02d}",
        family="obr",
        registry_index=15 + index,
        r=1.75,
        z=height,
        radial_cosine=1.0,
        axial_sine=0.0,
    )
    for index, height in enumerate(np.linspace(-1.2, 1.2, 9))
)


@pytest.fixture(scope="module")
def system():
    """Return the linkage, nominal resistance, coupling and channels."""

    linkage = linkage_matrix(CIRCUITS)
    return (
        linkage,
        nominal_resistance(CIRCUITS),
        probe_coupling(CIRCUITS, TARGETS),
        [target.channel for target in TARGETS],
    )


def synthesise(
    system,
    truth: dict[str, float],
    *,
    shot: int,
    modes: int = RESOLVED_MODE_COUNT,
    noise: float = 1.0e-6,
    seed: int = 0,
    excitation_family: str = "p4",
    weights: tuple[float, ...] | None = None,
) -> DecayTransient:
    """Build a decay the given resistivity model produces exactly.

    The amplitudes are chosen here and refitted by the code under test, which is
    what makes a recovered multiplier a recovery rather than a coincidence: the
    fit is never told how much of each mode was present.
    """

    linkage, resistance, coupling, channels = system
    names = class_names(CIRCUITS)
    values = np.asarray([truth[name] for name in names])
    predicted = mode_set(
        linkage,
        resistance,
        coupling,
        multipliers=circuit_multipliers(CIRCUITS, names, values),
    )
    selection = resolvable_modes(predicted)[:modes]
    time = np.linspace(0.0, 0.06, 240)
    generator = np.random.default_rng(seed)
    amplitude = weights or tuple(1.0 for _ in selection)
    signal = np.zeros((len(channels), time.size))
    for weight, index in zip(amplitude, selection, strict=False):
        envelope = np.exp(-time / predicted.tau[index])
        signal += weight * np.outer(predicted.signature[:, index], envelope)
    signal = signal + generator.normal(0.0, noise, size=signal.shape)
    return DecayTransient(
        shot=shot,
        channels=tuple(channels),
        time=time,
        signal=signal,
        noise=np.full(len(channels), noise),
        excitation_family=excitation_family,
        driven_families=(f"{excitation_family}_upper",),
        peak_drive=1.0e4,
        residual_drive=0.0,
    )


class TestResistivityClasses:
    """Which multiplier a family draws on, declared by what the conductor is."""

    def test_the_named_families_carry_their_own_class(self):
        """A source that distinguishes a material earns that material a class."""

        assert resistivity_class("coil_cases") == "coil_case"
        assert resistivity_class("incon") == "centre_column"
        assert resistivity_class("rodgr") == "centre_rod"

    def test_an_undistinguished_family_falls_to_the_vessel(self):
        """One fabrication in one material shares one multiplier."""

        for family in ("vertw", "lhorw", "ring", "p2larm", "mid"):
            assert resistivity_class(family) == DEFAULT_RESISTIVITY_CLASS

    def test_classes_expand_onto_one_multiplier_per_circuit(self):
        """The fit's few unknowns reach every circuit through its family."""

        names = class_names(CIRCUITS)
        values = np.asarray([1.0 + index for index in range(len(names))])
        expanded = circuit_multipliers(CIRCUITS, names, values)
        assert expanded.shape == (len(CIRCUITS),)
        assert expanded[2] == expanded[3]
        assert expanded[0] != expanded[4]

    def test_only_the_populated_classes_are_carried(self):
        """A class with no circuit in it would be an unconstrained free parameter."""

        assert set(class_names(CIRCUITS)) == {
            "centre_column",
            "coil_case",
            DEFAULT_RESISTIVITY_CLASS,
        }


class TestModeSet:
    """The eigenproblem that ties every decay time to its own spatial pattern."""

    def test_every_mode_decays(self, system):
        """A positive resistance and a positive-definite linkage cannot ring."""

        linkage, resistance, coupling, _ = system
        modes = mode_set(linkage, resistance, coupling)
        assert modes.mode_count == len(CIRCUITS)
        assert np.all(modes.tau > 0.0)

    def test_the_modes_are_ordered_slowest_first(self, system):
        """The slowest mode is the one whose history a per-slice fit cannot absorb."""

        linkage, resistance, coupling, _ = system
        modes = mode_set(linkage, resistance, coupling)
        assert np.all(np.diff(modes.tau) <= 0.0)

    def test_raising_resistance_shortens_every_time_constant(self, system):
        """A uniform multiplier scales the whole spectrum, which is what fixes it."""

        linkage, resistance, coupling, _ = system
        base = mode_set(linkage, resistance, coupling)
        doubled = mode_set(
            linkage, resistance, coupling, multipliers=np.full(len(CIRCUITS), 2.0)
        )
        assert np.allclose(doubled.tau, 0.5 * base.tau, rtol=1e-9)

    def test_a_non_positive_multiplier_is_refused(self, system):
        """A zero resistance is a superconductor, not a slow circuit."""

        linkage, resistance, coupling, _ = system
        with pytest.raises(DecayModeError):
            mode_set(linkage, resistance, coupling, multipliers=np.zeros(len(CIRCUITS)))

    def test_a_mismatched_multiplier_length_is_refused(self, system):
        """Silently broadcasting would scale the wrong circuits."""

        linkage, resistance, coupling, _ = system
        with pytest.raises(DecayModeError):
            mode_set(linkage, resistance, coupling, multipliers=np.ones(2))

    def test_the_resolvable_band_excludes_both_extremes(self, system):
        """A mode outside the window is a basis function, not an observation."""

        linkage, resistance, coupling, _ = system
        modes = mode_set(linkage, resistance, coupling)
        admitted = resolvable_modes(modes)
        assert np.all(modes.tau[admitted] >= FASTEST_RESOLVABLE_TIME)
        assert np.all(modes.tau[admitted] <= SLOWEST_RESOLVABLE_TIME)

    def test_no_resolvable_mode_is_an_error_not_an_empty_fit(self, system):
        """Fitting nothing would report a perfect residual on no information."""

        linkage, resistance, coupling, _ = system
        modes = mode_set(
            linkage,
            resistance,
            coupling,
            multipliers=np.full(len(CIRCUITS), MULTIPLIER_BOUNDS[1]),
        )
        with pytest.raises(DecayModeError):
            visible_modes(
                modes,
                synthesise(
                    system, {name: 1.0 for name in class_names(CIRCUITS)}, shot=1
                ),
                np.arange(len(TARGETS)),
                slowest=FASTEST_RESOLVABLE_TIME / 2.0,
                fastest=FASTEST_RESOLVABLE_TIME / 4.0,
            )


class TestReconstruction:
    """Fitting the free amplitudes of a fixed mode set to one decay."""

    def test_the_true_model_explains_a_clean_decay(self, system):
        """A decay built from a mode set is reproduced by that same mode set."""

        linkage, resistance, coupling, channels = system
        truth = {name: 1.0 for name in class_names(CIRCUITS)}
        transient = synthesise(system, truth, shot=11, noise=1.0e-9)
        modes = mode_set(linkage, resistance, coupling)
        rows = channel_rows(transient, channels)
        selection = visible_modes(modes, transient, rows)
        outcome = reconstruct(transient, modes, rows, selection)
        assert outcome.variance_explained > 0.999

    def test_a_wrong_model_leaves_residual(self, system):
        """The leftover is what carries the statement about resistance."""

        linkage, resistance, coupling, channels = system
        names = class_names(CIRCUITS)
        transient = synthesise(
            system, {name: 1.0 for name in names}, shot=12, noise=1e-9
        )
        wrong = mode_set(
            linkage, resistance, coupling, multipliers=np.full(len(CIRCUITS), 6.0)
        )
        rows = channel_rows(transient, channels)
        selection = visible_modes(wrong, transient, rows)
        outcome = reconstruct(transient, wrong, rows, selection)
        assert outcome.variance_explained < 0.99

    def test_an_unposed_channel_is_refused(self, system):
        """A channel the model cannot predict must not be silently dropped."""

        transient = synthesise(
            system, {name: 1.0 for name in class_names(CIRCUITS)}, shot=13
        )
        with pytest.raises(DecayModeError):
            channel_rows(transient, ["obv01", "obv02"])


class TestFit:
    """Recovering a planted resistivity model from synthetic decays."""

    @pytest.fixture(scope="class")
    def truth(self):
        """Return the resistivity model the synthetic decays are built from."""

        return {"centre_column": 1.0, "coil_case": 3.0, DEFAULT_RESISTIVITY_CLASS: 2.0}

    @pytest.fixture(scope="class")
    def training(self, system, truth):
        """Return several decays of the same machine with different mode mixtures."""

        return tuple(
            synthesise(
                system,
                truth,
                shot=100 + index,
                seed=index,
                noise=2.0e-8,
                weights=weights,
            )
            for index, weights in enumerate(
                ((1.0, 0.4, 0.2), (0.3, 1.0, 0.5), (0.8, 0.8, 1.0), (1.0, 0.1, 0.9))
            )
        )

    def test_the_fit_recovers_the_planted_multipliers(self, system, truth, training):
        """The whole point: a known resistance model is found back from decays."""

        linkage, resistance, coupling, channels = system
        outcome = fit_resistivity(
            training, linkage, resistance, coupling, channels, CIRCUITS
        )
        for name, planted in truth.items():
            assert outcome.multiplier(name) == pytest.approx(planted, rel=0.25)

    def test_a_class_is_recovered_as_precisely_as_the_probes_see_it(
        self, system, truth, training
    ):
        """Recovery accuracy tracks how strongly a class shows up in the array.

        The coil-case pair sits a probe-array width away and produces a clear
        up-down pattern; the centre column at small major radius produces almost
        none of the signal this outboard array measures.  The fit must recover the
        first far more tightly than the second, because the data says more about
        it -- a fit that returned both equally well would be reporting its prior.
        """

        linkage, resistance, coupling, channels = system
        outcome = fit_resistivity(
            training, linkage, resistance, coupling, channels, CIRCUITS
        )
        seen = abs(outcome.multiplier("coil_case") / truth["coil_case"] - 1.0)
        unseen = abs(outcome.multiplier("centre_column") / truth["centre_column"] - 1.0)
        assert seen < 0.05
        assert seen < unseen

    def test_the_fit_improves_on_the_nominal_model(self, system, training):
        """A model that recovers the truth must beat the model that does not."""

        linkage, resistance, coupling, channels = system
        outcome = fit_resistivity(
            training, linkage, resistance, coupling, channels, CIRCUITS
        )
        assert outcome.misfit < outcome.nominal_misfit
        assert outcome.improvement > 0.0

    def test_a_populated_class_profiles_to_a_closed_interval(
        self, system, truth, training
    ):
        """An identified class has a profile that shuts inside the search bounds."""

        linkage, resistance, coupling, channels = system
        outcome = fit_resistivity(
            training, linkage, resistance, coupling, channels, CIRCUITS
        )
        profile = profile_class(
            "coil_case",
            outcome,
            training,
            linkage,
            resistance,
            coupling,
            channels,
            CIRCUITS,
            points=7,
        )
        assert profile.identified
        assert profile.lower > MULTIPLIER_BOUNDS[0]
        assert profile.upper < MULTIPLIER_BOUNDS[1]
        assert profile.upper / profile.lower < 1.5
        assert profile.curvature > 0.0

    def test_a_decay_buried_in_noise_profiles_open(self, system, truth):
        """With the transient below the noise, no class may report itself pinned.

        This is the case the identifiability test exists for.  The optimiser still
        returns a number for every class -- it always does -- so the only thing
        standing between that number and a promotion is a profile that refuses to
        close, and it has to refuse here.
        """

        linkage, resistance, coupling, channels = system
        buried = tuple(
            synthesise(system, truth, shot=300 + index, seed=index, noise=1.0)
            for index in range(3)
        )
        outcome = fit_resistivity(
            buried, linkage, resistance, coupling, channels, CIRCUITS
        )
        profile = profile_class(
            "coil_case",
            outcome,
            buried,
            linkage,
            resistance,
            coupling,
            channels,
            CIRCUITS,
            points=7,
        )
        assert not profile.identified
        assert profile.upper / profile.lower > 50.0

    def test_the_fitted_model_predicts_a_decay_it_never_saw(
        self, system, truth, training
    ):
        """Held-out improvement is the test the promotion contract turns on."""

        linkage, resistance, coupling, channels = system
        outcome = fit_resistivity(
            training, linkage, resistance, coupling, channels, CIRCUITS
        )
        unseen = (
            synthesise(
                system, truth, shot=200, seed=9, noise=2e-8, weights=(0.6, 0.7, 0.3)
            ),
        )
        score = held_out_score(
            outcome, unseen, linkage, resistance, coupling, channels, CIRCUITS
        )
        assert score["improvement"] > 0.0
        assert score["fitted_misfit"] < score["nominal_misfit"]

    def test_dropping_a_shot_does_not_move_an_identified_class(
        self, system, truth, training
    ):
        """Cross-shot stability is a property of the answer, not of the sample."""

        linkage, resistance, coupling, channels = system
        outcome = fit_resistivity(
            training, linkage, resistance, coupling, channels, CIRCUITS
        )
        spread = leave_one_out(
            training,
            linkage,
            resistance,
            coupling,
            channels,
            CIRCUITS,
            names=outcome.names,
            start=outcome.multipliers,
        )
        assert spread["coil_case"]["relative_spread"] < 0.5

    def test_a_starting_point_does_not_change_where_the_fit_lands(
        self, system, truth, training
    ):
        """A refit from a known optimum is an economy, not a different answer."""

        linkage, resistance, coupling, channels = system
        cold = fit_resistivity(
            training, linkage, resistance, coupling, channels, CIRCUITS
        )
        warm = fit_resistivity(
            training,
            linkage,
            resistance,
            coupling,
            channels,
            CIRCUITS,
            start=cold.multipliers,
            maxiter=40,
        )
        assert warm.misfit == pytest.approx(cold.misfit, rel=0.02)

    def test_an_empty_transient_set_is_refused(self, system):
        """A fit on nothing would report a perfect residual."""

        linkage, resistance, coupling, channels = system
        with pytest.raises(DecayModeError):
            decay_misfit(
                (),
                linkage,
                resistance,
                coupling,
                channels,
                CIRCUITS,
                class_names(CIRCUITS),
                np.ones(len(class_names(CIRCUITS))),
            )


def switched_shot(
    *,
    shot: int = 500,
    stop: float = 0.5,
    hold: float = 2.0 * EXCITATION_CURRENT,
    tail_slope: float = 0.0,
    offset: float = 1.0e-3,
    dead_channels: tuple[str, ...] = (),
    samples: int = 900,
) -> ShotWaveforms:
    """Build a shot that drives one coil, holds, then switches it off.

    ``tail_slope`` leaves the drive still creeping after the switch-off while
    staying below the excitation threshold, which is the case the settle test
    exists to catch: a window does open, and what is inside it is not a free
    decay.  A tail that stayed above the threshold would be refused earlier, by
    the window never opening at all.
    """

    time = np.linspace(-0.2, 1.0, samples)
    current = np.where((time >= 0.0) & (time <= stop), hold, 0.0) + np.where(
        time > stop, tail_slope * (time - stop), 0.0
    )
    generator = np.random.default_rng(shot)
    probes = {}
    for index, target in enumerate(TARGETS):
        signal = np.full_like(time, offset * (1 + index % 4))
        signal += 1.0e-3 * np.where(
            time > stop, np.exp(-(time - stop) / 0.03), (time >= 0.0).astype(float)
        )
        signal += generator.normal(0.0, 2.0e-6, size=time.shape)
        if target.channel in dead_channels:
            signal = np.full_like(time, np.nan)
        probes[target.channel] = signal
    return ShotWaveforms(
        shot=shot,
        time=time,
        drives={"p4_upper": current},
        probes=probes,
        plasma_current=np.zeros_like(time),
        sample_mask=np.ones(time.shape, dtype=bool),
        baseline_mask=time < -0.05,
    )


class TestTransientReading:
    """The window, the offset and the noise, all taken from the shot itself."""

    def test_the_window_opens_after_the_switch_off(self):
        """A free decay starts where the deliberate excitation stops."""

        transient = read_transient(switched_shot(), TARGETS, excitation_family="p4")
        assert transient.time[0] == pytest.approx(0.0, abs=1e-9)
        assert transient.sample_count > 20
        assert transient.driven_families == ("p4_upper",)

    def test_the_standing_offset_is_removed(self):
        """Each channel carries a different pedestal, and none of it is signal."""

        transient = read_transient(
            switched_shot(offset=5.0e-3), TARGETS, excitation_family="p4"
        )
        assert abs(float(transient.signal[:, -1].mean())) < 2.0e-4

    def test_the_noise_floor_is_measured_per_channel(self):
        """A nominal floor would whiten a quiet channel as if it were loud."""

        transient = read_transient(switched_shot(), TARGETS, excitation_family="p4")
        assert transient.noise.size == len(transient.channels)
        assert np.all(transient.noise > 0.0)
        assert transient.signal_to_noise > 1.0

    def test_a_refused_channel_never_enters_the_pattern(self):
        """A screen that removes a channel must remove it from the fit as well."""

        refused = ("obv01", "obv02", "obr01")
        transient = read_transient(
            switched_shot(),
            TARGETS,
            excitation_family="p4",
            refused_channels=refused,
        )
        assert not set(refused) & set(transient.channels)
        assert transient.refused_channels == tuple(sorted(refused))

    def test_a_channel_with_no_samples_is_dropped(self):
        """A padded record is absent data, not a flat measurement."""

        transient = read_transient(
            switched_shot(dead_channels=("obv03", "obv04")),
            TARGETS,
            excitation_family="p4",
        )
        assert "obv03" not in transient.channels
        assert "obv04" not in transient.channels

    def test_too_few_channels_is_refused(self):
        """A spatial pattern on three probes is not a pattern."""

        with pytest.raises(DecayModeError):
            read_transient(
                switched_shot(),
                TARGETS,
                excitation_family="p4",
                refused_channels=[target.channel for target in TARGETS[:-3]],
            )

    def test_a_still_ramping_drive_is_visible_in_the_residual(self):
        """A drive that keeps moving has no place in a free-decay model."""

        quiet = read_transient(switched_shot(), TARGETS, excitation_family="p4")
        creeping = read_transient(
            switched_shot(tail_slope=1.6e3), TARGETS, excitation_family="p4"
        )
        assert quiet.residual_drive < 1.0e-9
        assert creeping.residual_drive > 0.02

    def test_a_drive_that_never_stops_opens_no_window(self):
        """A record that ends still driven has no free decay in it to read."""

        with pytest.raises(PassiveError):
            read_transient(
                switched_shot(tail_slope=2.0e4), TARGETS, excitation_family="p4"
            )

    def test_a_shot_that_drove_nothing_is_refused(self):
        """There is no decay window without a switch-off to open it."""

        idle = switched_shot(hold=0.0)
        with pytest.raises(PassiveError):
            read_transient(idle, TARGETS, excitation_family="p4")

    def test_the_record_reports_what_it_read(self):
        """The provenance a promotion rests on has to be inspectable."""

        payload = read_transient(
            switched_shot(), TARGETS, excitation_family="p4"
        ).as_dict()
        assert payload["channel_count"] == len(TARGETS)
        assert payload["excitation_family"] == "p4"
        assert payload["window_span"] > 0.0
        assert math.isfinite(payload["signal_to_noise"])
