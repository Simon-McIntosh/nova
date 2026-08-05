"""The passive-calibration driver's caches, splits and promotion contract.

Every stage of this driver hands its result to the next one through a file, and a
field dropped on the way through is silent: the fit still runs, still converges,
and still reports a number, having quietly stopped modelling something.  So the
round trip is tested field by field rather than by whether the next stage crashes.

The promotion contract is tested the same way -- against the cases that must be
refused, not against one that passes.
"""

from __future__ import annotations

import numpy as np
import pytest

from nova.imas.mast_passive_decay_modes import DecayTransient
from nova.scripts.mast_passive_calibration import (
    HELD_OUT_COIL,
    coil_field_response,
    excitation_coil,
    load_transients,
    promotion_verdict,
    split_transients,
)


def transient(shot: int, family: str, *, drives: int = 2) -> DecayTransient:
    """Return a transient with drive columns attached."""

    channels = ("obv01", "obv02", "obr01")
    time = np.linspace(0.0, 0.06, 12)
    generator = np.random.default_rng(shot)
    return DecayTransient(
        shot=shot,
        channels=channels,
        time=time,
        signal=generator.normal(0.0, 1e-3, size=(len(channels), time.size)),
        noise=np.full(len(channels), 1e-5),
        excitation_family=family,
        driven_families=(f"{family}_lower",),
        peak_drive=9000.0,
        residual_drive=0.12,
        drive_patterns=generator.normal(0.0, 1e-7, size=(len(channels), drives)),
        drive_waveforms=generator.normal(0.0, 50.0, size=(drives, time.size)),
        drive_names=tuple(f"p{index + 4}_lower" for index in range(drives)),
    )


class TestTransientCache:
    """What the transient cache has to carry from one stage to the next."""

    def test_the_round_trip_keeps_the_drive_columns(self, tmp_path):
        """A dropped drive column leaves the fit modelling less, and saying nothing.

        The reconstruction simply gains residual it cannot explain, the resistance
        search absorbs what it can into the modes, and every number downstream
        still looks reasonable -- which is why this is asserted rather than left to
        be noticed.
        """

        rows = (transient(14061, "p4"), transient(14070, "p5", drives=1))
        path = tmp_path / "transients.npz"
        np.savez_compressed(
            path,
            **{
                f"{key}_{row.shot}": value
                for row in rows
                for key, value in (
                    ("time", row.time),
                    ("signal", row.signal),
                    ("noise", row.noise),
                    ("channels", np.asarray(row.channels)),
                    ("drive_patterns", row.drive_patterns),
                    ("drive_waveforms", row.drive_waveforms),
                    ("drive_names", np.asarray(row.drive_names, dtype="<U32")),
                    ("family", np.asarray([row.excitation_family])),
                    ("driven", np.asarray(row.driven_families)),
                    ("refused", np.asarray(row.refused_channels)),
                    ("activity", np.asarray([row.peak_drive, row.residual_drive])),
                )
            },
            shots=np.asarray([row.shot for row in rows]),
        )
        restored = load_transients(path)

        assert len(restored) == len(rows)
        for before, after in zip(rows, restored, strict=True):
            assert after.drive_names == before.drive_names
            assert after.drive_patterns is not None
            assert np.allclose(after.drive_patterns, before.drive_patterns)
            assert np.allclose(after.drive_waveforms, before.drive_waveforms)
            assert after.drive_columns.shape[1] == len(before.drive_names)
            assert after.drive_share == pytest.approx(before.drive_share)

    def test_the_round_trip_keeps_the_signal_and_its_noise(self, tmp_path):
        """The whitening is only meaningful if both halves survive together."""

        row = transient(14061, "p4")
        path = tmp_path / "transients.npz"
        np.savez_compressed(
            path,
            **{
                f"{key}_{row.shot}": value
                for key, value in (
                    ("time", row.time),
                    ("signal", row.signal),
                    ("noise", row.noise),
                    ("channels", np.asarray(row.channels)),
                    ("drive_patterns", row.drive_patterns),
                    ("drive_waveforms", row.drive_waveforms),
                    ("drive_names", np.asarray(row.drive_names, dtype="<U32")),
                    ("family", np.asarray([row.excitation_family])),
                    ("driven", np.asarray(row.driven_families)),
                    ("refused", np.asarray(row.refused_channels)),
                    ("activity", np.asarray([row.peak_drive, row.residual_drive])),
                )
            },
            shots=np.asarray([row.shot]),
        )
        restored = load_transients(path)[0]
        assert np.allclose(restored.signal, row.signal)
        assert np.allclose(restored.noise, row.noise)
        assert restored.signal_to_noise == pytest.approx(row.signal_to_noise)
        assert restored.excitation_family == "p4"


class TestSplit:
    """The two held-out challenges every promotion has to clear."""

    def test_the_held_out_coil_never_reaches_training(self):
        """A coil withheld from the fit is what tests generalisation across coils."""

        rows = (
            transient(1, "p4"),
            transient(2, HELD_OUT_COIL),
            transient(3, f"{HELD_OUT_COIL}_lower+{HELD_OUT_COIL}_upper"),
            transient(4, "p3"),
        )
        split = split_transients(rows, held_out_shots=set())
        assert {row.shot for row in split["held_out_coil"]} == {2, 3}
        assert {row.shot for row in split["training"]} == {1, 4}

    def test_the_binding_held_out_shots_are_honoured(self):
        """The declared-before-fitting split is read, never re-derived."""

        rows = (transient(1, "p4"), transient(2, "p3"), transient(3, "p2"))
        split = split_transients(rows, held_out_shots={2})
        assert {row.shot for row in split["held_out_shots"]} == {2}
        assert {row.shot for row in split["training"]} == {1, 3}

    def test_the_coil_takes_precedence_over_the_shot_split(self):
        """A held-out coil's shot must not be counted twice."""

        rows = (transient(1, HELD_OUT_COIL),)
        split = split_transients(rows, held_out_shots={1})
        assert len(split["held_out_coil"]) == 1
        assert not split["held_out_shots"]
        assert not split["training"]

    def test_a_mixed_excitation_reports_no_single_coil(self):
        """A shot driving several sets cannot be attributed to one of them."""

        assert excitation_coil(("p4_lower", "p4_upper")) == "p4"
        assert excitation_coil(("p4_lower", "p6_upper")) == ""


class TestPromotionContract:
    """Four tests a value has to pass, and what each one refuses."""

    def test_a_value_meeting_every_test_is_promoted(self):
        """The contract has to be passable, or it is not a contract."""

        verdict = promotion_verdict(
            "coil_case",
            1.4,
            (1.2, 1.7),
            {"relative_spread": 0.1, "minimum": 1.3, "maximum": 1.5},
            identified=True,
            improvement=0.08,
        )
        assert verdict["promoted"]
        assert not verdict["refusals"]

    def test_an_unidentified_value_is_refused(self):
        """The optimiser always returns a number; the profile says if it means one."""

        verdict = promotion_verdict(
            "coil_case",
            1.4,
            (0.2, 20.0),
            {"relative_spread": 0.1, "minimum": 1.3, "maximum": 1.5},
            identified=False,
            improvement=0.08,
        )
        assert not verdict["promoted"]
        assert any("profile" in reason for reason in verdict["refusals"])

    def test_an_unstable_value_is_refused(self):
        """A value that depends on which shots were in the set is not the machine's."""

        verdict = promotion_verdict(
            "coil_case",
            1.4,
            (1.2, 1.7),
            {"relative_spread": 1.8, "minimum": 0.5, "maximum": 3.0},
            identified=True,
            improvement=0.08,
        )
        assert not verdict["promoted"]
        assert any("leave-one-out" in reason for reason in verdict["refusals"])

    def test_a_value_that_does_not_generalise_is_refused(self):
        """Held-out improvement is what separates a fit from a calibration."""

        verdict = promotion_verdict(
            "coil_case",
            1.4,
            (1.2, 1.7),
            {"relative_spread": 0.1, "minimum": 1.3, "maximum": 1.5},
            identified=True,
            improvement=-0.02,
        )
        assert not verdict["promoted"]
        assert any("held-out" in reason for reason in verdict["refusals"])

    def test_the_interval_is_the_union_of_both_widths(self):
        """A misfit band alone is narrowest where the data is cleanest -- backwards."""

        verdict = promotion_verdict(
            "coil_case",
            1.4,
            (1.39, 1.41),
            {"relative_spread": 0.3, "minimum": 1.1, "maximum": 1.9},
            identified=True,
            improvement=0.08,
        )
        assert verdict["interval"] == [1.1, 1.9]
        assert verdict["profile_interval"] == [1.39, 1.41]

    def test_a_resistivity_outside_the_material_interval_is_flagged_not_refused(self):
        """A welded shell is expected to out-resist the bulk metal it is made of."""

        verdict = promotion_verdict(
            "vessel_shell",
            4.0,
            (3.5, 4.5),
            {"relative_spread": 0.1, "minimum": 3.6, "maximum": 4.4},
            identified=True,
            improvement=0.08,
        )
        assert verdict["promoted"]
        assert not verdict["resistivity_inside_material_interval"]
        assert verdict["resistivity"] > verdict["nominal_resistivity"]


class TestDriveResponse:
    """The coil field pattern the residual drive column is built from."""

    def test_every_family_reaches_every_channel(self):
        """A missing entry would silently zero one coil's modelled contribution."""

        class Target:
            def __init__(self, channel):
                self.channel = channel

        class Model:
            families = ("p4_lower", "p5_upper")
            targets = (Target("obv01"), Target("obr01"))
            response = np.array([[1.0e-6, 2.0e-6], [3.0e-6, 4.0e-6]])

        response = coil_field_response(Model())
        assert set(response) == {"p4_lower", "p5_upper"}
        assert response["p4_lower"] == {"obv01": 1.0e-6, "obr01": 3.0e-6}
        assert response["p5_upper"] == {"obv01": 2.0e-6, "obr01": 4.0e-6}
