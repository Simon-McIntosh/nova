"""Unit tests for the vault / error-field trial machinery (nova.assembly.trial).

Pure-mechanics tests exercise the attribute serialization, cache group-name
hashing, signal generation, gap reduction and sampling on the smallest viable
synthetic inputs -- no manifest netCDF caches or large sample counts. The
driver tests build real trials from the manifest and skip visibly when the
underlying model data (the structural / electromagnetic / overlap netCDFs) is
unavailable.
"""

from dataclasses import dataclass

import matplotlib

matplotlib.use("Agg", force=True)

import numpy as np
import pytest

from nova.assembly.trial import (
    ErrorField,
    Trial,
    TrialAttrs,
    Vault,
    build_error_field,
    build_vault,
    run_trials,
)


@dataclass
class _MechTrial(Trial):
    """Concrete Trial that skips dataset load/build for mechanics tests.

    The overridden ``__post_init__`` sets only the sampling RNG, leaving the
    dataclass fields (samples, component, theta, pdf, ...) populated by the
    generated ``__init__`` so the pure reducers can be driven on hand-built
    inputs without touching the netCDF cache or the physics models.
    """

    filename: str = "mech_trial"

    def __post_init__(self):
        self.rng = np.random.default_rng(self.sead)

    def build(self):  # pragma: no cover - dataset assembly is not under test
        pass


def _two_component_trial(samples: int = 4, **kwargs) -> _MechTrial:
    """Return a mechanics trial with two uniform components."""
    return _MechTrial(
        samples=samples,
        component=["radial", "tangential"],
        theta=[1.5, 1.5],
        pdf=["uniform", "uniform"],
        **kwargs,
    )


class TestTrialAttrs:
    """attrs serializes the hashable scalar parameters only."""

    def test_excludes_transient_fields(self):
        attrs = TrialAttrs(force=True, fixed_coils={0: {"radial": 1.0}}).attrs
        assert "force" not in attrs
        assert "fixed_coils" not in attrs

    def test_bool_cast_to_int(self):
        assert TrialAttrs(adjust_gap=True).attrs["adjust_gap"] == 1
        assert TrialAttrs(adjust_gap=False).attrs["adjust_gap"] == 0

    def test_lists_and_none_excluded(self):
        attrs = TrialAttrs(
            component=["radial"], theta=[1.5], pdf=["uniform"], measured_sectors=None
        ).attrs
        # list-valued and None-valued fields never enter the scalar attrs
        assert "component" not in attrs
        assert "theta" not in attrs
        assert "measured_sectors" not in attrs
        assert attrs["samples"] == TrialAttrs.samples

    def test_field_names_cover_dataclass_fields(self):
        names = TrialAttrs().field_names
        for expected in ["samples", "theta", "pdf", "adjust_gap", "measured_sectors"]:
            assert expected in names


class TestGroupName:
    """group_name is a deterministic hash over the hashable parameters."""

    def test_identical_parameters_hash_equal(self):
        assert _two_component_trial().group_name == _two_component_trial().group_name

    def test_sample_count_changes_hash(self):
        assert (
            _two_component_trial(samples=4).group_name
            != _two_component_trial(samples=5).group_name
        )

    def test_theta_changes_hash(self):
        other = _MechTrial(
            samples=4,
            component=["radial", "tangential"],
            theta=[1.5, 2.0],
            pdf=["uniform", "uniform"],
        )
        assert _two_component_trial().group_name != other.group_name

    def test_transient_fields_do_not_change_hash(self):
        base = _two_component_trial().group_name
        assert _two_component_trial(force=True).group_name == base
        assert _two_component_trial(fixed_coils={0: {"radial": 1.0}}).group_name == base

    def test_measured_sectors_change_hash(self):
        base = _two_component_trial().group_name
        with_sectors = _two_component_trial(measured_sectors=[6, 7]).group_name
        assert with_sectors != base

    def test_measured_sectors_order_independent(self):
        assert (
            _two_component_trial(measured_sectors=[6, 7]).group_name
            == _two_component_trial(measured_sectors=[7, 6]).group_name
        )


class TestBuildSignal:
    """build_signal populates one array per component from its distribution."""

    def test_component_arrays_shape(self):
        trial = _two_component_trial(samples=6)
        trial.build_signal()
        for component in trial.component:
            assert trial.data[component].shape == (6, Trial.ncoil)

    def test_fixed_coils_injected(self):
        trial = _two_component_trial(samples=8, fixed_coils={2: {"radial": 9.9}})
        trial.build_signal()
        # the fixed coil column is broadcast across all samples
        assert np.all(trial.data["radial"].values[:, 2] == 9.9)
        # untouched columns retain their sampled (non-constant) values
        assert not np.all(trial.data["radial"].values[:, 0] == 9.9)


class TestGapReduction:
    """Gap reductions accumulate radial/tangential waveforms per coil."""

    def _built(self, samples: int = 5) -> _MechTrial:
        trial = _two_component_trial(samples=samples)
        trial.build_signal()
        trial.build_gap()
        return trial

    def test_gap_shape_and_nominal_offset(self):
        trial = self._built()
        gap = trial.gap
        assert gap.shape == (5, Trial.ncoil)
        # gap is the summed radial+tangential contribution plus the nominal gap
        raw = trial.data.gap.sum(axis=-1).data
        assert np.allclose(gap, raw + trial.nominal_gap)

    def test_cumulative_gap_is_row_sum(self):
        trial = self._built()
        assert trial.cumulative_gap.shape == (5,)
        assert np.allclose(trial.cumulative_gap, trial.gap.sum(axis=-1))

    def test_gap_recomputed_after_rebuild(self):
        trial = self._built()
        first = trial.gap.copy()
        # zeroing the tangential waveform then rebuilding must refresh the cache
        trial.data["tangential"][:] = 0.0
        trial.build_gap()
        assert not np.allclose(first, trial.gap)

    def test_adjust_nominal_gap_matches_formula(self):
        trial = self._built()
        before = trial.nominal_gap
        quantile = np.quantile(trial.cumulative_gap, 0.99)
        expected = before - (quantile / Trial.ncoil - trial.max_nominal_gap)
        trial.adjust_nominal_gap()
        assert trial.nominal_gap == pytest.approx(expected)


class TestSampling:
    """The distribution samplers return (samples, ncoil) draws."""

    def test_normal_shape_and_scale(self):
        trial = _two_component_trial(samples=50_000)
        draw = trial.normal(4.0)
        assert draw.shape == (50_000, Trial.ncoil)
        # variance argument is the target variance (std == sqrt(variance))
        assert np.std(draw) == pytest.approx(2.0, abs=0.05)

    def test_uniform_bounds(self):
        trial = _two_component_trial(samples=10_000)
        draw = trial.uniform(3.0)
        assert draw.shape == (10_000, Trial.ncoil)
        assert draw.min() >= -3.0
        assert draw.max() <= 3.0


def _skip_without_model_data(factory):
    """Return a built trial, or skip naming the missing model artifact."""
    try:
        return factory()
    except (FileNotFoundError, OSError) as exc:
        pytest.skip(f"trial model netCDF data unavailable: {exc}")


@pytest.fixture(scope="module")
def tiny_vault():
    """Build a tiny vault trial or skip when model data is missing."""
    return _skip_without_model_data(
        lambda: build_vault("baseline_2021", samples=64, force=True)
    )


@pytest.fixture(scope="module")
def tiny_error_field():
    """Build a tiny error-field trial or skip when model data is missing."""
    return _skip_without_model_data(
        lambda: build_error_field("baseline_2021", samples=64, force=True)
    )


class TestDrivers:
    """The lifted drivers build the manifest trials end to end."""

    def test_build_vault(self, tiny_vault):
        assert isinstance(tiny_vault, Vault)
        assert tiny_vault.samples == 64
        assert "peaktopeak" in tiny_vault.data

    def test_build_error_field(self, tiny_error_field):
        assert isinstance(tiny_error_field, ErrorField)
        assert tiny_error_field.samples == 64
        assert "overlap" in tiny_error_field.data

    def test_run_trials_returns_both(self, tiny_vault, tiny_error_field):
        vault, error_field = _skip_without_model_data(
            lambda: run_trials("baseline_2021", samples=64, force=True)
        )
        assert isinstance(vault, Vault)
        assert isinstance(error_field, ErrorField)

    def test_plot_vault_renders(self, tiny_vault):
        from nova.assembly.trial import plot_vault

        assert plot_vault(tiny_vault) is tiny_vault
