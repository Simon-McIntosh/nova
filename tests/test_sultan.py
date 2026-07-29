"""Exercise the SULTAN conductor-test pipeline against a committed cache.

The pipeline is built around a local cache that a campaign is downloaded into
once and read from thereafter, so the whole chain -- campaign metadata, test
plan, phase, shot, sample conditioning, waveform extraction and LTI fitting --
runs with no server reachable provided the cache holds the files. These tests
point the cache at a copy of a committed fixture and clear the FTP credentials
from the environment, which makes any accidental reach for the network a hard
ConnectionError rather than a hang or a skip.

The fixture is synthesised, not extracted: see tests/data/sultan/README for its
provenance and layout. It reproduces the SULTAN file format the readers
implement -- a campaign workbook whose first column carries the strand template,
section headings and per-shot rows, and one comma-separated instrumentation
export per shot -- at the smallest size that still drives every parsing branch.

The LTI fitting is checked separately against a waveform whose output is the
exact response of a known model, so the fit is asked to recover parameters that
are known rather than judged against measured data.
"""

import shutil
from pathlib import Path

import numpy as np
import pandas
import pytest
import scipy

from nova.utilities.importmanager import skip_import

with skip_import("thermofluids"):
    import CoolProp  # noqa: F401
    import ftputil  # noqa: F401
    import nlopt  # noqa: F401

    from nova.thermalhydralic.sultan.campaign import Campaign
    from nova.thermalhydralic.sultan.phase import Phase
    from nova.thermalhydralic.sultan.trial import Trial
    from nova.thermalhydralic.sultan.sample import Sample
    from nova.thermalhydralic.sultan.sourcedata import SourceData
    from nova.thermalhydralic.sultan.sampledata import SampleData
    from nova.thermalhydralic.sultan.profile import Profile
    from nova.thermalhydralic.sultan.waveform import WaveForm
    from nova.thermalhydralic.sultan.remotedata import FTPData
    from nova.thermalhydralic.sultan.model import Model
    from nova.thermalhydralic.sultan.fluidmodel import FluidModel
    from nova.thermalhydralic.sultan.fluidprofile import FluidProfile
    from nova.thermalhydralic.sultan.fitfluid import COEFFICENT, FitFluid

FIXTURE = Path(__file__).parent / "data" / "sultan"

EXPERIMENT = "SYNTH_1"
ABSENT_EXPERIMENT = "SYNTH_2"

#: Sample counts written into the fixture, one entry per ac shot.
SAMPLE_LENGTH = (150, 120)


@pytest.fixture
def cache(tmp_path, monkeypatch):
    """Point the Sultan data directory at a copy of the committed fixture.

    The cache is rebuilt per test: the campaign parses its workbook on first
    read and stores an HDF summary beside it, so a shared cache would leave only
    the first test exercising the workbook path.
    """
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path))
    monkeypatch.delenv("SULTAN_FTP_USER", raising=False)
    monkeypatch.delenv("SULTAN_FTP_PASSWORD", raising=False)
    shutil.copytree(FIXTURE, tmp_path / "Sultan")
    return tmp_path


def test_no_credentials_refuses_the_connection(cache):
    """Reaching the server without credentials fails at the connection."""
    with pytest.raises(ConnectionError):
        FTPData().listdir()


def test_absent_campaign_reaches_for_the_server(cache):
    """A campaign the cache does not hold is looked for remotely, and fails."""
    with pytest.raises(ConnectionError):
        Campaign(ABSENT_EXPERIMENT)


def test_experiment(cache):
    campaign = Campaign(EXPERIMENT)
    assert campaign.experiment == EXPERIMENT


def test_mode(cache):
    campaign = Campaign(EXPERIMENT, "ac")
    assert campaign.mode == "ac"


def test_set_mode(cache):
    campaign = Campaign(EXPERIMENT, "ac")
    campaign.mode = "dc"
    assert campaign.mode == "dc"


def test_rejects_unknown_mode(cache):
    """The mode is validated when it is first read, not at construction."""
    campaign = Campaign(EXPERIMENT, "sideways")
    with pytest.raises(IndexError):
        campaign.mode


def test_campaign_plan(cache):
    campaign = Campaign(EXPERIMENT, "ac")
    assert campaign.plan["ac0"] == "AC Loss Initial"


def test_campaign_metadata_is_cached(cache):
    """The workbook is parsed once and answered from the binary store after."""
    campaign = Campaign(EXPERIMENT, "ac")
    assert Path(campaign.metadatafile).is_file()
    assert Campaign(EXPERIMENT, "ac").plan["ac0"] == campaign.plan["ac0"]


def test_phase(cache):
    campaign = Campaign(EXPERIMENT, "ac")
    phase = Phase(campaign)
    assert phase.index == ["ac0"]


def test_mode_splits_the_phase_index(cache):
    """The ac and dc sections of one workbook index separately."""
    assert Campaign(EXPERIMENT, "ac").index == ["ac0"]
    assert Campaign(EXPERIMENT, "dc").index == ["dc0"]
    assert Campaign(EXPERIMENT, "full").index == ["ac0", "dc0"]


def test_shot_phase_name(cache):
    campaign = Campaign(EXPERIMENT, "full")
    phase = Phase(campaign, -2)
    assert phase.name == "ac0"


def test_phase_name_out_of_range(cache):
    campaign = Campaign(EXPERIMENT, "ac")
    with pytest.raises(IndexError):
        Phase(campaign, 4)


def test_trial_update(cache):
    trial = Trial(EXPERIMENT, -1, "ac")
    trial.campaign.mode = "dc"
    assert trial.phase.name == "dc0"


def test_samplenumber(cache):
    trial = Trial(EXPERIMENT, -1, "ac")
    assert trial.samplenumber == len(SAMPLE_LENGTH)


def test_database_is_shared(cache):
    sample = Sample(EXPERIMENT)
    assert sample.sourcedata.sultandata.database is sample.trial.database


def test_sultandata_update(cache):
    sample = Sample(EXPERIMENT)
    sample.shot = 1
    assert sample.filename == sample.sourcedata.sultandata.filename


def test_sourcedata_name(cache):
    trial = Trial(EXPERIMENT, -1, "ac")
    sourcedata = SourceData(trial, 1)
    assert sourcedata.filename == "SYNTH1A110612"


def test_sourcedata_update(cache):
    trial = Trial(EXPERIMENT, -1, "ac")
    sourcedata = SourceData(trial)
    sourcedata.shot = 1
    assert sourcedata.filename == sourcedata.sultandata.filename


def test_shot_out_of_range(cache):
    trial = Trial(EXPERIMENT, -1, "ac")
    with pytest.raises(IndexError):
        SourceData(trial, 7)


def test_side_out_of_range(cache):
    """The side is validated when it is first read, not at construction."""
    trial = Trial(EXPERIMENT, -1, "ac")
    sourcedata = SourceData(trial, 0, "sideways")
    with pytest.raises(IndexError):
        sourcedata.side


def test_notes_index_the_shot_files(cache):
    """Every plan row carries its note through from the workbook."""
    trial = Trial(EXPERIMENT, -1, "ac")
    notes = trial.notes
    assert list(notes["index"]) == list(trial.plan["File"])
    assert notes.loc[0, "note"] == "1 Hz sweep"


def test_sampledataframe_lowpass_filter(cache):
    trial = Trial(EXPERIMENT, -1, "ac")
    sourcedata = SourceData(trial, 1)
    sampledata = SampleData(sourcedata, _lowpass_filter=False)
    lowpass_array = []
    lowpass_array.append(sampledata.lowpass_filter)
    with sampledata(lowpass_filter=True):
        lowpass_array.append(sampledata.lowpass_filter)
    lowpass_array.append(sampledata.lowpass_filter)
    assert lowpass_array == [False, True, False]


def test_heatindex_brackets_the_pulse(cache):
    """The heat index starts and stops inside the driven window."""
    sample = Sample(EXPERIMENT, 0, "Left")
    heatindex = sample.heatindex
    current = sample.rawdata.loc[:, ("Ipulse", "A")].abs()
    threshold = heatindex.threshold * current.max()
    assert current.iloc[heatindex.start] >= threshold
    assert current.iloc[heatindex.stop - 1] >= threshold
    assert current.iloc[heatindex.start - 1] < threshold


def test_both_sides_read(cache):
    """The left and right instrumentation rakes both resolve."""
    for side in ("Left", "Right"):
        sample = Sample(EXPERIMENT, 0, side)
        assert sample.side == side
        assert len(sample.rawdata) == SAMPLE_LENGTH[0]


def test_profile_offset(cache):
    profile = Profile(Sample(EXPERIMENT, 0, "Left"))
    assert np.isclose(
        profile.timeseries(profile.sample.heatindex.start), (0.0, 0.0)
    ).all()


def test_shot_change_reloads_the_profile(cache):
    """Advancing the shot re-reads the timeseries rather than serving the old."""
    sample = Sample(EXPERIMENT, 0, "Left")
    profile = Profile(sample)
    assert len(profile.time) == SAMPLE_LENGTH[0]
    sample.shot = 1
    assert len(profile.time) == SAMPLE_LENGTH[1]


def test_profile_reports_a_steady_shot(cache):
    """The synthesised pulse settles, so all three stability ratios pass."""
    profile = Profile(Sample(EXPERIMENT, 0, "Left"))
    assert profile.steady


def test_waveform_carries_the_excitation_amplitudes(cache):
    """The waveform frame reports every amplitude the fit coefficients read."""
    waveform = WaveForm(Profile(Sample(EXPERIMENT, 0, "Left")), 0.9)
    data = waveform.data
    assert list(data.columns) == [
        "time",
        "field",
        "fieldsq",
        "fieldrate",
        "fieldratesq",
        "output",
    ]
    amplitude = data.attrs["field_amplitude"]
    rate = data.attrs["fieldrate_amplitude"]
    assert np.isclose(data.attrs["fieldsq_amplitude"], amplitude**2)
    assert np.isclose(data.attrs["fieldratesq_amplitude"], rate**2)
    assert np.isclose(rate, 2 * np.pi * data.attrs["frequency"] * amplitude)
    assert data.attrs["samplenumber"] == len(data)


def test_waveform_zeroes_the_input_outside_the_pulse(cache):
    """With pulse set, the drive is zero everywhere outside the heated window."""
    waveform = WaveForm(Profile(Sample(EXPERIMENT, 0, "Left")), 0.9, _pulse=True)
    data = waveform.data
    outside = np.full(len(data), True)
    outside[data.attrs["heatindex"]] = False
    assert np.all(data.fieldrate.to_numpy()[outside] == 0)
    assert np.any(data.fieldrate.to_numpy()[~outside] != 0)


def test_fluidprofile_fits_a_cached_shot(cache):
    """The whole chain fits a cached shot and reports consistent coefficients."""
    fluidprofile = FluidProfile(Sample(EXPERIMENT, 0, "Left"), [4], 0, verbose=False)
    coefficents = fluidprofile.coefficents
    assert set(COEFFICENT) <= set(coefficents.index)
    # the reported steady state is half the drive amplitude times the dc gain
    assert np.isclose(
        coefficents.steadystate,
        0.5 * coefficents.fieldratesq_amplitude * coefficents.dcgain,
    )
    assert coefficents.pole0 > 0
    assert coefficents.energy_model > 0


def synthetic_waveform(model, dcgain, pole, frequency=1.0, samples=256):
    """Return a waveform frame whose output is the exact response of model.

    The drive is the squared field rate of a sinusoidal excitation gated to a
    pulse, matching what WaveForm builds, and the output is that drive pushed
    through a known LTI model. A fit is then asked to recover parameters that
    are known exactly rather than judged against measured heat.
    """
    time = np.linspace(0, 24, samples)
    rate_amplitude = 2 * np.pi * frequency * 0.2
    fieldrate = rate_amplitude * np.cos(2 * np.pi * frequency * time)
    heated = time < 16
    fieldrate = np.where(heated, fieldrate, 0.0)
    fluid = FluidModel(model)
    fluid.model.update_pole(pole)
    fluid.model.update_dcgain(dcgain)
    fluid.timeseries = (time, fieldrate**2)
    data = pandas.DataFrame(
        {
            "time": time,
            "field": np.zeros(samples),
            "fieldsq": np.zeros(samples),
            "fieldrate": fieldrate,
            "fieldratesq": fieldrate**2,
            "output": fluid.output,
        }
    )
    data.attrs |= {
        "filename": "synthetic",
        "field_amplitude": 0.2,
        "fieldsq_amplitude": 0.04,
        "fieldrate_amplitude": rate_amplitude,
        "fieldratesq_amplitude": rate_amplitude**2,
        "frequency": frequency,
        "massflow": 8.0,
        "samplenumber": samples,
        "heatindex": slice(0, int(heated.sum())),
    }
    return data


def test_fitfluid_recovers_a_known_model():
    """A fit against a model's own response returns that model's parameters."""
    dcgain, pole = 30.0, 0.8
    data = synthetic_waveform(Model(4, delay=False), dcgain, pole)
    fit = FitFluid(Model(4, delay=False), verbose=False)
    fit.optimize(data)
    assert np.isclose(fit.fluid.model.dcgain, dcgain, rtol=1e-2)
    assert np.isclose(fit.fluid.model.repeated_pole[0], pole, rtol=1e-2)


def test_fitfluid_reports_every_declared_coefficient():
    """coefficents returns exactly the declared names, all finite."""
    data = synthetic_waveform(Model(4, delay=False), 30.0, 0.8)
    fit = FitFluid(Model(4, delay=False), verbose=False)
    fit.optimize(data)
    coefficents = fit.coefficents
    assert list(coefficents.index) == list(COEFFICENT)
    assert np.isfinite(coefficents.to_numpy().astype(float)).all()


def test_fitfluid_reduces_the_residual():
    """Optimising lowers the residual below the seeded model's."""
    data = synthetic_waveform(Model(4, delay=False), 30.0, 0.8)
    fit = FitFluid(Model(4, delay=False), verbose=False)
    fit.extract_data(data)
    fit.initialize_model()
    seeded = fit.model_error(fit.fluid.model.vector)
    fit.optimize(data)
    assert fit.model_error(fit.fluid.model.vector) < seeded


def test_fitfluid_steadystate_follows_the_dc_gain():
    """The reported steady state is half the drive amplitude times the gain."""
    data = synthetic_waveform(Model(4, delay=False), 30.0, 0.8)
    fit = FitFluid(Model(4, delay=False), verbose=False)
    fit.optimize(data)
    assert np.isclose(
        fit.steadystate,
        0.5 * data.attrs["fieldratesq_amplitude"] * fit.fluid.model.dcgain,
    )


def test_fitfluid_rejects_an_unknown_optimizer():
    data = synthetic_waveform(Model(4, delay=False), 30.0, 0.8)
    fit = FitFluid(Model(4, delay=False), verbose=False)
    with pytest.raises(IndexError):
        fit.optimize(data, optimizer="wishful")


def test_fluidmodel_settles_on_the_dc_gain():
    """A constant drive drives the model output to gain times that drive."""
    fluid = FluidModel(Model(4, delay=False))
    fluid.model.update_pole(2.0)
    fluid.model.update_dcgain(7.5)
    time = np.linspace(0, 60, 2001)
    fluid.timeseries = (time, np.ones_like(time))
    assert np.isclose(fluid.output[-1], 7.5, rtol=1e-6)


def test_model_dc_gain():
    model = Model(6, _dcgain=20.5)
    assert model.dcgain == 20.5


def test_model_dc_gain_step():
    model = Model(6, _dcgain=20.5)
    assert np.isclose(scipy.signal.step(model.lti, T=[0, 1e4])[1][-1], 20.5)


def test_model_delay_boolean():
    model = Model([6, 3], delay=False)
    assert len(model.vector) == 3


def test_model_delay_update():
    model = Model(6, delay=False)
    model.delay = True
    assert len(model.vector) == 3


def test_model_rejects_a_short_seed():
    with pytest.raises(IndexError):
        Model([6, 3], _pole=[0.5])


if __name__ == "__main__":
    pytest.main([__file__])
