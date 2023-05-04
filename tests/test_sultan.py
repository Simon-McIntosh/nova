import pytest
import numpy as np
import scipy

from nova import skip_import
with skip_import('thermofluids'):
    import ftputil

    from nova.thermalhydralic.sultan.campaign import Campaign
    from nova.thermalhydralic.sultan.phase import Phase
    from nova.thermalhydralic.sultan.trial import Trial
    from nova.thermalhydralic.sultan.sample import Sample
    from nova.thermalhydralic.sultan.sourcedata import SourceData
    from nova.thermalhydralic.sultan.sampledata import SampleData
    from nova.thermalhydralic.sultan.profile import Profile
    from nova.thermalhydralic.sultan.remotedata import FTPData
    from nova.thermalhydralic.sultan.model import Model

try:
    FTPData().listdir()
except ftputil.error.FTPOSError:
    pytest.skip('FTPOSError: [Errno 111] Connection refused',
                allow_module_level=True)


def test_experiment():
    campaign = Campaign('CSJA13')
    assert campaign.experiment == 'CSJA13'


def test_set_experiment():
    campaign = Campaign('CSJA13')
    campaign.experiment = 'CSJA_3'
    assert campaign.experiment == 'CSJA_3'


def test_mode():
    campaign = Campaign('CSJA13', 'ac')
    assert campaign.mode == 'ac'


def test_set_mode():
    campaign = Campaign('CSJA13', 'ac')
    campaign.mode = 'dc'
    assert campaign.mode == 'dc'


def test_campaign_plan():
    campaign = Campaign('CSJA13', 'ac')
    assert campaign.plan['ac0'] == 'AC Loss Initial'


def test_phase():
    campaign = Campaign('CSJA13', 'ac')
    phase = Phase(campaign)
    assert phase.index == ['ac0', 'ac1']


def test_shot_phase_name():
    campaign = Campaign('CSJA13')
    phase = Phase(campaign, -2)
    assert phase.name == 'ac0'


def test_trial_update():
    trial = Trial('CSJA13', -1, 'ac')
    trial.campaign.mode = 'dc'
    assert trial.phase.name == 'dc1'


def test_samplenumber():
    trial = Trial('CSJA13', -1, 'ac')
    assert trial.samplenumber == 13


def test_database_update():
    sample = Sample('CSJA13')
    sample.trial.campaign.experiment = 'CSJA_3'
    assert sample.sourcedata.sultandata.database == sample.trial.database


def test_sultandata_update():
    sample = Sample('CSJA13')
    sample.shot = 3
    assert sample.filename == sample.sourcedata.sultandata.filename


def test_sourcedata_name():
    trial = Trial('CSJA13', -1, 'ac')
    sourcedata = SourceData(trial, 2)
    assert sourcedata.filename == 'CSJA13A110611'


def test_sourcedata_update():
    trial = Trial('CSJA13', -1, 'ac')
    sourcedata = SourceData(trial)
    sourcedata.shot = 2
    assert sourcedata.filename == sourcedata.sultandata.filename


def test_sampledataframe_lowpass_filter():
    trial = Trial('CSJA13', -1, 'ac')
    sourcedata = SourceData(trial, 2)
    sampledata = SampleData(sourcedata, _lowpass_filter=False)
    lowpass_array = []
    lowpass_array.append(sampledata.lowpass_filter)
    with sampledata(lowpass_filter=True):
        lowpass_array.append(sampledata.lowpass_filter)
    lowpass_array.append(sampledata.lowpass_filter)
    assert lowpass_array == [False, True, False]


def test_profile_offset():
    profile = Profile('CSJA13')
    profile.sample.shot = -11
    assert np.isclose(profile.timeseries(profile.sample.heatindex.start),
                      (0.0, 0.0)).all()


def test_model_dc_gain():
    model = Model(6, _dcgain=20.5)
    assert model.dcgain == 20.5


def test_model_dc_gain_step():
    model = Model(6, _dcgain=20.5)
    assert np.isclose(scipy.signal.step(model.lti, T=[0, 1e4])[1][-1], 20.5)


def test_phase_name_update():
    sample = Sample('CSJA13', 0, 'Left')
    sample.trial.phase.name = 0
    profile = Profile(sample)

    ac0_length = len(profile.time)
    sample.trial.phase.name = 1
    ac1_length = len(profile.time)
    assert ac0_length != ac1_length


def test_model_delay_boolean():
    model = Model([6, 3], delay=False)
    assert len(model.vector) == 3


def test_model_delay_update():
    model = Model(6, delay=False)
    model.delay = True
    assert len(model.vector) == 3


if __name__ == '__main__':
    pytest.main([__file__])
