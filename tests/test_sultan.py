import pytest

from nova.thermalhydralic.sultan.campaign import Campaign
from nova.thermalhydralic.sultan.phase import Phase
from nova.thermalhydralic.sultan.trial import Trial
from nova.thermalhydralic.sultan.sample import Sample
from nova.thermalhydralic.sultan.sourcedata import SourceData
from nova.thermalhydralic.sultan.sampledataframe import SampleDataFrame


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
    assert sample.source.sultan.database == sample.trial.database


def test_sultandata_update():
    sample = Sample('CSJA13')
    sample.shot = 3
    assert sample.filename == sample.source.sultan.filename


def test_source_name():
    trial = Trial('CSJA13', -1, 'ac')
    source = SourceData(trial, 2)
    assert source.filename == 'CSJA13A110610'


def test_source_update():
    trial = Trial('CSJA13', -1, 'ac')
    source = SourceData(trial)
    source.shot = 2
    assert source.filename == source.sultan.filename


def test_sampledataframe_lowpass_filter():
    trial = Trial('CSJA13', -1, 'ac')
    source = SourceData(trial, 2)
    dataframe = SampleDataFrame(source, False)
    lowpass_array = []
    lowpass_array.append(dataframe.lowpass_filter)
    with dataframe(True):
        lowpass_array.append(dataframe.lowpass_filter)
    lowpass_array.append(dataframe.lowpass_filter)
    assert lowpass_array == [False, True, False]


if __name__ == '__main__':
    pytest.main([__file__])
