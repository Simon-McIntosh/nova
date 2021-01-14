import pytest

from nova.thermalhydralic.sultan.campaign import Campaign
from nova.thermalhydralic.sultan.trial import Trial


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


def test_trial():
    campaign = Campaign('CSJA13')
    trial = Trial(campaign)
    assert trial.plan == ['ac0', 'ac1']


def test_shot_trial_name():
    trial = Trial(['ac0', 'ac1', 'ac2'], -3)
    assert trial.name == 'ac0'


if __name__ == '__main__':
    pytest.main([__file__])
