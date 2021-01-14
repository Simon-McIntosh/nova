"""Manage sultan test."""
from dataclasses import dataclass, field, InitVar
from typing import Union
from types import SimpleNamespace

from nova.thermalhydralic.sultan.campaign import Campaign
from nova.thermalhydralic.sultan.trial import Trial


@dataclass
class TestPlan:
    """Manage sultan test metadata."""

    campaign: Union[Campaign, str]
    _name: Union[int, str] = field(default=0, repr=False)
    _mode: str = field(default='ac', repr=False)
    trial: Trial = field(init=False)
    reload: SimpleNamespace = field(init=False, repr=False,
                                    default_factory=SimpleNamespace)

    def __post_init__(self):
        """Init reload."""
        self.reload.__init__(index=True)
        if not isinstance(self.campaign, Campaign):
            self.campaign = Campaign(self.campaign, self._mode)
        else:
            self.campaign.mode = self._mode
        self.trial = Trial(self.campaign, self._name)
        del self._name
        del self._mode

    @property
    def name(self):
        """Return trial name."""
        return self.trial.name

    @property
    def mode(self):
        """Return campaign mode."""
        return self.campaign.mode

    def _reload(self):
        if self.campaign.reload.trial:
            self.reload.index = True
            self.reload.name = True
            self.campaign.reload.trial = False

    @property
    def plan(self):
        """Return testplan, read-only."""
        return self.campaign.metadata[self.name]

    @property
    def database(self):
        """Return database instance."""
        return self.campaign.database

    @property
    def shotnumber(self):
        """Return shot number for current test."""
        return len(self.plan.index)

    @property
    def notes(self):
        """Return testplan notes, read-only."""
        campaign_notes = self.campaign.note.set_index('index')
        trial_notes = campaign_notes.loc[self.plan['File'], :]
        return trial_notes.reset_index()


if __name__ == '__main__':

    testplan = TestPlan('CSJA13', -1, 'ac')
