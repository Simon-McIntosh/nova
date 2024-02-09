"""Manage sultan test."""

from dataclasses import dataclass, field
from typing import Union

from nova.thermalhydralic.sultan.campaign import Campaign
from nova.thermalhydralic.sultan.phase import Phase


@dataclass
class Trial:
    """Manage sultan trial."""

    campaign: Union[Campaign, str]
    _name: Union[int, str] = field(default=0, repr=False)
    _mode: str = field(default="ac", repr=False)
    phase: Phase = field(init=False)

    def __post_init__(self):
        """Init campaign and phase instances."""
        if not isinstance(self.campaign, Campaign):
            self.campaign = Campaign(self.campaign, self._mode)
        else:
            self.campaign.mode = self._mode
        self.phase = Phase(self.campaign, self._name)
        del self._name
        del self._mode

    @property
    def name(self):
        """Return test phase name."""
        return self.phase.name

    @property
    def mode(self):
        """Return campaign mode."""
        return self.campaign.mode

    @property
    def testname(self):
        """Return test name."""
        return f"{self.campaign.experiment}_{self.name}"

    @property
    def plan(self):
        """Return testplan, read-only."""
        return self.campaign.metadata[self.name]

    def filename(self, shot):
        """Return shot filename."""
        return self.plan.File[shot]

    def frequency(self, shot):
        """Return shot frequency."""
        return self.plan.at[shot, ("frequency", "Hz")]

    @property
    def database(self):
        """Return database instance."""
        return self.campaign.database

    @property
    def samplenumber(self):
        """Return sample number for current trial."""
        return len(self.plan.index)

    @property
    def notes(self):
        """Return testplan notes, read-only."""
        campaign_notes = self.campaign.note.set_index("index")
        phase_notes = campaign_notes.loc[self.plan["File"], :]
        return phase_notes.reset_index()


if __name__ == "__main__":
    trial = Trial("CSJA13", -1, "ac")
