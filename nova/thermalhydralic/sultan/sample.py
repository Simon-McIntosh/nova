"""Manage sultan shot."""

from dataclasses import dataclass, field, InitVar
from typing import Union

import pandas

from nova.thermalhydralic.sultan.campaign import Campaign
from nova.thermalhydralic.sultan.trial import Trial
from nova.thermalhydralic.sultan.sourcedata import SourceData
from nova.thermalhydralic.sultan.sampledata import SampleData


@dataclass
class Sample:
    """Manage sultan shot instance."""

    trial: Union[Trial, Campaign, str] = field(repr=False)
    _shot: InitVar[int] = field(default=0)
    _side: InitVar[str] = field(default="Left")
    sourcedata: SourceData = field(init=False, repr=True)
    sampledata: SampleData = field(init=False, repr=False)

    def __post_init__(self, _shot, _side):
        """Initialize sample instance."""
        if not isinstance(self.trial, Trial):
            self.trial = Trial(self.trial)
        self.sourcedata = SourceData(self.trial, _shot, _side)
        self.sampledata = SampleData(self.sourcedata)

    @property
    def shot(self):
        """Return shot index."""
        return self.sourcedata.shot

    @shot.setter
    def shot(self, shot):
        self.sourcedata.shot = shot

    @property
    def side(self):
        """Return sample side."""
        return self.sourcedata.side

    @side.setter
    def side(self, side):
        self.sourcedata.side = side

    @property
    def plan(self):
        """Return trial.plan."""
        return self.trial.plan

    @property
    def sultandata(self):
        """Return sultan source, update sample filename if required."""
        return self.sourcedata.sultandata

    @property
    def rawdata(self):
        """Return rawdata."""
        return self.sampledata.raw

    @property
    def lowpassdata(self):
        """Return lowpassdata."""
        return self.sampledata.lowpass

    @property
    def heatindex(self):
        """Return heatindex."""
        return self.sampledata.heatindex

    @property
    def metadata(self):
        """Return trial metadata, read-only."""
        metadata = pandas.Series(self.trial.plan.iloc[self.shot, :])
        metadata["note"] = self.note
        return metadata

    @property
    def note(self):
        """Return sample note, read-only."""
        return self.trial.notes.loc[self.shot, "note"]

    @property
    def filename(self):
        """Return sample filename."""
        return self.sourcedata.filename

    @property
    def testname(self):
        """Return testname, read-only."""
        return self.trial.testname

    @property
    def frequency(self):
        """Return sample frequency."""
        return self.sourcedata.frequency

    @property
    def massflow(self):
        """Return time average shot massflow."""
        return self.sultandata.data[f"dm/dt {self.side}"].mean()

    @property
    def field(self):
        """Return background field."""
        return self.trial.plan.at[self.shot, ("Be", "T")]

    @property
    def current(self):
        """Return transport current."""
        return self.trial.plan.at[self.shot, ("Isample", "kA")]

    @property
    def samplename(self):
        """Return sample name."""
        samplename = f"{self.trial.campaign.experiment} "
        samplename += f"{self.trial.phase.name} {self.shot} {self.side}"
        return samplename

    @property
    def samplenumber(self):
        """Return trial sample number."""
        return self.trial.samplenumber

    def sequence(self):
        """Return filename generator."""
        for shot in range(self.samplenumber):
            self.shot = shot
            yield shot

    @property
    def label(self):
        """Return sample label."""
        label = f"{self.samplename} "
        label += "("
        label += rf"$\dot{{m}}$={self.massflow:1.1f}g/s, "
        label += rf"$f$={self.frequency:1.1f}Hz, "
        label += rf"$B$={self.field:1.0f}T, "
        label += rf"$I$={self.current:1.0f}kA"
        label += ")"
        return label


if __name__ == "__main__":
    sample = Sample("CSJA13")
