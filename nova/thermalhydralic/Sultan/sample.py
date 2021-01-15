"""Manage sultan shot."""
from dataclasses import dataclass, field, InitVar
from typing import Union

import pandas

from nova.thermalhydralic.sultan.campaign import Campaign
from nova.thermalhydralic.sultan.trial import Trial
from nova.thermalhydralic.sultan.sourcedata import SourceData
from nova.thermalhydralic.sultan.sampledataframe import SampleDataFrame


@dataclass
class Sample:
    """Manage sultan shot instance."""

    trial: Union[Trial, Campaign, str] = field(repr=False)
    _shot: InitVar[int] = field(default=0, repr=False)
    _side: InitVar[str] = field(default='Left', repr=False)
    source: SourceData = field(init=False)
    dataframe: SampleDataFrame = field(init=False)

    def __post_init__(self, _shot, _side):
        """Init sample instance."""
        if not isinstance(self.trial, Trial):
            self.trial = Trial(self.trial)
        self.source = SourceData(self.trial, _shot, _side)
        self.dataframe = SampleDataFrame(self.source)

    @property
    def shot(self):
        """Return shot index."""
        return self.source.shot

    @shot.setter
    def shot(self, shot):
        self.source.shot = shot

    @property
    def side(self):
        """Return sample side."""
        return self.source.side

    @side.setter
    def side(self, side):
        self.source.side = side

    @property
    def plan(self):
        """Return trial.plan."""
        return self.trial.plan

    @property
    def sultandata(self):
        """Return sultan source, update sample filename if required."""
        return self.source.sultandata

    @property
    def rawdata(self):
        """Return rawdata."""
        return self.dataframe.raw

    @property
    def lowpassdata(self):
        """Return lowpassdata."""
        return self.dataframe.lowpass

    @property
    def heatindex(self):
        """Return heatindex."""
        return self.dataframe.heatindex

    @property
    def metadata(self):
        """Return trial metadata, read-only."""
        metadata = pandas.Series(self.trial.plan.iloc[self.shot, :])
        metadata['note'] = self.note
        return metadata

    @property
    def note(self):
        """Return sample note, read-only."""
        return self.trial.note.loc[self.shot, 'note']

    @property
    def filename(self):
        """Return sample filename."""
        return self.source.filename

    @property
    def experiment(self):
        """Return experiment, read-only."""
        return self.trial.experiment

    @property
    def testname(self):
        """Return testname, read-only."""
        return self.trial.testname

    @property
    def name(self):
        """Return sample name."""
        return f'{self.experiment}_{self.testname}_{self.side}_{self.shot}'

    def sequence(self):
        """Return filename generator."""
        for i in range(self.trial.samplenumber):
            self.shot = i
            yield self.filename


if __name__ == '__main__':

    sample = Sample('CSJA13')
