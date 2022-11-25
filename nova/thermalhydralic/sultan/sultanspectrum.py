
from dataclasses import dataclass, InitVar
from typing import Union
import sys

import pandas
import pygmo
import numpy as np
import scipy
import nlopt

from nova.thermalhydralic.sultan.trial import Trial
from nova.thermalhydralic.sultan.sample import Sample
from nova.thermalhydralic.sultan.fluidprofile import FluidProfile
from nova.thermalhydralic.sultan.sultanio import SultanIO
from nova.thermalhydralic.sultan.database import DataBase
from nova.plot import plt
from nova.utilities.time import clock


@dataclass
class SultanSpectrum(SultanIO):
    """Manage fluid response data."""

    experiment: str
    phase: Union[int, str]
    side: str
    reload: InitVar[bool] = False

    def __post_init__(self, reload):
        """Create fluidprofile instance."""
        self.database = DataBase(self.experiment)
        self.plan, self.phase = self.load_plan()
        if reload:
            self.data = self.read_data()
        else:
            self.data = self.load_data()

    def load_plan(self):
        """Return trial plan."""
        trial = Trial(self.experiment, self.phase)
        plan = trial.plan.droplevel(1, axis=1)
        return plan.drop(columns='File'), trial.phase.name

    @property
    def binaryfilepath(self):
        """Return full path of binary datafile."""
        return self.database.binary_filepath('fluidresponse.h5')

    @property
    def filepath(self):
        """Return full path of source datafile, read-only."""
        return self.database.datafile(self.filename)

    @property
    def filename(self):
        """Return datafile filename."""
        return f'{self.experiment}_{self.phase}_{self.side}'

    def _read_data(self):
        """Refit fluid model to waveform data."""
        trial = Trial(self.experiment, self.phase)
        sample = Sample(trial, 0, self.side)
        fluidprofile = FluidProfile(sample, [4], 0, verbose=False)
        coefficents = pandas.DataFrame(
            index=range(sample.samplenumber),
            columns=fluidprofile.shot_coefficents.index, dtype='float')
        tick = clock(sample.samplenumber,
                     header=f'Calculating {self.filename} fluid response.')
        for shot in sample.sequence():
            fluidprofile.read_data()
            coefficents.iloc[shot] = fluidprofile.shot_coefficents
            tick.tock()
        return coefficents.astype('float')

    def plot(self):
        """Plot spectrum."""
        plt.plot(self.data.frequency, self.data.steadystate, 'o')


if __name__ == '__main__':

    spectrum = SultanSpectrum('CSJA_5', 0, 'Left')
    spectrum.plot()
