
from dataclasses import dataclass
from typing import Union

import pandas

from nova.thermalhydralic.sultan.trial import Trial
from nova.thermalhydralic.sultan.sample import Sample
from nova.thermalhydralic.sultan.fluidprofile import FluidProfile
from nova.thermalhydralic.sultan.sultanio import SultanIO
from nova.thermalhydralic.sultan.remotedata import FTPData
from nova.thermalhydralic.sultan.campaign import Campaign
from nova.utilities.pyplot import plt
from nova.utilities.time import clock


@dataclass
class FluidResponse(SultanIO):
    """Manage fluid response data."""

    experiment: str
    phase: Union[int, str]
    side: str
    order: Union[int, list[int]] = 7
    threshold: float = 0.9

    def __post_init__(self):
        """Create fluidprofile instance."""
        trial = Trial(self.experiment, self.phase)
        sample = Sample(trial, _side=self.side)
        self.fluidprofile = FluidProfile(sample, self.order, self.threshold,
                                         verbose=False)
        self.coefficents = self.load_data()
        self._load_plan()

    def _load_plan(self):
        self.plan = self.sample.trial.plan.droplevel(1, axis=1)
        self.plan.drop(columns='File', inplace=True)

    @property
    def profile(self):
        """Return profile instance."""
        return self.fluidprofile.profile

    @property
    def sample(self):
        """Return sample instance."""
        return self.profile.sample

    @property
    def database(self):
        """Return database instance."""
        return self.sample.trial.database

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
        filename = f'{self.sample.trial.campaign.experiment}'
        filename += f'_{self.sample.trial.name}_{self.sample.side}'
        return filename

    @property
    def shot_coefficents(self):
        """Return profile, model and fit shot coefficents."""
        return pandas.concat([self.profile.coefficents,
                              self.fluidprofile.coefficents])

    def _read_data(self):
        """Refit fluid model to waveform data."""
        coefficents = pandas.DataFrame(index=range(self.sample.samplenumber),
                                       columns=self.shot_coefficents.index,
                                       dtype='float')
        tick = clock(self.sample.samplenumber,
                     header=f'Calculating {self.filename} fluid response.')
        for shot in self.sample.sequence():
            coefficents.iloc[shot] = self.shot_coefficents
            tick.tock()
        return coefficents.astype('float')

    def plot(self, Be):
        """Plot frequency response."""
        index = self.plan['Be'] == Be
        plt.loglog(2*self.coefficents['frequency'][index],
                   self.coefficents['steadystate'][index])


if __name__ == '__main__':

    '''
    ftp = FTPData('')
    for experiment in ftp.listdir(select='CSJA'):
        print(experiment)
        for phase in Campaign(experiment).index:
            for side in ['Left', 'Right']:
                fluidresponse = FluidResponse(experiment, phase, side)
    '''

    response = FluidResponse('CSJA_8', 0, 'Left')

    response.plot(2)
    response.plot(9)
    #fluidresponse.sample.shot = 5
    #fluidresponse.fluidprofile.plot()

    """
    fluidresponse = FluidResponse('CFETR', 0, 'Left')
    index = fluidresponse.sample.trial.plan[('Be', 'T')] == 2
    coefficents = fluidresponse.load_data()
    plt.loglog(2*coefficents['frequency'][index],
               coefficents['steadystate'][index])

    fluidresponse = FluidResponse('CSJA_8', 0, 'Left')
    index = fluidresponse.sample.trial.plan[('Be', 'T')] == 2
    coefficents = fluidresponse.load_data()
    plt.loglog(2*coefficents['frequency'][index],
               coefficents['steadystate'][index])
    """
