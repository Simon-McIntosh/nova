
import pandas

from nova.thermalhydralic.sultan.trial import Trial
from nova.thermalhydralic.sultan.sample import Sample
from nova.thermalhydralic.sultan.waveform import WaveForm
from nova.thermalhydralic.sultan.model import Model
from nova.thermalhydralic.sultan.fluid import Fluid, FluidSeries
from nova.thermalhydralic.sultan.sultanio import SultanIO
from nova.utilities.time import clock
import matplotlib.pyplot as plt


class Field(SultanIO):

    def __init__(self, experiment, name, side, cooldown_threshold=0.9):
        self.trial = Trial(experiment, name)
        self.sample = Sample(self.trial, _side=side)
        self.waveform = WaveForm(self.sample, cooldown_threshold, pulse=True)
        self.fluid = Fluid(self.waveform.data, Model(5))
        self.response = self.load_data()

    @property
    def binaryfilepath(self):
        """Return full path of binary datafile."""
        return self.trial.database.binary_filepath('fluid.h5')

    @property
    def filename(self):
        """Manage datafile filename."""
        return f'{self.trial.testname}_{self.sample.side}'

    def _read_data(self):
        """
        Return flow response dataframe.

        Returns
        -------
        data : pandas.DataFrame
            Shot data.

        """
        data = pandas.DataFrame(index=range(self.sample.samplenumber),
                                columns=FluidSeries().index)
        tick = clock(self.sample.samplenumber,
                     header=f'loading {self.filename} fluid response')
        for shot in self.sample.sequence():
            self.fluid.__post_init__(self.waveform.data)
            self.fluid.optimize()
            data.iloc[shot] = self.fluid.coefficents
            tick.tock()
        data = pandas.concat(
            [self.trial.plan.droplevel(1, axis=1), data], axis=1)
        data = data.loc[:, ~data.columns.duplicated()]
        print(data)
        print(data.dtypes)
        data.drop(columns=['File', 'Ipulse'], inplace=True)
        return data.astype(float)


if __name__ == '__main__':

    field = Field('CSJA_7', -1, 'Right')
    plt.loglog(field.response.frequency, field.response.dcgain)

    field = Field('CSJA_8', -1, 'Right')
    plt.loglog(field.response.frequency, field.response.dcgain)