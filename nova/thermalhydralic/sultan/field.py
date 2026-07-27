"""Fit every shot in a trial to build a field-response table."""

import pandas

from nova.thermalhydralic.sultan.trial import Trial
from nova.thermalhydralic.sultan.sample import Sample
from nova.thermalhydralic.sultan.waveform import WaveForm
from nova.thermalhydralic.sultan.model import Model
from nova.thermalhydralic.sultan.fitfluid import COEFFICENT, FitFluid
from nova.thermalhydralic.plotimport import clock
from nova.utilities.pandasdata import PandasHDF


class Field(PandasHDF):
    """Tabulate fit coefficients across a trial's shot sequence."""

    def __init__(self, experiment, name, side, cooldown_threshold=0.9):
        """Build the shot sequence and load or refit the response table."""
        self.trial = Trial(experiment, name)
        self.sample = Sample(self.trial, _side=side)
        self.waveform = WaveForm(self.sample, cooldown_threshold, _pulse=True)
        self.fluid = FitFluid(Model(5))
        self.response = self.load_data()

    @property
    def binaryfilepath(self):
        """Return full path of binary datafile."""
        return self.trial.database.binary_filepath("fluid.h5")

    @property
    def filename(self):
        """Manage datafile filename."""
        return f"{self.trial.testname}_{self.sample.side}"

    def _read_data(self):
        """
        Return flow response dataframe.

        Returns
        -------
        data : pandas.DataFrame
            Shot data.

        """
        data = pandas.DataFrame(
            index=range(self.sample.samplenumber), columns=list(COEFFICENT)
        )
        tick = clock(
            self.sample.samplenumber, header=f"loading {self.filename} fluid response"
        )
        for shot in self.sample.sequence():
            # advancing the shot invalidates the waveform, so each fit reads the
            # freshly extracted timeseries rather than the one built at init
            self.fluid.optimize(self.waveform.data)
            data.iloc[shot] = self.fluid.coefficents
            tick.tock()
        data = pandas.concat([self.trial.plan.droplevel(1, axis=1), data], axis=1)
        data = data.loc[:, ~data.columns.duplicated()]
        data.drop(columns=["File", "Ipulse"], inplace=True)
        return data.astype(float)
