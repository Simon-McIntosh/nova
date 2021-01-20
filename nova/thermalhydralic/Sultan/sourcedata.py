"""Manage sultan shot."""
import re
from dataclasses import dataclass, field
from types import SimpleNamespace

import numpy as np

from nova.thermalhydralic.sultan.trial import Trial
from nova.thermalhydralic.sultan.sultandata import SultanData


@dataclass
class SourceData:
    """Manage data file parameters."""

    trial: Trial = field(repr=False)
    _shot: int = 0
    _side: str = 'Left'
    sultandata: SultanData = field(init=False, repr=False)
    reload: SimpleNamespace = field(init=False, repr=False,
                                    default_factory=SimpleNamespace)

    def __post_init__(self):
        """Typecheck trial and initialize shot instance."""
        self.reload.__init__(shot=True, side=True, sampledata=True)
        self.sultandata = SultanData(self.trial.database)
        self.sultandata.filename = self.filename

    @property
    def shot(self):
        """
        Sample identifier.

        Parameters
        ----------
        shot : int
            Sample identifier.

        Raises
        ------
        IndexError
            Shot index not set (is None) or is set out of range.

        Returns
        -------
        shot : int
            Sample identifier.

        """
        if self.reload.shot:
            self.shot = self._shot
        return self._shot

    @shot.setter
    def shot(self, shot):
        try:
            self._shot = self.trial.plan.index[shot]
        except IndexError as index_error:
            raise IndexError(f'shot index {shot} '
                             'out of bounds for trial.plan.index '
                             f'{self.trial.plan.index}') from index_error
        self.reload.shot = False
        self.sultandata.filename = self.filename
        self.reload.sampledata = True

    @property
    def side(self):
        """
        Manage side property. reload raw and lowpass data if changed.

        Parameters
        ----------
        side : str
            Side of Sultan experement ['Left', 'Right'].

        Returns
        -------
        side : str

        """
        if self.reload.side:
            self.side = self._side
        return self._side

    @side.setter
    def side(self, side):
        side = side.capitalize()
        if side not in ['Left', 'Right']:
            raise IndexError(f'side {side} not in [Left, Right]')
        self._side = side
        self.reload.side = False
        self.reload.sampledata = True

    @property
    def filename(self):
        """Return data filename."""
        return self.trial.filename(self.shot)

    @property
    def data(self):
        """Return sultan data."""
        return self.sultandata.data

    @property
    def frequency(self):
        """Return sample frequency, Hz."""
        return self.trial.frequency(self.shot)

    @property
    def current_label(self):
        """Return sample excitation current string."""
        print(self.trial.plan)
        return self.trial.plan.at[self.shot, ('Ipulse', 'A')]

    @property
    def excitation_field(self):
        """
        Return amplitude of excitation field.

        Parameters
        ----------
        Ipulse : str
            Sultan Ipulse field.

        Returns
        -------
        excitation_field : float
            External excitation field.

        """
        try:
            current = float(re.findall(r'\d+', self.current_label)[0])
        except TypeError:
            current = 230
        excitation_field = current * 0.2/230  # excitation field amplitude
        return excitation_field

    @property
    def excitation_field_rate(self):
        """Return amplitude of exciation field rate of change."""
        return 2*np.pi*self.frequency*self.excitation_field


if __name__ == '__main__':
    trial = Trial('CSJA13', -1, 'ac')
    sourcedata = SourceData(trial, 2)
