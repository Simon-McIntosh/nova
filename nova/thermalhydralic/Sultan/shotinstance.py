"""Manage single Sultan shot instances."""

from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Union

import pandas
import numpy as np

from nova.thermalhydralic.sultan.testplan import TestPlan
from nova.thermalhydralic.sultan.sultandata import SultanData
from nova.utilities.pyplot import plt


@dataclass
class ShotInstance:
    """Manage Sultan test instance (shot)."""

    testplan: Union[TestPlan, str] = field(repr=True)
    _side: str = 'Left'
    sultan: SultanData = field(init=False)
    shot: ShotData = field(init=False)
    reload: SimpleNamespace = field(init=False, repr=False,
                                    default_factory=SimpleNamespace)

    def __post_init__(self):
        """Typecheck testplan and initialize shot instance."""
        self.reload.__init__(index=True, side=True, sultandata=True,
                             shotdata=True)
        if not isinstance(self.testplan, TestPlan):
            self.testplan = TestPlan(self.testplan)
        self.sultan = SultanData(self.database)

    @property
    def sultandata(self):
        """Return sultan datafile, update shot filename if required."""
        if self.reload.sultandata:
            self.sultan.filename = self.filename
            self.reload.sultandata = False
        return self.sultan.data

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
        self.reload.shotdata = True

    @property
    def metadata(self):
        """Return shot metadata, read-only."""
        metadata = pandas.Series(self.testplan.plan.iloc[self.index, :])
        metadata['note'] = self.note
        return metadata

    @property
    def note(self):
        """Return shot note, read-only."""
        return self.testplan.note.loc[self.index, 'note']

    @property
    def frequency(self):
        """Return shot frequency."""
        return self.testplan.plan.at[self.index, ('frequency', 'Hz')]

    @property
    def current_label(self):
        """Return shot excitation current string."""
        return self.testplan.plan.at[self.index, ('Ipulse', 'A')]

    @property
    def filename(self):
        """Return shot filename."""
        return self.metadata.loc['File'][0]

    @property
    def experiment(self):
        """Return experiment, read-only."""
        return self.testplan.experiment

    @property
    def testname(self):
        """Return testname, read-only."""
        return self.testplan.testname

    @property
    def shotname(self):
        """Return shotname."""
        return f'{self.experiment}_{self.testname}_{self.side}_{self.index}'

    def sequence(self):
        """Return filename generator."""
        for i in range(self.testplan.shotnumber):
            self.index = i
            yield self.filename

    @property
    def database(self):
        """Return database instance."""
        return self.testplan.database


if __name__ == '__main__':

    shot = ShotInstance('CSJA_3')
