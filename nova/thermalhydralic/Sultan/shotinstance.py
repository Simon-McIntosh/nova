"""Manage single Sultan shot instances."""

from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Union

import pandas

from nova.thermalhydralic.sultan.testplan import TestPlan


@dataclass
class ShotInstance:
    """Manage Sultan test instance (shot)."""

    testplan: Union[TestPlan, str] = field(repr=True)
    _index: int = 0
    _side: str = 'Left'
    reload: SimpleNamespace = field(init=False, repr=False,
                                    default_factory=SimpleNamespace)

    def __post_init__(self):
        """Typecheck testplan and initialize shot instance."""
        self.reload.__init__(index=True, side=True, sultandata=True)
        if not isinstance(self.testplan, TestPlan):
            self.testplan = TestPlan(self.testplan)

    @property
    def index(self):
        """
        Shot identifier.

        Parameters
        ----------
        index : int
            Shot identifier.

        Raises
        ------
        IndexError
            Shot not set (is None) or is set out of range.

        Returns
        -------
        shot : pandas.Series
            Shot identifier.

        """
        if self.reload.index:
            self.index = self._index
        return self._index

    @index.setter
    def index(self, index):
        try:
            self._index = self.testplan.plan.index[index]
        except IndexError as index_error:
            raise IndexError(f'shot index {index} '
                             'out of bounds for testplan index '
                             f'{self.testplan.plan.index}') from index_error
        self.reload.index = False
        self.reload.sultandata = True

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
        self.reload.sultandata = True

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
