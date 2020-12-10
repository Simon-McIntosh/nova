"""Manage single Sultan shot instances."""

from dataclasses import dataclass, field

from nova.thermalhydralic.Sultan.testplan import TestPlan


@dataclass
class SultanShot:
    """Manage Sultan test instance (shot)."""

    testplan: TestPlan = field(repr=False)
    _shotindex: int = 0

    def __post_init__(self):
        """Typecheck testplan and initialize shot instance."""
        if isinstance(self.testplan, str):
            self.testplan = TestPlan(self.testplan)
        self.shotindex = self._shotindex

    @property
    def shotindex(self):
        """
        Shot identifier.

        Parameters
        ----------
        shotindex : int
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
        return self._shotindex

    @shotindex.setter
    def shotindex(self, shotindex):
        try:
            self._shotindex = self.testplan.plan.index[shotindex]
        except IndexError as index_error:
            raise IndexError(f'shot index {shotindex} '
                             'out of bounds for testplan index '
                             f'{self.testplan.plan.index}') from index_error

    @property
    def metadata(self):
        """Return shot metadata, read-only."""
        metadata = self.testplan.plan.iloc[self.shotindex, :]
        metadata.loc['note'] = self.note
        return metadata

    @property
    def note(self):
        """Return shot note, read-only."""
        return self.testplan.note.loc[self.shotindex, 'note']

    @property
    def filename(self):
        """Return shot filename."""
        return self.metadata.loc['File'][0]

    def sequence(self):
        """Return filename generator."""
        for i in range(self.testplan.shotnumber):
            self.shotindex = i
            yield self.filename

    @property
    def database(self):
        """Return database instance."""
        return self.testplan.database


if __name__ == '__main__':

    shot = SultanShot('CSJA_3')
