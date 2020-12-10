"""Manage Sultan testplan."""
from dataclasses import dataclass
from typing import Any

from nova.thermalhydralic.Sultan.testcampaign import TestCampaign


@dataclass
class TestPlan:
    """
    Load Sultan experiment test instance (shot).

    Parameters
    ----------
    experiment : str
        Experiment label.

    """

    _experiment: str
    _testname: Any = 0
    _mode: str = 'ac'

    def __post_init__(self):
        """Link properties."""
        self.experiment = self._experiment
        self.testname = self._testname
        self.mode = self._mode

    @property
    def experiment(self):
        """Manage experiment name."""
        return self._experiment

    @experiment.setter
    def experiment(self, experiment):
        self._experiment = experiment
        self.campaign = TestCampaign(self.experiment)

    @property
    def database(self):
        """Return database instance."""
        return self.campaign.database

    @property
    def mode(self):
        """
        Manage sultan test mode.

        Parameters
        ----------
        mode : str
            Sultan test mode.

        Raises
        ------
        IndexError
            Mode not in [ac, dc, full].

        Returns
        -------
        mode : str

        """
        return self._mode

    @mode.setter
    def mode(self, mode):
        mode = mode.lower()
        if mode not in ['cal', 'ac', 'dc', 'full']:
            raise IndexError('mode not in [cal, ac, dc, full]')
        self._mode = mode

    @property
    def testname(self):
        """
        Manage testname.

        Parameters
        ----------
        testname : str or int
            Test identifier.

        Raises
        ------
        IndexError
            testname out of range.

        Returns
        -------
        testname : str

        """
        return self._testname

    @testname.setter
    def testname(self, testname):
        if isinstance(testname, int):
            testindex = testname
            try:
                testname = self.index.index[testindex]
            except IndexError as index_error:
                raise IndexError(f'testname index {testindex} out of range\n\n'
                                 f'{self.index}') from index_error
        elif isinstance(testname, str):
            if testname not in self.index.index:
                raise IndexError(f'testname {testname} not found in '
                                 f'\n{self.index}')
        self._testname = testname

    @property
    def index(self):
        """Return testplan index, read-only."""
        campaign_index = self.campaign.index
        if self.mode == 'full':
            index = campaign_index.loc[:, 'name']
        else:
            testindex = campaign_index['mode'] == self.mode[0]
            index = campaign_index.loc[testindex, 'name']
        return index

    @property
    def plan(self):
        """Return testplan, read-only."""
        return self.campaign.metadata[self.testname]

    @property
    def shotnumber(self):
        """Return shot number for current test."""
        return len(self.plan.index)

    @property
    def note(self):
        """Return testplan notes."""
        campaign_note = self.campaign.note.set_index('index')
        note = campaign_note.loc[self.plan['File'], :]
        return note.reset_index()


if __name__ == '__main__':

    testplan = TestPlan('CSJA_3', -1)
    print(testplan.index)
    print(testplan.testname)
    print(testplan.plan['File'])
    print(testplan.note)
