"""Manage Sultan testplan."""
from dataclasses import dataclass, field
from typing import Optional

from nova.thermalhydralic.Sultan.testcampaign import TestCampaign


@dataclass
class Reload:
    """Reload datastructure for testplan."""

    experiment: bool = True
    testname: bool = True
    mode: bool = True
    campaign: bool = True


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
    _testname: Optional[str] = field(default=0)
    _mode: str = 'ac'
    reload: Reload = field(init=False, default=Reload(), repr=False)

    def __post_init__(self):
        """Init test campaign."""
        self._campaign = TestCampaign(self.experiment)

    @property
    def experiment(self):
        """Manage experiment name."""
        if self.reload.experiment:
            self.experiment = self._experiment
        return self._experiment

    @experiment.setter
    def experiment(self, experiment):
        self._experiment = experiment
        self.reload.campaign = True
        self.reload.experiment = False

    @property
    def campaign(self):
        """Return campaign data, read-only."""
        if self.reload.campaign:
            self._campaign.experiment = self.experiment
            self.reload.campaign = False
        return self._campaign

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
        if self.reload.mode:
            self.mode = self._mode
        return self._mode

    @mode.setter
    def mode(self, mode):
        mode = mode.lower()
        if mode not in ['cal', 'ac', 'dc', 'full']:
            raise IndexError('mode not in [cal, ac, dc, full]')
        self._mode = mode
        self.reload.mode = False

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
        if self.reload.testname:
            self.testname = self._testname
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
        self.reload.testname = False

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
