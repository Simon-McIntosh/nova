"""Manage Sultan testplan."""
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Optional, Union

from nova.thermalhydralic.sultan.campaign import Campaign


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
    _testname: Optional[Union[str, int]] = field(default=0)
    _testmode: Optional[str] = field(default='ac')
    _testnameindex: int = field(init=False, default=None)
    reload: SimpleNamespace = field(
        init=False, repr=False,
        default=SimpleNamespace(experiment=True, testname=True, testmode=True,
                                campaign=True))

    def __post_init__(self):
        """Init test campaign."""
        self._campaign = Campaign(self.experiment)

    @property
    def experiment(self):
        """Manage experiment name."""
        if self.reload.experiment:
            self.experiment = self._experiment
        return self._experiment

    @experiment.setter
    def experiment(self, experiment):
        self._experiment = experiment
        if self._testnameindex is not None:
            self._testname = self._testnameindex
            self.reload.testname = True
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
    def testmode(self):
        """
        Manage sultan test mode.

        Parameters
        ----------
        testmode : str
            Sultan test mode.

        Raises
        ------
        IndexError
            Mode not in [ac, dc, full].

        Returns
        -------
        testmode : str

        """
        if self.reload.testmode:
            self.testmode = self._testmode
        return self._testmode

    @testmode.setter
    def testmode(self, testmode):
        testmode = testmode.lower()
        if testmode not in ['cal', 'ac', 'dc', 'full']:
            raise IndexError('testmode not in [cal, ac, dc, full]')
        self._testmode = testmode
        self.reload.testmode = False

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
        self._testnameindex = testname  # store testname index (int or str)
        if isinstance(testname, int):
            try:
                testname = self.testindex.index[self._testnameindex]
            except IndexError as index_error:
                raise IndexError(f'testname index {self._testnameindex} '
                                 'out of range\n\n'
                                 f'{self.testindex}') from index_error
        elif isinstance(testname, str):
            if testname not in self.testindex.index:
                raise IndexError(f'testname {testname} not found in '
                                 f'\n{self.testindex}')
        self._testname = testname
        self.reload.testname = False

    @property
    def testindex(self):
        """Return testplan index, read-only."""
        campaign_index = self.campaign.index
        if self.testmode == 'full':
            index = campaign_index.loc[:, 'name']
        else:
            testindex = campaign_index['testmode'] == self.testmode[0]
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
    print(testplan.testindex)
    print(testplan.testname)
    print(testplan.plan['File'])
    print(testplan.note)
    print(testplan.testmode)
