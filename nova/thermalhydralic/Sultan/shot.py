"""Manage sultan shot."""
from dataclasses import dataclass, field, InitVar
from typing import Union
from types import SimpleNamespace

from nova.thermalhydralic.sultan.campaign import Campaign
from nova.thermalhydralic.sultan.testplan import TestPlan


@dataclass
class Shot:

    testplan: Union[TestPlan, Campaign, str] = field(repr=False)
    _index: int = 0

    def __post_init__(self):
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
        self._reload()
        if self.reload.index:
            self.index = self._index
        return self._index

    @index.setter
    def index(self, index):
        try:
            self._index = self.plan.index[index]
        except IndexError as index_error:
            raise IndexError(f'shot index {index} '
                             'out of bounds for testplan index '
                             f'{self.plan.index}') from index_error
        self.reload.index = False
        #self.reload.data = True

    @property
    def plan(self):
        """Return testplan.plan."""
        return self.testplan.plan

if __name__ == '__main__':

    shot = Shot('CSJA13')