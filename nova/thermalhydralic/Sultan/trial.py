"""Manage sultan trial."""
from dataclasses import dataclass, field
from typing import Union
from types import SimpleNamespace

from nova.thermalhydralic.sultan.campaign import Campaign


@dataclass
class Trial:
    """Manage sultan trial."""

    campaign: Campaign = field(repr=False)
    _name: Union[str, int] = 0
    index: int = field(init=False, default=None)
    reload: SimpleNamespace = field(init=False, repr=False,
                                    default_factory=SimpleNamespace)

    def __post_init__(self):
        """Init reload."""
        self.reload.__init__(plan=True, name=True)
        self.name = self._name

    def _reload(self):
        if self.campaign.reload.trial:
            self.reload.plan = True
            self.reload.name = True

    @property
    def plan(self):
        """Manage trial plan."""
        self._reload()
        if self.reload.plan:
            self.plan = self.campaign.trial_plan
        return self._plan

    @plan.setter
    def plan(self, plan):
        self._plan = plan
        self.reload.plan = False
        self.reload.name = True

    @property
    def name(self):
        """
        Manage name.

        Parameters
        ----------
        name : str or int
            Test identifier.

        Raises
        ------
        IndexError
            name out of range.

        Returns
        -------
        name : str

        """
        self._reload()
        if self.reload.name:
            if self.index is not None:
                self.name = self.index
            else:
                self.name = self._name
        return self._name

    @name.setter
    def name(self, name):
        self.index = name  # store name index (int or str)
        if isinstance(name, int):
            try:
                name = self.plan[name]
            except IndexError as index_error:
                raise IndexError(f'name index {name} '
                                 'out of range\n\n'
                                 f'{self.plan}') from index_error
        elif isinstance(name, str):
            if name not in self.plan:
                raise IndexError(f'name {name} not found in '
                                 f'\n{self.plan}')
        self._name = name
        self.reload.name = False
        #self.reload.response = True


if __name__ == '__main__':

    campaign = Campaign('CSJA13')
    trial = Trial(campaign)
