"""Manage sultan phase."""
from dataclasses import dataclass, field
from typing import Union
from types import SimpleNamespace

from nova.thermalhydralic.sultan.campaign import Campaign


@dataclass
class Phase:
    """Manage sultan test phase."""

    campaign: Campaign = field(repr=False)
    _name: Union[str, int] = 0
    _nameindex: int = field(init=False, default=None, repr=None)
    reload: SimpleNamespace = field(
        init=False, repr=False, default_factory=SimpleNamespace
    )

    def __post_init__(self):
        """Init reload."""
        self.reload.__init__(index=True, name=True, sourcedata=True)
        self.name = self._name

    def propagate_reload(self):
        """Propagate reload flags."""
        if self.campaign.reload.phase:
            self.reload.index = True
            self.reload.name = True
            self.campaign.reload.phase = False

    @property
    def index(self):
        """Manage phase index."""
        self.propagate_reload()
        if self.reload.index:
            self.index = self.campaign.index
        return self._index

    @index.setter
    def index(self, index):
        self._index = index
        self.reload.index = False
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
        self.propagate_reload()
        if self.reload.name:
            if self._nameindex is not None:
                self.name = self._nameindex
            else:
                self.name = self._name
        return self._name

    @name.setter
    def name(self, name):
        self._nameindex = name  # store name index (int or str)
        if isinstance(name, int):
            try:
                name = self.index[name]
            except IndexError as index_error:
                raise IndexError(
                    f"name index {name} " "out of range\n\n" f"{self.index}"
                ) from index_error
        elif isinstance(name, str):
            if name not in self.index:
                raise IndexError(f"name {name} not found in " f"\n{self.index}")
        self._name = name
        self.reload.name = False
        self.reload.sourcedata = True


if __name__ == "__main__":
    campaign = Campaign("CSJA13")
    phase = Phase(campaign)
