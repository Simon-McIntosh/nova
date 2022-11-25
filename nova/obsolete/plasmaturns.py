from dataclasses import dataclass, field


from nova.frame.metamethod import MetaMethod
from nova.frame.dataframe import DataFrame
from nova.utilities.xpu import xp


@dataclass
class PlasmaTurns(MetaMethod):
    """Manage dependant frame energization parameters."""

    name = 'plasma_turns'

    frame: DataFrame = field(repr=False)
    required: list[str] = field(default_factory=lambda: [
        'plasma', 'nturn'], repr=False)
    require_all: bool = field(repr=False, default=True)
    additional: list[str] = field(init=False, default_factory=lambda: [])

    def __post_init__(self):
        """Extend additional with unique values extracted from match."""
        if not self.generate or self.frame.empty:
            return
        try:
            self.frame.attrs['plasmaturns'] = xp.array(
                self.frame['nturn'][self.frame['plasma']], dtype=xp.float32)
        except ValueError:
            pass

    def initialize(self):
        """Update frame selection labels."""
        if self.frame.empty:
            return
