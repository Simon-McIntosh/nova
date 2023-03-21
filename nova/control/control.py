"""Manage control factroy methods."""
from dataclasses import dataclass
from functools import cached_property

from nova.control import Circuit
from nova.frame.framedata import FrameData
from nova.frame.frameset import FrameSet, frame_factory
from nova.frame.framesetloc import ArrayLocIndexer, LocIndexer
from nova.frame.framespace import FrameSpace


@dataclass
class ControlLoc(FrameData):
    """
    Control Loc indexer.

        - cloc: Loc access to supply attributes.
        - caloc: Array access to supply attributes.

    """

    #@cached_property
    #def aloc_hash(self):
    #    """Return interger hash computed on aloc array attribute."""
    #    return HashLoc('array_hash', self.aloc, self.saloc)

    @cached_property
    def cloc(self):
        """Return fast frame array attributes."""
        return LocIndexer('loc', self.supply)

    @cached_property
    def caloc(self):
        """Return fast frame array attributes."""
        return ArrayLocIndexer('array', self.supply)


@dataclass
class Control(FrameSet, ControlLoc):
    """Manage methods for control instance initiation."""

    def __post_init__(self):
        """Create voltage source frame."""
        self.supply = FrameSpace(
            base=[], required=['R'], additional=['V', 'Is', 'R'],
            available=[], subspace=[],
            array=['V', 'Is', 'R'], delim='_', version=[])
        super().__post_init__()

    def load(self):
        """Load supply dataset from file."""
        super().load()
        self.supply.load(self.filepath, self.subgroup('supply'))
        return self

    def store(self):
        """Store supply dataset as group within netCDF file."""
        super().store()
        self.supply.store(self.filepath, self.subgroup('supply'), 'a')
        return self

    @frame_factory(Circuit)
    def circuit(self):
        """Return circuit constructor kwargs."""
        return dict(supply=self.supply)
