"""Baseclass for dataframe geometroy objects."""

from abc import abstractmethod
from dataclasses import dataclass


@dataclass
class GeoFrame:
    """Geometry object abstract base class."""

    def __post_init__(self):
        """Propogate for cooperative inheritance."""
        if hasattr(super(), "__post_init__"):
            super().__post_init__()

    def __str__(self):
        """Return name if set."""
        if hasattr(self, "name"):
            return self.name
        return None

    @abstractmethod
    def __eq__(self, other):
        """Evaluate equality with other geometry object."""

    @abstractmethod
    def dumps(self) -> str:
        """Return instance string representation."""

    @classmethod
    @abstractmethod
    def loads(cls, poly: str):
        """Load geojson prepresentation."""
