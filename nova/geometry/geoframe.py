"""Baseclass for dataframe geometroy objects."""
from abc import ABC, abstractmethod


class GeoFrame(ABC):
    """Geometry object abstract base class."""

    def __init__(self):
        self.name = 'geoframe'
        if hasattr(super(), '__post_init__'):
            super().__post_init__()

    def __str__(self):
        """Return name."""
        return self.name

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
