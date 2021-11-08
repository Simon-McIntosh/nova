from abc import ABC, abstractmethod
from dataclasses import dataclass

@dataclass 
class GeoFrame(ABC):
    """Geometry object abstract base class."""
    
    @abstractmethod 
    def dumps(self) -> str:
        """Return instance string representation."""
     
    @classmethod
    @abstractmethod 
    def loads(cls, poly: str):
        """Load geojson prepresentation."""
    
    