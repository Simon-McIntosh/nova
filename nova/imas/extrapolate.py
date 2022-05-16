"""Extrapolate equilibria beyond separatrix."""
from dataclasses import dataclass

from nova.imas.database import Database


@dataclass
class Extrapolate(Database):
    """Extrapolate equlibrium beyond separatrix ids."""

    pulse: int
    run: int
    machine: str = 'iter'
    ids_name: str = 'extrapolate'

    def __post_init__(self):
        """Load data."""
        super().__post_init__()

    def build(self):
        """Calculate derived quantities."""
        try:
            self.load()
        except (FileNotFoundError, OSError, KeyError):
            self.build()


if __name__ == '__main__':

    ext = Extrapolate(114101, 41)
