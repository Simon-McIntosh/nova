"""Manage default configuration parameters."""

from dataclasses import dataclass
from nova.imas import database

#: The library defaults the base class restores during construction when no
#: ids are open; declared defaults restore over exactly those sentinels.
_BASE_DEFAULTS = {
    "pulse": 0,
    "run": 0,
    "name": None,
    "occurrence": 0,
    "machine": "iter",
    "user": "public",
    "backend": "hdf5",
}


@dataclass
class IDS(database.Database):
    """Default input ids."""

    pulse: int = 135013
    run: int = 2
    machine: str = "iter"
    occurrence: int = 0
    user: str = "public"
    name: str | None = None
    backend: str = "hdf5"

    def __post_init__(self):
        """Apply the base defaults, then restore this class's declared ones."""
        super().__post_init__()
        for name, value in _BASE_DEFAULTS.items():
            if getattr(self, name) == value:
                setattr(self, name, getattr(type(self), name))


ids_attrs = IDS().ids_attrs
