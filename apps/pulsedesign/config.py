"""Manage default configuration parameters."""

from dataclasses import dataclass
from nova.imas import database


@dataclass
class IDS(database.IDS):
    """Default input ids."""

    pulse: int = 135013
    run: int = 2
    machine: str = "iter"
    occurrence: int = 0
    user: str = "public"
    name: str | None = None
    backend: str = "hdf5"


ids_attrs = IDS().ids_attrs
