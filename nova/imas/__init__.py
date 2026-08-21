"""Manage access to IMAS data structures."""

__all__ = [
    "CoilData",
    "Database",
    "DiiidCurrentAdapter",
    "Ids",
    "IdsData",
    "ImasIds",
    "complete_profile_current_adapter",
    "resolve_diiid_currents",
]

from .database import CoilData, Database, IdsData
from .dataset import Ids, ImasIds


def __getattr__(name: str):
    """Load the DIII-D current seam only when it is requested."""

    if name in {
        "DiiidCurrentAdapter",
        "complete_profile_current_adapter",
        "resolve_diiid_currents",
    }:
        from . import diiid_current

        return getattr(diiid_current, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
