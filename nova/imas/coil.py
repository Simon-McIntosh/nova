"""Manage common coil data access methods."""
from nova.imas.database import ImasIds


def coil_name(coil):
    """Return coil identifier, return coil name if empty."""
    if not coil.identifier:
        return coil.name
    return coil.identifier


def coil_names(coil_node: ImasIds) -> list[str]:
    """Return stripped coil name list."""
    return [coil_name(coil).strip() for coil in coil_node]
