"""Manage common coil data access methods."""
from nova.imas.database import ImasIds


def coil_name(coil):
    """Return coil identifier, return coil name if empty."""
    if not coil.identifier:
        return coil.name
    return coil.identifier


def part_name(name):
    """Return coil part."""
    if not isinstance(name, str):
        name = coil_name(name)
    if name[:2] in ["EU", "EE", "EL"]:
        return "elm"
    if name[-2:] == "CC" or name[:2] == "CC" or name[1:3] == "CC":
        return "cc"
    raise NotImplementedError(f"coil part not implemented for {name}")


def coil_names(coil_node: ImasIds) -> list[str]:
    """Return stripped coil name list."""
    return [coil_name(coil).strip() for coil in coil_node]
