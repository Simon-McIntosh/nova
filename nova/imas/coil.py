"""Manage common coil data access methods."""
from nova.imas.database import ImasIds


def coil_name(coil):
    """Return coil identifier, return coil name if empty."""
    if not coil.identifier:
        return coil.name
    return coil.identifier


def full_coil_name(identifier: str):
    """Return full coil name from identifier."""
    match list(identifier):
        case str(prefix), "C", "C", "_", str(i), "-", str(j) if prefix in "TMB":
            position = dict(zip("TMB", ["Top", "Middle", "Bottom"]))[prefix]
            return f"{position} Correction Coils, {prefix}CC-{i} and {prefix}CC-{j}"
        case "C", "S", str(index), str(postfix):
            position = dict(zip("UL", ["Upper", "Lower"]))[postfix]
            return f"Central Solenoid Module {index} {position}"
        case _:
            raise NotImplementedError(f"coil name not implemented for {identifier}")


def part_name(name):
    """Return coil part."""
    if not isinstance(name, str):
        name = coil_name(name)
    if name[:2] in ["EU", "EE", "EL"]:
        return "elm"
    if name[-2:] == "CC" or name[:2] == "CC" or name[1:3] == "CC":
        return "cc"
    if name[:2] == "CS":
        return "cs"
    raise NotImplementedError(f"coil part not implemented for {name}")


def coil_names(coil_node: ImasIds) -> list[str]:
    """Return stripped coil name list."""
    return [coil_name(coil).strip() for coil in coil_node]
