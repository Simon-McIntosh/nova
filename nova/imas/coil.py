"""Manage common coil data access methods."""

from nova.imas.dataset import ImasIds


def coil_name(coil):
    """Return coil identifier, return coil name if empty."""
    if " " not in coil.name:
        return coil.name
    return coil.identifier


def coil_label(coil):
    """Return verbose coil label."""
    return coil.name


def full_coil_name(identifier: str):
    """Return full coil name from identifier."""
    match list(identifier):
        case str(prefix), "C", "C", "_", str(i), "-", str(j) if prefix in "TSB":
            position = dict(zip("TSB", ["Top", "Side", "Bottom"]))[prefix]
            return f"{position} Correction Coils, {prefix}CC-{i} and {prefix}CC-{j}"
        case "C", "S", str(index), str(postfix):
            position = dict(zip("UL", ["Upper", "Lower"]))[postfix]
            return f"Central Solenoid Module {index} {position}"
        case "V", str(position), *index:
            position = dict(zip("UL", ["Upper", "Lower"]))[position]
            return f"{position} Vertical Stability Coil {''.join(index)}"
        case "E", str(position), *index:
            position = dict(zip("UEL", ["Upper", "Middle", "Lower"]))[position]
            return f"{position} ELM Coil {''.join(index)}"
        case "P", "F", str(index):
            return f"Poloidal Field Coil {index}"
        case "T", "F", "_", *index:
            coil_a, coil_b = "".join(index).split("_", 2)
            return f"Toroidal Field Coil Pair, TF{coil_a} and TF{coil_b}"
        case _:
            raise NotImplementedError(f"coil name not implemented for {identifier}")


def part_name(name):
    """Return coil part."""
    if not isinstance(name, str):
        name = coil_name(name)
    if name[:2] in ["EU", "EE", "EL"]:
        return "elm"
    if name[:2] in ["VU", "VL"]:
        return "vs3"
    if name[-2:] == "CC" or name[:2] == "CC" or name[1:3] == "CC":
        return "cc"
    if name[:2] == "CS":
        return "cs"
    if name[:2] == "PF":
        return "pf"
    if name[:2] == "TF":
        return "tf"
    raise NotImplementedError(f"coil part not implemented for {name}")


def coil_names(coil_node: ImasIds) -> list[str]:
    """Return stripped coil name list."""
    return [coil_name(coil).strip() for coil in coil_node]


def coil_labels(coil_node: ImasIds) -> list[str]:
    """Return verbose coil labels."""
    return [coil_label(coil).strip() for coil in coil_node]
