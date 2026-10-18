"""Rebuild the census with its former proposal and admission steps restored."""

from pathlib import Path

SOURCE = Path(__file__).with_name("census.py")
REPAIRED = "cross >= -rounding"
DROPPED = "cross >= 0"
RING = "inside = jnp.any(inside_ring, axis=1)"
OWN = "inside = inside_ring[:, 0]"


def _with_own_cell_proposal(text):
    if text.count(RING) != 1:
        raise RuntimeError(f"expected exactly one {RING!r} in {SOURCE}")
    return text.replace(RING, OWN)


def containment_only_census():
    """The live census with only the proposal region narrowed to its own cell."""
    text = _with_own_cell_proposal(SOURCE.read_text())
    namespace = {"__name__": "census_containment_control", "__file__": str(SOURCE)}
    exec(compile(text, str(SOURCE), "exec"), namespace)
    return namespace["census"]


def base_admission_census():
    """The census with the named dropping step restored, from the live source."""
    text = _with_own_cell_proposal(SOURCE.read_text())
    count = text.count(REPAIRED)
    if count != 1:
        raise RuntimeError(
            f"expected exactly one {REPAIRED!r} in {SOURCE}, found {count}; "
            "the dropping step this control restores is no longer there"
        )
    namespace = {"__name__": "census_base_step_mutated", "__file__": str(SOURCE)}
    exec(compile(text.replace(REPAIRED, DROPPED), str(SOURCE), "exec"), namespace)
    return namespace["census"]
