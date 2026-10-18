"""Load the stationary census with its exact-zero cell-membership step restored.

The repaired census admits a cell seed whose edge cross product is negative only
beyond the rounding that value carries. To falsify that repair it is not enough
to read a trace taken earlier: the dropping step has to be reinstated and the
census run again. This module does that by taking the census source as written,
replacing the rounding-bearing comparison with the exact-zero one it replaced,
and executing the result. The returned function is the census the repair exists
to supersede, built from the current source rather than from a copy, so a later
edit to the admission step cannot leave this control measuring a stale program.
"""

from pathlib import Path

SOURCE = Path(__file__).with_name("census.py")
REPAIRED = "cross >= -rounding"
DROPPED = "cross >= 0"


def base_admission_census():
    """The census with the named dropping step restored, from the live source."""
    text = SOURCE.read_text()
    count = text.count(REPAIRED)
    if count != 1:
        raise RuntimeError(
            f"expected exactly one {REPAIRED!r} in {SOURCE}, found {count}; "
            "the dropping step this control restores is no longer there"
        )
    namespace = {"__name__": "census_base_step_mutated", "__file__": str(SOURCE)}
    exec(compile(text.replace(REPAIRED, DROPPED), str(SOURCE), "exec"), namespace)
    return namespace["census"]