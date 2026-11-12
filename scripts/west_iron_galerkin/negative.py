"""Require the reciprocal-energy test to reject a centroid-sampling extraction."""

import argparse
from pathlib import Path

import mutant

RELATIVE = "nova/biot/magnetisation.py"


def centroid_sampling(source: str) -> str:
    """Replace the volume-tested interaction with centroid point sampling."""
    begin = source.index("def _galerkin_interaction")
    end = source.index("def _interaction(")
    return (
        source[:begin]
        + "def _galerkin_interaction(cells, plan=_PLAN):\n"
        + "    return cell_interaction(\n"
        + "        tuple(p * _SHEET_LENGTH_SCALE for p in cells.polygons),\n"
        + "        cells.centroids[:, 0] * _SHEET_LENGTH_SCALE,\n"
        + "        cells.centroids[:, 1] * _SHEET_LENGTH_SCALE,\n"
        + "        plan=plan,\n"
        + "    )\n\n\n"
        + source[end:]
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--log", type=Path, default=None)
    args = parser.parse_args()
    log = args.log or args.run / "negative-control.log"
    rejected = mutant.require_rejection(
        repo=args.repo,
        prefix="centroid-mutant-",
        relative=RELATIVE,
        mutate=centroid_sampling,
        module="nova.biot.magnetisation",
        function="_galerkin_interaction",
        marker="cells.centroids[:, 0]",
        selection="cell_average_interaction_is_reciprocal_and_energy_bounded",
        log=log,
    )
    print(f"CENTROID_MUTANT_REJECTED_BY={rejected} test ids", flush=True)
    print("CENTROID_MUTANT_REJECTED; product source blob matches HEAD", flush=True)


if __name__ == "__main__":
    main()
