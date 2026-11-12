"""Require the pre-symmetrisation asymmetry test to reject a non-reciprocal pair."""

import argparse
from pathlib import Path

import mutant

RELATIVE = "nova/biot/tiledassembly.py"

ANCHOR = """                mutual[rows[chosen], columns[chosen]] = value
                mutual[columns[chosen], rows[chosen]] = value
"""

PERTURBATION = """                mutual[rows[chosen], columns[chosen]] = value
                mutual[columns[chosen], rows[chosen]] = value
                if rows[chosen[0]] != columns[chosen[0]]:
                    # one edge-pair quadrature, mirrored only, so reciprocity breaks
                    mutual[columns[chosen[0]], rows[chosen[0]]] = 1.001 * value[0]
"""


def one_mirrored_pair(source: str) -> str:
    """Overwrite one off-diagonal edge-pair quadrature without its mirror."""
    assert source.count(ANCHOR) == 1, source.count(ANCHOR)
    return source.replace(ANCHOR, PERTURBATION)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--log", type=Path, default=None)
    args = parser.parse_args()
    log = args.log or args.run / "negative-control.log"
    rejected = mutant.require_rejection(
        repo=args.repo,
        prefix="asymmetry-mutant-",
        relative=RELATIVE,
        mutate=one_mirrored_pair,
        module="nova.biot.tiledassembly",
        function="galerkin_cell_energy",
        marker="1.001 * value[0]",
        selection="pre_symmetrisation_energy_asymmetry",
        log=log,
    )
    print(f"ASYMMETRY_MUTANT_REJECTED_BY={rejected} test ids", flush=True)
    print("ASYMMETRY_MUTANT_REJECTED; product source blob matches HEAD", flush=True)


if __name__ == "__main__":
    main()
