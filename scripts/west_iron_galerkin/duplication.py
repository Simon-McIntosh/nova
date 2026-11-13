"""Require one copy of the scratch-extraction mutant sequence in this directory."""

import argparse
from pathlib import Path
import shutil

import mutant

SEQUENCE = (
    '"archive"',
    "tar.extractall",
    "MUTANT_MODULE=",
)
OWNER = "mutant.py"
SELF = Path(__file__).name


def carriers(directory: Path) -> list[str]:
    """Return the mutants that carry the whole extract-mutate-verify sequence."""
    found = []
    for path in sorted(directory.glob("*.py")):
        if path.name == SELF:
            continue
        text = path.read_text()
        if all(token in text for token in SEQUENCE):
            found.append(path.name)
    return found


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument(
        "--base",
        default=None,
        help="revision the check must fail at, extracted into a scratch tree",
    )
    args = parser.parse_args()
    directory = args.repo / "scripts/west_iron_galerkin"
    head = mutant.revision(args.repo)
    found = carriers(directory)
    print(f"HEAD={head} CARRIERS={found}", flush=True)
    if args.base is not None:
        scratch = mutant.extract(args.repo, args.base, prefix="duplication-base-")
        try:
            before = carriers(scratch / "scripts/west_iron_galerkin")
            print(f"BASE={args.base} CARRIERS={before}", flush=True)
            assert len(before) > 1, before
            assert OWNER not in before, before
        finally:
            shutil.rmtree(scratch)
        print("DUPLICATION_CHECK_REJECTED_THE_BASE", flush=True)
    assert found == [OWNER], found
    print("DUPLICATION_CHECK_ACCEPTED_THE_HEAD", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
