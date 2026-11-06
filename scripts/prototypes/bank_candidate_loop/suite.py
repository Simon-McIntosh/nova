"""Run paired isolated CPU files and the declared support-slot permutation."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import difflib
import json
import os
from pathlib import Path
import subprocess
import tempfile

from scripts.prototypes.bank_wall_events.suite import BASE_FILES, archive, run_file
from scripts.prototypes.bank_solve_cycle.receipts import assemble


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("run", type=Path)
    parser.add_argument("base")
    args = parser.parse_args()
    head = subprocess.check_output(
        ["git", "-C", str(args.root), "rev-parse", "HEAD"], text=True
    ).strip()
    with tempfile.TemporaryDirectory(
        prefix="candidate-suite-", dir=os.environ["TMPDIR"]
    ) as temporary:
        scratch = Path(temporary)
        base, negative = scratch / "base", scratch / "negative"
        archive(args.root, args.base, base)
        archive(args.root, head, negative)
        path = negative / "nova/equilibrium/topology.py"
        source = path.read_text()
        target = "candidates_admitted = saddles & accepted"
        assert source.count(target) == 1
        path.write_text(
            source.replace(
                target,
                "candidates_admitted = saddles & "
                "accepted.at[0].set(accepted[1]).at[1].set(accepted[0])",
            )
        )
        (args.run / "negative-mutation.diff").write_text(
            "".join(
                difflib.unified_diff(
                    source.splitlines(True), path.read_text().splitlines(True)
                )
            )
        )
        run_file(
            negative,
            head + "+swapped-support",
            "test_candidate_inspection_batched",
            "negative",
            args.run,
            scratch,
            "slots_match_individual",
        )

        def arm(label, tree, revision, names):
            for name in names:
                selection = (
                    "not certificate"
                    if name == "test_candidate_inspection_batched"
                    else None
                )
                run_file(tree, revision, name, label, args.run, scratch, selection)
            return assemble(
                args.run,
                [label + "-" + name for name in names],
                revision,
                label + "-suite.log",
            )

        # This driver runs in its own sirius allocation, where the arms may overlap.
        with ThreadPoolExecutor(max_workers=2) as pool:
            before = pool.submit(arm, "base", base, args.base, BASE_FILES)
            after = pool.submit(
                arm,
                "head",
                args.root,
                head,
                ("test_candidate_inspection_batched", *BASE_FILES),
            )
            suites = {"base": before.result(), "head": after.result()}
        (args.run / "suites.json").write_text(json.dumps(suites, indent=2))
        added = set(suites["head"]["failure_ids"]) - set(suites["base"]["failure_ids"])
        print("ADDED_FAILURES", sorted(added), flush=True)
        print("Removing scratch", scratch, flush=True)
        return bool(added)


if __name__ == "__main__":
    raise SystemExit(main())
