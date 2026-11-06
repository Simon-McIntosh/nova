"""Pair fresh per-file CPU processes and challenge both candidate guards."""

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
    names = ("test_candidate_inspection_batched", *BASE_FILES)
    with tempfile.TemporaryDirectory(
        prefix="candidate-suite-", dir=os.environ["TMPDIR"]
    ) as temporary:
        scratch = Path(temporary)
        base = scratch / "base"
        archive(args.root, args.base, base)

        def arm(label, tree, revision):
            for name in names:
                run_file(
                    tree,
                    revision,
                    name,
                    label,
                    args.run,
                    scratch,
                    "not certificate" if name == names[0] else None,
                )
            return assemble(
                args.run,
                [label + "-" + name for name in names],
                revision,
                label + "-suite.log",
            )

        with ThreadPoolExecutor(max_workers=2) as pool:
            before = pool.submit(arm, "base", base, args.base)
            after = pool.submit(arm, "head", args.root, head)
            suites = {"base": before.result(), "head": after.result()}
        (args.run / "suites.json").write_text(json.dumps(suites, indent=2))
        mutations = (
            (
                "overflow",
                "any_lane_overflow(jnp.sum(retained) > compact_capacity),",
                "jnp.asarray(False),",
                "overflow_matches_full",
            ),
            (
                "unrolled",
                "return jax.vmap(inspect)(indices)",
                "return jax.tree.map(lambda *leaves: jnp.stack(leaves), "
                "*[inspect(i) for i in range(policy.null_capacity)])",
                "share_one_program",
            ),
        )
        negative_statuses = []
        for label, target, replacement, selection in mutations:
            negative = scratch / label
            archive(args.root, head, negative)
            path = negative / "nova/equilibrium/topology.py"
            source = path.read_text()
            assert source.count(target) == 1
            changed = source.replace(target, replacement)
            path.write_text(changed)
            (args.run / (label + "-mutation.diff")).write_text(
                "".join(
                    difflib.unified_diff(
                        source.splitlines(True),
                        changed.splitlines(True),
                        fromfile="candidate",
                        tofile=label,
                    )
                )
            )
            negative_statuses.append(
                run_file(
                    negative,
                    head + "+" + label,
                    names[0],
                    label,
                    args.run,
                    scratch,
                    selection,
                )
            )
        added = set(suites["head"]["failure_ids"]) - set(suites["base"]["failure_ids"])
        print("ADDED_FAILURES", sorted(added), flush=True)
        print("NEGATIVE_STATUSES", negative_statuses, flush=True)
        print("Removing scratch", scratch, flush=True)
        return bool(added or negative_statuses != [1, 1])


if __name__ == "__main__":
    raise SystemExit(main())
