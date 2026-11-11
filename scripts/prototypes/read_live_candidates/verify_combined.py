"""Compare merged read owners in fresh, revision-pinned test processes."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile

from scripts.prototypes.read_live_candidates.run import PYTHON, run, snapshot


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--base", required=True)
    parser.add_argument("--head", required=True)
    args = parser.parse_args()
    root, output = args.root.resolve(), args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    record = {
        "base_revision": args.base,
        "head_revision": args.head,
        "job": os.environ.get("SLURM_JOB_ID"),
        "launcher_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "arms": {},
    }
    destination = output / "combined-comparison.json"

    def persist():
        destination.write_text(json.dumps(record, indent=2))

    with tempfile.TemporaryDirectory(
        prefix="combined-read-", dir=os.environ["TMPDIR"]
    ) as temporary:
        scratch = Path(temporary)
        env = dict(os.environ, TMPDIR=str(scratch), JAX_PLATFORMS="cpu")
        trees = {}
        for arm, revision in (("base", args.base), ("head", args.head)):
            tree = scratch / arm
            snapshot(root, revision, tree)
            trees[arm] = tree
            record["arms"][arm] = {}
            names = ["tracked_read", "support_read", "read_tie_break", "read_wall_skip"]
            if arm == "head":
                names.append("read_live_candidates")
            for name in names:
                file = f"tests/test_{name}.py"
                command = [PYTHON, "-m", "pytest", "-p", "no:cacheprovider", "-q", file]
                log = output / f"{arm}-{name}.log"
                code = run(tree, revision, command, log, env)
                text = log.read_text()
                failures = sorted(
                    set(re.findall(r"^(?:FAILED|ERROR) (\S+)", text, re.M))
                )
                record["arms"][arm][file] = {
                    "revision": revision,
                    "command": "PYTHONPATH=. JAX_PLATFORMS=cpu " + " ".join(command),
                    "exit_status": code,
                    "log_path": str(log),
                    "completed": bool(
                        re.search(r"\d+ (?:passed|failed|skipped|error)", text)
                    ),
                    "failure_ids": failures,
                }
                persist()
        base_ids = {
            f for row in record["arms"]["base"].values() for f in row["failure_ids"]
        }
        head_ids = {
            f for row in record["arms"]["head"].values() for f in row["failure_ids"]
        }
        record["added_failure_ids"] = sorted(head_ids - base_ids)
        mutant = trees["head"] / "nova/equilibrium/topology.py"
        source = mutant.read_text()
        anchor = "saddles, size=compact_capacity"
        assert source.count(anchor) == 1
        mutant.write_text(
            source.replace(anchor, "jnp.zeros_like(saddles), size=compact_capacity")
        )
        test = (
            "tests/test_read_live_candidates.py::"
            "test_read_and_book_matches_full_candidates[diverted-1]"
        )
        control = output / "negative-control.log"
        code = run(
            trees["head"],
            args.head,
            [PYTHON, "-m", "pytest", "-p", "no:cacheprovider", "-q", test],
            control,
            env,
        )
        record["negative_control"] = {"exit_status": code, "log_path": str(control)}
        record["complete"] = True
        persist()
        print("COMBINED_DELTA", json.dumps(record), flush=True)
        assert not record["added_failure_ids"], record["added_failure_ids"]
        assert all(
            row["completed"] for arm in record["arms"].values() for row in arm.values()
        )
        assert code == 1 and "FAILED " in control.read_text()


if __name__ == "__main__":
    main()
