"""Run archived revision pairs in fresh processes with per-revision caches."""

import os
from pathlib import Path
import sys
import tempfile

from scripts.prototypes.bank_wall_events.device import execute
from scripts.prototypes.bank_wall_events.suite import archive


def main():
    root, run = map(Path, sys.argv[1:3])
    revisions = sys.argv[3:5]
    with tempfile.TemporaryDirectory(
        prefix="read-sensitivity-", dir=os.environ["TMPDIR"]
    ) as temporary:
        scratch = Path(temporary)
        trees = [scratch / name for name in ("base", "head")]
        for tree, revision in zip(trees, revisions, strict=True):
            archive(root, revision, tree)
        for process in range(3):
            for arm, tree, revision in zip(
                ("base", "head"), trees, revisions, strict=True
            ):
                command = [
                    "bash",
                    str(
                        root
                        / "scripts/prototypes/bank_booking_determinism/measurement.sh"
                    ),
                    str(run / f"cache-{arm}"),
                    sys.executable,
                    "-P",
                    str(
                        root
                        / "scripts/prototypes/bank_read_count_sensitivity/capture.py"
                    ),
                    "--output",
                    str(run / f"{arm}-{process}.json"),
                    "--revision",
                    revision,
                ]
                execute(
                    tree,
                    revision,
                    command,
                    run / f"{arm}-{process}.log",
                    dict(
                        os.environ,
                        PYTHONPATH=str(tree),
                        TMPDIR=str(scratch),
                        JAX_DEBUG_LOG_MODULES="jax._src.compiler",
                    ),
                )
        print("Removing scratch", scratch, flush=True)


if __name__ == "__main__":
    main()
