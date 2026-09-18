"""Publish the pinned cache location a completed pre-warm compiled into.

The lane reads this document to serve the directory the pre-warm wrote, so the
version key is taken from the same resolver the drivers call rather than
restated here: a key that disagreed with the driver would point the lane at an
empty directory and read as a cache that never hit.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from nova.jax.config import (
    configure_persistent_compilation_cache,
    default_persistent_compilation_cache_root,
)


def read_rows(receipt: Path) -> list[dict]:
    """Return the per-program receipt rows the pre-warm appended."""
    if not receipt.is_file():
        return []
    return [
        json.loads(line)
        for line in receipt.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--pin", type=Path, required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--job-id", default=os.environ.get("SLURM_JOB_ID", "unknown"))
    args = parser.parse_args()

    selected = configure_persistent_compilation_cache(
        default_persistent_compilation_cache_root()
    )
    rows = read_rows(args.receipt)
    document = {
        "cache_root": str(selected.root),
        "directory": str(selected.directory),
        "version_key": selected.version_key,
        "source_revision": args.revision,
        "slurm_job_id": args.job_id,
        "program_count": len(rows),
        "misses": sum(int(row["misses"]) for row in rows),
        "hits": sum(int(row["hits"]) for row in rows),
        "compile_seconds": round(sum(float(row["compile_seconds"]) for row in rows), 3),
        "rows": rows,
    }
    args.pin.parent.mkdir(parents=True, exist_ok=True)
    args.pin.write_text(
        json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    summary = {key: value for key, value in document.items() if key != "rows"}
    print("PREWARM_PIN=%s" % args.pin, flush=True)
    print("PREWARM_SUMMARY=%s" % json.dumps(summary, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
