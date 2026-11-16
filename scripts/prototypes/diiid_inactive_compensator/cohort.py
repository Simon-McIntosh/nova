"""Run the established four-row cohort in one GPU job at the committed head."""

import os
from pathlib import Path
import subprocess
import sys

TREE = Path(__file__).resolve().parents[3]


def main():
    output = Path(sys.argv[1]).resolve()
    revision = subprocess.check_output(
        ["git", "-C", str(TREE), "rev-parse", "HEAD"], text=True
    ).strip()
    os.environ.update(
        OMP_NUM_THREADS="4",
        OPENBLAS_NUM_THREADS="1",
        JAX_ENABLE_COMPILATION_CACHE="0",
        JAX_PERSISTENT_CACHE_ENABLE_XLA_CACHES="none",
        XLA_PYTHON_CLIENT_PREALLOCATE="false",
    )
    command = [
        sys.executable,
        str(Path(__file__).with_name("batch.py")),
        "--tree",
        str(TREE),
        "--revision",
        revision,
        "--output",
        str(output),
        "--trace-helper",
        str(TREE / "scripts/prototypes/diiid_hollow_current/trace.py"),
        "--slots",
        "1",
        "--cards",
        "1",
    ]
    result = subprocess.run(command, cwd=TREE)
    print(f"EXIT={result.returncode}", flush=True)
    raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
