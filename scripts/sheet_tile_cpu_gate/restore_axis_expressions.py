"""Restore the unguarded sheet owner only inside a scratch extraction."""

import subprocess
import sys
from pathlib import Path

source = subprocess.check_output(
    ["git", "show", "4221309d2e66fd30eadafa7eacc3d967dd233481:nova/biot/sheet.py"]
)
path = Path(sys.argv[1]).resolve()
if path == Path("nova/biot/sheet.py").resolve():
    raise SystemExit("refusing to mutate the worktree source")
path.write_bytes(source)
assert b"def _zero_source_terms" not in path.read_bytes()
assert b"width = xp.hypot(gamma, radius - source) / xp.sqrt(radius * source)" in source
print(
    "Restored zero-modulus division, grading width and target-radius division:",
    path,
)
