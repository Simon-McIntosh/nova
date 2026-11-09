"""Restore the exterior cylinder's subtractive angular projection in a scratch tree."""

from pathlib import Path
import sys

path = Path(sys.argv[1])
text = path.read_text()
replacements = (
    ("if self._slope == 0.0:", "if self._slope == 0.0 and r <= radius:"),
    (
        "b1 == 0.0,\n                cylinder_height,",
        "(b1 == 0.0) & (target_r <= source),\n                cylinder_height,",
    ),
)
for before, after in replacements:
    if text.count(before) != 1:
        raise SystemExit(f"Expected one exterior-cylinder selector: {before!r}")
    text = text.replace(before, after)
path.write_text(text)
print(f"Restored exterior-cylinder angular projection in {path}")
