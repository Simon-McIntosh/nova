"""The section module must import with the viz extra absent.

A consumer that carries no renderer still reads and transforms sections, so
``nova.geometry.section`` has to keep its vedo usage inside the section's
rendering method and out of the module import graph.
"""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

PROGRAM = """
import sys

# A consumer without the viz extra resolves vedo to nothing.  Setting the entry
# to None makes ``import vedo`` raise, which is how an absent optional
# dependency presents.
sys.modules['vedo'] = None

try:
    import vedo  # noqa: F401
except ImportError:
    pass
else:
    raise AssertionError('the vedo block is not in force')

import nova.geometry.section as section

assert section.Section is not None
print('SECTION-IMPORT-OK')
"""


def test_section_imports_without_vedo():
    """Importing the section module must not require vedo."""
    completed = subprocess.run(
        [sys.executable, "-c", PROGRAM],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    assert "SECTION-IMPORT-OK" in completed.stdout
