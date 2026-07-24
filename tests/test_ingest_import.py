"""Pin the ingest-IO import graph as pandas-free.

The ingest-IO modules historically imported ``pandas`` at module scope and so
only imported when the optional ``interchange`` extra (pandas + openpyxl) was
installed. They now use numpy / columnar equivalents and read spreadsheets
through a lazily imported ``openpyxl`` at call time, so importing them must not
bind ``pandas`` or ``openpyxl`` into the module namespace.

``pandas`` cannot be blocked outright with a meta-path finder the way
``matplotlib`` is in ``test_plot_import`` -- xarray, a core dependency, imports
pandas unconditionally. The verifiable guarantee is instead that these modules
carry no direct ``import pandas`` (nothing named ``pandas`` in their globals)
and import cleanly without the interchange extra.
"""

import subprocess
import sys
import textwrap


INGEST_MODULES = [
    "nova.imas.io_magnetics",
    "nova.imas.magnetic_diagnostics",
    "nova.imas.extrapolate",
]


def test_ingest_modules_import_without_pandas_binding():
    """Importing each ingest module binds neither pandas nor openpyxl."""
    script = textwrap.dedent(
        """
        import importlib
        for name in {modules!r}:
            module = importlib.import_module(name)
            for optional in ("pandas", "openpyxl"):
                assert not hasattr(module, optional), (
                    f"{{name}} imports {{optional}} at module scope"
                )
        """
    ).format(modules=INGEST_MODULES)
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr
