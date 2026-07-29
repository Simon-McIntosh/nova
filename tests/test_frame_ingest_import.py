"""Pin the nova.frame geometry-ingest modules as pandas-free.

These modules historically read spreadsheets and coil tables through pandas at
module scope, so they only imported with the optional ``interchange`` extra
installed. They now read via a lazily imported ``openpyxl`` and numpy columnar
equivalents, so importing them must bind neither ``pandas`` nor ``openpyxl``
into the module namespace.

``pandas`` cannot be blocked outright with a meta-path finder -- xarray, a core
dependency, imports it unconditionally. The verifiable guarantee is instead
that these modules carry no direct ``import pandas`` (nothing named ``pandas``
in their globals) and import cleanly without the interchange extra.
"""

import importlib

import pytest


INGEST_MODULES = [
    "nova.frame.machinedata",
    "nova.frame.itergeom",
]


@pytest.mark.parametrize("name", INGEST_MODULES)
def test_frame_ingest_module_has_no_pandas_binding(name):
    """Importing each ingest module binds neither pandas nor openpyxl."""
    module = importlib.import_module(name)
    for optional in ("pandas", "pd", "openpyxl"):
        assert not hasattr(module, optional), f"{name} binds {optional} at module scope"


def test_vde_source_is_pandas_free():
    """vde carries no module-scope pandas or moviepy import.

    vde depends on further optional packages (moviepy for animation) and a
    disruption-waveform reader, so it is not importable in the base test
    environment; its pandas independence is verified at the source level: the
    concatenation that fed ``pandas`` is now a numpy stack and the heavy
    imports are deferred to their call sites.
    """
    import ast
    from pathlib import Path

    import nova.frame

    source = Path(nova.frame.__file__).parent / "vde.py"
    tree = ast.parse(source.read_text())
    module_imports = {
        alias.name.split(".")[0]
        for node in tree.body
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in getattr(node, "names", [])
    } | {
        node.module.split(".")[0]
        for node in tree.body
        if isinstance(node, ast.ImportFrom) and node.module
    }
    assert "pandas" not in module_imports
    assert "moviepy" not in module_imports
