"""The named data roots resolve through FilePath with an environment override.

Product modules used to hard-code GPFS and home paths. Each named root now
comes from :mod:`nova.database.filepath`, resolves from one environment
variable, and keeps its former literal as the default when the variable is
unset. These tests pin both halves: the default is unchanged on this
workstation, and an override reaches the product constants as well as the
accessors.
"""

from __future__ import annotations

import importlib
from pathlib import Path

import pytest

from nova.database import filepath


CACHE_ROOT = Path.home() / ".cache" / "nova-mast"

DEFAULTS = (
    (filepath.mast_level1_shots, Path("/work/projects/imas_gpu/mast/level1/shots")),
    (filepath.mast_level2_shots, Path("/work/projects/imas_gpu/mast/level2/shots")),
    (
        filepath.diiid_machine_source,
        Path("/home/ITER/tribolp/Public/imasdb/DIII-D/200000.nc"),
    ),
    (filepath.derived_cache_root, CACHE_ROOT),
    (filepath.misfit_hunt_root, CACHE_ROOT / "misfit-hunt"),
    (
        filepath.labeller_session_root,
        Path("/work/projects/imas_gpu/sophelio/labeller_sessions/76906a29"),
    ),
)

OVERRIDES = (
    ("NOVA_MAST_DATA_ROOT", "/data/mast", filepath.mast_level1_shots, "level1/shots"),
    ("NOVA_MAST_DATA_ROOT", "/data/mast", filepath.mast_level2_shots, "level2/shots"),
    ("NOVA_DIIIID_SOURCE", "/data/diiid.nc", filepath.diiid_machine_source, ""),
    ("NOVA_CACHE_ROOT", "/data/cache", filepath.derived_cache_root, ""),
    ("NOVA_CACHE_ROOT", "/data/cache", filepath.misfit_hunt_root, "misfit-hunt"),
    ("NOVA_LABELLER_SOURCE", "/data/labeller", filepath.labeller_session_root, ""),
)


@pytest.fixture
def reloaded(monkeypatch):
    """Reload modules under an environment override, then restore them."""
    loaded = []

    def load(module_name: str):
        module = importlib.reload(importlib.import_module(module_name))
        loaded.append(module)
        return module

    yield load

    monkeypatch.undo()
    for module in loaded:
        importlib.reload(module)


@pytest.mark.parametrize("accessor, expected", DEFAULTS)
def test_default_is_unchanged(monkeypatch, accessor, expected):
    for variable in (
        "NOVA_MAST_DATA_ROOT",
        "NOVA_DIIIID_SOURCE",
        "NOVA_CACHE_ROOT",
        "NOVA_LABELLER_SOURCE",
        "NOVA_PUBLIC_HOME_ROOT",
    ):
        monkeypatch.delenv(variable, raising=False)
    assert accessor() == expected


@pytest.mark.parametrize("variable, value, accessor, relative", OVERRIDES)
def test_override_is_honoured(monkeypatch, variable, value, accessor, relative):
    monkeypatch.setenv(variable, value)
    expected = Path(value) / relative if relative else Path(value)
    assert accessor() == expected


def test_public_imasdb_root_default_and_override(monkeypatch):
    monkeypatch.delenv("NOVA_PUBLIC_HOME_ROOT", raising=False)
    assert filepath.public_imasdb_root("tribolp", "DIII-D") == Path(
        "/home/ITER/tribolp/public/imasdb/diii-d"
    )
    monkeypatch.setenv("NOVA_PUBLIC_HOME_ROOT", "/data/home")
    assert filepath.public_imasdb_root("tribolp", "DIII-D") == Path(
        "/data/home/tribolp/public/imasdb/diii-d"
    )


@pytest.mark.parametrize(
    "module_name, attribute, variable, value, relative",
    (
        (
            "nova.imas.mast_vacuum_cohort",
            "SHOT_STORE",
            "NOVA_MAST_DATA_ROOT",
            "/data/mast",
            "level1/shots",
        ),
        (
            "nova.catalog.mast_geometry",
            "DEFAULT_LEVEL1_ROOT",
            "NOVA_MAST_DATA_ROOT",
            "/data/mast",
            "level1/shots",
        ),
        (
            "nova.catalog.mast_geometry",
            "DEFAULT_LEVEL2_ROOT",
            "NOVA_MAST_DATA_ROOT",
            "/data/mast",
            "level2/shots",
        ),
        (
            "nova.scripts.identify_source_cocos",
            "LEVEL1",
            "NOVA_MAST_DATA_ROOT",
            "/data/mast",
            "level1/shots",
        ),
        (
            "nova.media.sources.diiid_efit",
            "NETCDF_SOURCE",
            "NOVA_DIIIID_SOURCE",
            "/data/diiid.nc",
            "",
        ),
        (
            "nova.imas.diiid_current",
            "DEFAULT_ACTIVE_COIL_ENTRY",
            "NOVA_DIIIID_SOURCE",
            "/data/diiid.nc",
            "",
        ),
        (
            "nova.imas.diiid_machine_ids",
            "SOURCE_PATH",
            "NOVA_DIIIID_SOURCE",
            "/data/diiid.nc",
            "",
        ),
        (
            "nova.media.sources.nova_labels",
            "SESSION_ROOT",
            "NOVA_LABELLER_SOURCE",
            "/data/labeller",
            "",
        ),
        (
            "nova.scripts.render_misfit_figures",
            "HUNT",
            "NOVA_CACHE_ROOT",
            "/data/cache",
            "misfit-hunt",
        ),
    ),
)
def test_product_constant_follows_filepath(
    monkeypatch, reloaded, module_name, attribute, variable, value, relative
):
    monkeypatch.setenv(variable, value)
    module = reloaded(module_name)
    expected = Path(value) / relative if relative else Path(value)
    assert getattr(module, attribute) == expected