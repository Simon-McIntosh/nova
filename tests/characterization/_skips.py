"""Runtime probes that decide whether a blocked entry point can run here.

An entry point that depends on a facility resource absent from this
environment -- an ANSYS installation, a pyvista API that has since drifted, the
IO-share sector-module workbooks -- must be a *visible skip* with a concrete
reason, never a silent omission. Each probe returns ``None`` when the entry
point is runnable and a short reason string when it is not, so the reason is
accurate to the machine the harness runs on rather than baked in.
"""

from __future__ import annotations

import importlib
import os
from pathlib import Path

from ._environment import repo_root


def ansys_missing() -> str | None:
    """Reason if the ANSYS-backed vault spectral proxy cannot import."""
    try:
        importlib.import_module("nova.assembly.fouriervault")
    except Exception as error:  # noqa: BLE001 - any import failure blocks it
        return f"vault spectral proxy needs an ANSYS install: {type(error).__name__}"
    return None


def pyvista_cell_points_drifted() -> str | None:
    """Reason if pyvista dropped the cell_points API the winding-pack path uses."""
    try:
        import pyvista

        mesh = pyvista.Sphere()
    except Exception as error:  # noqa: BLE001
        return f"pyvista unavailable: {type(error).__name__}"
    if not hasattr(mesh, "cell_points"):
        return (
            "winding-pack / centerline path calls PolyData.cell_points, removed in "
            f"pyvista {pyvista.__version__}; needs a get_cell(...).points shim"
        )
    return None


def sector_modules_absent() -> str | None:
    """Reason if the sector-module fixtures cannot be provided here.

    Runnable when the in-repo canonical units can be transcoded to workbooks,
    or when facility workbooks / cached datasets are already staged. Only a
    genuine absence of all of those is a visible skip.
    """
    from . import _fixtures

    if _fixtures.sector_modules_available():
        return None
    candidates = [
        Path.home() / ".local" / "share" / "nova" / "sector_modules",
    ]
    for base in candidates:
        if base.is_dir() and any(base.glob("*.xlsx")):
            return None
        if base.is_dir() and any(base.glob("*.nc")):
            return None
    return (
        "full sector/pit build needs the in-repo canonical units "
        "(data/Assembly/sector_modules/*.csv) with the workbook transcoder, the "
        "IO-share workbooks (Sector_Module_#*.xlsx), or a cached fiducial_data.nc"
    )


def nominal_ilis_absent() -> str | None:
    """Reason if the nominal ILIS point cloud fixture is not available.

    The pit-gap integration fits every coil's ILIS surfaces against a nominal
    ILIS point cloud (``ILIS_nominal.txt`` on the facility share, or a cached
    ``ILIS_nominal.pickle``). That reference is not part of the in-repo
    canonical corpus, so its absence is a visible skip distinct from the
    sector-module workbooks.
    """
    from nova.database.filepath import FilePath

    try:
        cached = Path(
            FilePath("ILIS_nominal.pickle", dirname=".nova/sector_modules").filepath
        )
    except Exception:  # noqa: BLE001 - cache path resolution should never block
        cached = None
    if cached is not None and cached.exists():
        return None
    share = Path("//io-ws-ccstore1/ANSYS_Data/mcintos/sector_modules/ILIS_nominal.txt")
    if share.exists():
        return None
    return (
        "pit-gap integration needs the nominal ILIS point cloud "
        "(ILIS_nominal.txt from the IO share, or a cached ILIS_nominal.pickle); "
        "sector-module workbooks are present but this reference is off-corpus"
    )


def pit_fixtures_absent() -> str | None:
    """Reason if either the sector modules or the nominal ILIS reference is absent."""
    return sector_modules_absent() or nominal_ilis_absent()


def windingpack_heavy() -> str | None:
    """Reason to skip the heavy winding-pack mesh build in the default lane."""
    if os.environ.get("NOVA_CHARACTERIZATION_HEAVY") != "1":
        return (
            "heavy pyvista mesh build (multi-MB vtk); opt in with "
            "NOVA_CHARACTERIZATION_HEAVY=1"
        )
    cache = repo_root() / "input" / "ITER" / "TF_UCCL.vtk"
    if not cache.exists():
        return "winding-pack vtk cache absent (input/ITER/TF_UCCL.vtk)"
    return None
