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
    """Reason if the recorded sector-module workbooks / cache are not present."""
    candidates = [
        repo_root() / "data" / "Assembly" / "sector_modules",
        Path.home() / ".local" / "share" / "nova" / "sector_modules",
    ]
    for base in candidates:
        if base.is_dir() and any(base.glob("*.xlsx")):
            return None
        if base.is_dir() and any(base.glob("*.nc")):
            return None
    return (
        "full sector/pit build needs the IO-share sector-module workbooks "
        "(Sector_Module_#*.xlsx) or a cached fiducial_data.nc fixture; provide via "
        "the assembly data registry"
    )


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
