"""Concrete characterized entry points over the recorded corpus.

Each ``_run_*`` callable performs one fitting/metrology operation and returns
the result object (an xarray Dataset, a numpy array, or a mapping).
:func:`build_registry` wires them into :class:`_registry.EntryPoint`
descriptors with their input files and per-array tolerance classes.

Entry points that depend on facility resources absent from this environment
(the IO-share sector-module workbooks, an ANSYS install, the heavy winding-pack
vtk caches) are registered with a ``skip_reason`` probe so they appear as
visible skips rather than being silently omitted -- their goldens are produced
once the fixtures are staged via the assembly data registry.

The assembly code is imported inside the callables so the registry stays cheap
to inspect and a missing optional dependency degrades to a skip, never a
collection error.
"""

from __future__ import annotations

from . import _skips
from ._registry import EntryPoint

# Recorded-corpus paths, relative to the repository root.
GAP_UNIFORM_INPUT = "data/Assembly/constant_adaptive_fourier.txt"
GAP_VELOCITY_INPUT = "data/Assembly/Gap_Size_18_Coils.txt"
ASBUILT_INPUT = "input/ITER/TFC18_asbuilt.xlsx"


# --- runnable, in-repo-reproducible ---------------------------------------


def _run_uniform_gap(name: str):
    """Build a uniform-gap dataset (gap waveform + Fourier proxy) for ``name``."""
    from nova.assembly.gap import UniformGap

    return UniformGap(name, dirname="root.data/Assembly").data


def _run_fiducial_idm():
    """Return the IDM-sourced CCL fiducial deltas (mm) per TF coil."""
    from nova.assembly.fiducialccl import FiducialIDM

    return FiducialIDM().delta


def _run_fiducial_re():
    """Return the reverse-engineering CCL fiducial deltas from the as-built book."""
    from nova.assembly.fiducialccl import FiducialRE

    return FiducialRE().delta


def _run_asbuilt_ccl_deltas():
    """Return the as-built CCL deltas (mm) read from the TFC18 workbook."""
    from nova.structural.asbuilt import AsBuilt

    return AsBuilt().ccl_deltas()


def _run_nominal_fiducials():
    """Return the fixed nominal fiducial coordinates (mm), index A-H."""
    from nova.assembly.fiducialdata import FiducialData

    return FiducialData.fiducials()


def _run_uniform_vault():
    """Build the uniform vault geometry model (coil/boundary/centroid)."""
    from nova.assembly.vault import UniformVault

    return UniformVault(0.824, 0.3, 0.3).data


def _run_base_assembly_vault():
    """Build the base-assembly vault geometry (nominal gap -> vault geometry)."""
    from nova.assembly.vault import BaseAssembly

    return BaseAssembly().vault.data


# --- blocked here: facility fixtures absent (visible skips) ----------------


def _run_sector_fit():
    from nova.assembly.fiducialsector import FiducialSector

    return FiducialSector(phase="SSAT BR", sectors={7: [8, 9]}, private=True).delta


def _run_pit_gaps():
    from nova.assembly.fiducialpit import FiducialPit

    return FiducialPit(sectors={7: [8, 9]}).gaps


def _run_vault_fourier_proxy():
    from nova.assembly.fouriervault import FourierVault  # noqa: F401

    raise RuntimeError("ANSYS-backed vault spectral proxy is not runnable offline")


def _run_winding_pack():
    from nova.assembly.uniformwindingpack import UniformWindingPack

    mesh = UniformWindingPack().mesh
    import numpy as np

    return {
        "n_points": np.array([mesh.n_points], dtype=float),
        "bounds": np.asarray(mesh.bounds, dtype=float),
        "center": np.asarray(mesh.center, dtype=float),
    }


def _gap_tolerances() -> dict[str, str]:
    return {
        "gap": "length_mm",
        "roll": "length_mm",
        "yaw": "length_mm",
        "delta": "length_mm",
        "fft": "coefficient",
    }


def build_registry() -> list[EntryPoint]:
    """Return the characterized entry points (runnable and visibly skipped)."""
    return [
        # Gap asymmetry + Fourier spectral proxy over the recorded gap files.
        EntryPoint(
            id="gap.uniform.k0",
            callable="nova.assembly.gap:UniformGap",
            run=lambda: _run_uniform_gap("k0"),
            inputs=(GAP_UNIFORM_INPUT,),
            tolerances=_gap_tolerances(),
        ),
        # CCL fiducial deltas -- the live SSAT/in-pit metrology deltas (mm).
        EntryPoint(
            id="ccl.fiducial_idm",
            callable="nova.assembly.fiducialccl:FiducialIDM",
            run=_run_fiducial_idm,
            inputs=(),
            tolerances_default="length_mm",
        ),
        EntryPoint(
            id="ccl.fiducial_re",
            callable="nova.assembly.fiducialccl:FiducialRE",
            run=_run_fiducial_re,
            inputs=(ASBUILT_INPUT,),
            tolerances_default="length_mm",
        ),
        EntryPoint(
            id="structural.asbuilt.ccl_deltas",
            callable="nova.structural.asbuilt:AsBuilt",
            run=_run_asbuilt_ccl_deltas,
            inputs=(ASBUILT_INPUT,),
            tolerances_default="length_mm",
        ),
        # Fixed nominal fiducial geometry (mm) underpinning coil-position extraction.
        EntryPoint(
            id="fiducials.nominal",
            callable="nova.assembly.fiducialdata:FiducialData.fiducials",
            run=_run_nominal_fiducials,
            inputs=(),
            tolerances_default="length_mm",
        ),
        # Vault geometry model (the runnable core of the vault spectral-proxy work;
        # the ANSYS Fourier extraction on top of it is the blocked entry below).
        EntryPoint(
            id="vault.uniform_geometry",
            callable="nova.assembly.vault:UniformVault",
            run=_run_uniform_vault,
            inputs=(),
            tolerances_default="coefficient",
        ),
        EntryPoint(
            id="vault.base_assembly_geometry",
            callable="nova.assembly.vault:BaseAssembly",
            run=_run_base_assembly_vault,
            inputs=(),
            tolerances_default="coefficient",
        ),
        # --- blocked here: staged via the assembly data registry later ---
        EntryPoint(
            id="sector.fit.ssat",
            callable="nova.assembly.fiducialsector:FiducialSector",
            run=_run_sector_fit,
            inputs=(),
            tolerances_default="length_mm",
            skip_reason=_skips.sector_modules_absent,
        ),
        EntryPoint(
            id="pit.gaps",
            callable="nova.assembly.fiducialpit:FiducialPit",
            run=_run_pit_gaps,
            inputs=(),
            tolerances_default="length_mm",
            skip_reason=_skips.sector_modules_absent,
        ),
        EntryPoint(
            id="vault.fourier_proxy",
            callable="nova.assembly.fouriervault:FourierVault",
            run=_run_vault_fourier_proxy,
            inputs=(),
            tolerances_default="coefficient",
            skip_reason=_skips.ansys_missing,
        ),
        EntryPoint(
            id="windingpack.uniform",
            callable="nova.assembly.uniformwindingpack:UniformWindingPack",
            run=_run_winding_pack,
            inputs=(),
            tolerances_default="length_m",
            skip_reason=_skips.windingpack_heavy,
        ),
    ]
