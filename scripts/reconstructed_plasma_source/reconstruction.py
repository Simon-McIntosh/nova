"""Drive the full ITER plant from the sectioned-set cache with a CORSICA reconstruction.

The presentation is the one the MAST reconstruction path already uses: the
equilibrium's stored last closed flux surface is named as the boundary the
plasma current is prescribed inside, with no separatrix outline or x-point for
the resolver to prefer, and the coil currents are resampled onto a coarser
cadence so they arrive on their own clock the way a reconstruction's separately
acquired traces do.  Nothing else about the slice changes: its flux map, its
flux functions and its declared boundary flux are the equilibrium's own.

The measurement drives that presented source and the same equilibrium's generic
presentation through the full sectioned ITER machine loaded from its cache, and
compares the plasma flux the plant links.  The machine is assembled by
``load_sectioned_machine``; no sectioned-set refinement or linkage build may run
while either plant is built, and that is counted rather than assumed.
"""

from __future__ import annotations

import json
import sys
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path

import numpy as np

from nova.imas.plasma_source import CoilCurrentSeries, PlasmaSource

CACHE = Path("/home/ITER/mcintos/Code/nova/.cache/iter-passive-plasma-circuit")
SECTIONED_KEY = "66c10162f3e37adb2679"
IMASDB = Path("/work/imas/shared/imasdb/ITER/3")
CORSICA = IMASDB / "130506" / "6"
CORSICA_DD_VERSION = "3.41.0"
CORSICA_INDEX = 45
COIL_STRIDE = 4
TOLERANCE = 1e-3
FIGURE = Path(
    "docs/figures/iter-passive-plasma-circuit/ippc-reconstruction-full-plant"
    "/agreement.svg"
)


def corsica_source(
    *,
    index: int = CORSICA_INDEX,
    provenance: str = "CORSICA 130506/6",
) -> PlasmaSource:
    """Open one CORSICA 130506/6 slice through the DD3 translator."""
    import imas

    with imas.DBEntry(
        f"imas:hdf5?path={CORSICA}", "r", dd_version=CORSICA_DD_VERSION
    ) as entry:
        equilibrium = entry.get("equilibrium", lazy=False, autoconvert=False)
        active = entry.get("pf_active", lazy=False, autoconvert=False)
    return PlasmaSource.from_imas(
        equilibrium, active, provenance=provenance, indices=[index]
    )


def reconstruction_style(
    source: PlasmaSource, *, stride: int = COIL_STRIDE
) -> PlasmaSource:
    """Present a source the way a reconstruction arrives at the plant.

    The stored separatrix outline and every x-point are dropped, so the boundary
    resolver falls through to the stored last closed flux surface, and the coil
    currents are resampled onto every ``stride``-th row of their own clock.  The
    slice's flux map, flux functions and boundary contour are untouched.
    """
    slices = tuple(
        replace(slice_, outline=np.empty((0, 2)), x_points=())
        for slice_ in source.slices
    )
    clock = np.asarray(source.coils.time, dtype=float)
    sampled = np.arange(0, clock.size, stride)
    coils = CoilCurrentSeries(
        clock[sampled],
        source.coils.names,
        np.asarray(source.coils.current)[sampled],
    )
    return replace(source, slices=slices, coils=coils)


@contextmanager
def rebuild_spy():
    """Refuse a subdivided set or linkage build, and record every binding guarded.

    ``subdivide_passive_thickness`` and ``circuit_linkage_matrix`` are bound by
    name into several modules, so patching one attribute leaves the others live.
    Every loaded module attribute that *is* one of those functions is replaced
    for the duration, and the names are yielded so a caller can assert the guard
    was installed at all.
    """
    from nova.circuit.conductor import subdivide_passive_thickness
    from nova.circuit.linkage import circuit_linkage_matrix

    guarded: list[str] = []
    saved: list[tuple[object, str, object]] = []

    def refusal(label):
        def spy(*args, **kwargs):
            raise AssertionError(f"{label} ran on the cached-machine path")

        return spy

    for module in list(sys.modules.values()):
        if module is None or not getattr(module, "__name__", None):
            continue
        for attribute, value in list(vars(module).items()):
            for original, label in (
                (subdivide_passive_thickness, "subdivide_passive_thickness"),
                (circuit_linkage_matrix, "circuit_linkage_matrix"),
            ):
                if value is original:
                    saved.append((module, attribute, value))
                    setattr(module, attribute, refusal(label))
                    guarded.append(f"{module.__name__}.{attribute}")
    try:
        yield guarded
    finally:
        for module, attribute, value in saved:
            setattr(module, attribute, value)


def _relative_error(reference: np.ndarray, measured: np.ndarray) -> dict:
    difference = np.abs(measured - reference)
    scale = np.abs(reference)
    relative = np.divide(
        difference, scale, out=np.zeros_like(difference), where=scale > 0
    )
    return {
        "max_abs": float(np.max(difference)) if difference.size else 0.0,
        "max_rel": float(np.max(relative)) if relative.size else 0.0,
        "violations": int(np.sum(relative > TOLERANCE)),
        "count": int(relative.size),
        "relative": relative,
    }


def measure(scratch: Path) -> dict:
    """Drive both presentations through the full plant and compare their flux."""
    from nova.circuit.coupled_plant import coupled_plant
    from nova.circuit.passive import load_sectioned_machine

    scratch = Path(scratch)
    generic = corsica_source(provenance="CORSICA 130506/6")
    reconstruction = reconstruction_style(
        corsica_source(provenance="CORSICA 130506/6 as a reconstruction")
    )

    with rebuild_spy() as guarded:
        assert guarded, "no rebuild binding was guarded"
        machine = load_sectioned_machine(
            CACHE / "sectioned-conductor-set" / SECTIONED_KEY
        )
        generic_plant = coupled_plant(machine, generic, cache_dir=scratch / "generic")
        reconstruction_plant = coupled_plant(
            machine, reconstruction, cache_dir=scratch / "reconstruction"
        )

    sections = _relative_error(
        generic_plant.section_plasma_flux, reconstruction_plant.section_plasma_flux
    )
    modes = _relative_error(
        generic_plant.mode_plasma_flux, reconstruction_plant.mode_plasma_flux
    )
    report = {
        "machine": machine.label,
        "n_section": int(machine.section_inductance.shape[0]),
        "n_modes": int(generic_plant.n_modes),
        "guarded_bindings": guarded,
        "rebuilds": 0,
        "tolerance": TOLERANCE,
        "generic_label": generic.label,
        "reconstruction_label": reconstruction.label,
        "generic_boundary": generic.prescribe(0).boundary_source,
        "reconstruction_boundary": reconstruction.prescribe(0).boundary_source,
        "coil_rows": [int(reconstruction.coils.time.size), int(generic.coils.time.size)],
        "section_plasma_flux": {
            key: value for key, value in sections.items() if key != "relative"
        },
        "mode_plasma_flux": {
            key: value for key, value in modes.items() if key != "relative"
        },
        "plasma_times_equal": bool(
            np.array_equal(generic_plant.plasma_times, reconstruction_plant.plasma_times)
        ),
        "figure": str(FIGURE),
        "plot": {
            "section": sections["relative"].ravel(),
            "mode": modes["relative"].ravel(),
        },
    }
    return report


def draw(report: dict, path: Path = FIGURE) -> None:
    """Draw each plant array's relative difference against the declared bound."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, (sections, modes) = plt.subplots(1, 2, figsize=(14, 7.2))
    for axis, values, label in (
        (sections, report["plot"]["section"], "winding sections"),
        (modes, report["plot"]["mode"], "passive modes"),
    ):
        axis.plot(np.arange(values.size), values, ".", color="#315f78")
        axis.set_yscale("log")
        axis.set_xlabel(label + " index")
        axis.set_ylabel("relative flux difference")
        axis.axhline(
            report["tolerance"], color="#666666", linewidth=1.2, linestyle=":"
        )
        axis.text(0, report["tolerance"], "1e-3 bound", color="#666666")
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(False)
    figure.suptitle("")
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, metadata={"Date": None})
    plt.close(figure)


def main() -> None:
    scratch = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("scratch")
    report = measure(scratch)
    draw(report)
    printable = {key: value for key, value in report.items() if key != "plot"}
    print("RECONSTRUCTION " + json.dumps(printable, sort_keys=True, default=float))
    if (
        report["section_plasma_flux"]["violations"]
        or report["mode_plasma_flux"]["violations"]
    ):
        raise AssertionError("reconstruction plant exceeds the returned tolerance")


if __name__ == "__main__":
    main()