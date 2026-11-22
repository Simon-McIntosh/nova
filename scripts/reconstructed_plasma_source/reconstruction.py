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
FIGURE_ONE_BOUNDARY = FIGURE.with_name("one-boundary-agreement.svg")


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


def _drive(scratch: Path, generic: PlasmaSource, reconstruction: PlasmaSource):
    """Build both plants on the cached machine with zero sectioned-set rebuilds.

    Every sectioned-set refinement and linkage build is refused for the
    duration, so a call raises rather than counting a quiet zero.
    """
    from nova.circuit.coupled_plant import coupled_plant
    from nova.circuit.passive import load_sectioned_machine

    scratch = Path(scratch)
    with rebuild_spy() as guarded:
        assert guarded, "no rebuild binding was guarded"
        machine = load_sectioned_machine(
            CACHE / "sectioned-conductor-set" / SECTIONED_KEY
        )
        baseline = coupled_plant(machine, generic, cache_dir=scratch / "generic")
        plant = coupled_plant(
            machine, reconstruction, cache_dir=scratch / "reconstruction"
        )
    return machine, baseline, plant, guarded


def one_boundary_sources() -> tuple[PlasmaSource, PlasmaSource]:
    """Present the same slice twice with the same boundary resolved.

    Both presentations drop the stored separatrix outline and every x-point, so
    both resolve the stored last closed flux surface and neither can differ from
    the other in the boundary it prescribes the current inside.  The one
    presentation difference left is the coil-current cadence: every row of the
    credited clock against every fourth.  The comparison therefore tests the
    cached machine and the presentation rather than the boundary choice.
    """
    generic = reconstruction_style(
        corsica_source(provenance="CORSICA 130506/6 as the generic path"), stride=1
    )
    reconstruction = reconstruction_style(
        corsica_source(provenance="CORSICA 130506/6 as a reconstruction")
    )
    return generic, reconstruction


def measure(scratch: Path, *, one_boundary: bool = False) -> dict:
    """Drive both presentations through the full plant and compare their flux.

    With ``one_boundary`` the second presentation resolves the same boundary as
    the first, so the comparison holds the boundary common and turns on the
    cached machine and the coil cadence alone.
    """
    if one_boundary:
        generic, reconstruction = one_boundary_sources()
        figure = FIGURE_ONE_BOUNDARY
    else:
        generic = corsica_source(provenance="CORSICA 130506/6")
        reconstruction = reconstruction_style(
            corsica_source(provenance="CORSICA 130506/6 as a reconstruction")
        )
        figure = FIGURE
    machine, generic_plant, reconstruction_plant, guarded = _drive(
        scratch, generic, reconstruction
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
        "one_boundary": one_boundary,
        "generic_boundary": generic.prescribe(0).boundary_source,
        "reconstruction_boundary": reconstruction.prescribe(0).boundary_source,
        "boundaries_agree": (
            generic.prescribe(0).boundary_source
            == reconstruction.prescribe(0).boundary_source
        ),
        "coil_rows": [
            int(reconstruction.coils.time.size),
            int(generic.coils.time.size),
        ],
        "section_plasma_flux": {
            key: value for key, value in sections.items() if key != "relative"
        },
        "mode_plasma_flux": {
            key: value for key, value in modes.items() if key != "relative"
        },
        "plasma_times_equal": bool(
            np.array_equal(
                generic_plant.plasma_times, reconstruction_plant.plasma_times
            )
        ),
        "figure": str(figure),
        "plot": {
            "section": sections["relative"].ravel(),
            "mode": modes["relative"].ravel(),
        },
    }
    if one_boundary and not report["boundaries_agree"]:
        raise AssertionError("the one-boundary arms resolved different boundaries")
    return report


def draw(report: dict, path: Path = FIGURE) -> None:
    """Draw each plant array's relative difference against the declared bound.

    Every sampled difference is drawn; the ones past the declared bound take the
    limit colour and the top layer, so the miss is visible without reading the
    numbers, and one shared legend names all three marks.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    plt.style.use("data-ink")
    bound = report["tolerance"]
    figure, (sections, modes) = plt.subplots(1, 2, figsize=(14, 7.2))
    for axis, values, label in (
        (sections, report["plot"]["section"], "winding section index"),
        (modes, report["plot"]["mode"], "passive mode index"),
    ):
        values = np.asarray(values, dtype=float)
        index = np.arange(values.size)
        within = values <= bound
        axis.plot(
            index[within],
            values[within],
            ".",
            color="#929292",
            markersize=4,
            zorder=4,
        )
        axis.plot(
            index[~within],
            values[~within],
            ".",
            color="#b2182b",
            markersize=5,
            zorder=5,
        )
        if values.size and not np.any(values > 0):
            # a log axis would carry no scale here, and an empty panel reads as
            # an absent measurement rather than as an exact one
            axis.set_ylim(-0.05 * bound, bound * 1.5)
        else:
            axis.set_yscale("log")
        axis.set_xlabel(label)
        axis.set_ylabel("relative flux difference")
        axis.axhline(
            bound,
            color="#52514e",
            linewidth=1.2,
            linestyle=(0, (1, 2)),
            zorder=1,
        )
    figure.legend(
        handles=[
            Line2D(
                [],
                [],
                marker=".",
                linestyle="",
                color="#929292",
                markersize=8,
                label="within the bound",
            ),
            Line2D(
                [],
                [],
                marker=".",
                linestyle="",
                color="#b2182b",
                markersize=8,
                label="past the bound",
            ),
            Line2D(
                [],
                [],
                color="#52514e",
                linewidth=1.2,
                linestyle=(0, (1, 2)),
                label="relative 1e-3 bound",
            ),
        ],
        loc="upper center",
        bbox_to_anchor=(0.5, 1.0),
        ncol=3,
        frameon=False,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.93))
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, metadata={"Date": None})
    plt.close(figure)


def _polygon_area(polygon: np.ndarray) -> float:
    x = np.asarray(polygon)[:, 0]
    z = np.asarray(polygon)[:, 1]
    return float(0.5 * abs(np.dot(x, np.roll(z, -1)) - np.dot(z, np.roll(x, -1))))


def _redistribution(generic, presented) -> dict:
    """How much of the prescribed current the two boundaries move."""
    density_difference = np.abs(presented.current_density - generic.current_density)
    cell_difference = np.abs(presented.cell_current - generic.cell_current)
    return {
        "boundary_source": [generic.boundary_source, presented.boundary_source],
        "boundary_points": [int(len(generic.boundary)), int(len(presented.boundary))],
        "boundary_area_m2": [
            _polygon_area(generic.boundary),
            _polygon_area(presented.boundary),
        ],
        "scale_factor": [generic.scale_factor, presented.scale_factor],
        "sum_abs_cell_current": float(np.abs(generic.cell_current).sum()),
        "sum_abs_cell_difference": float(cell_difference.sum()),
        "max_abs_cell_difference": float(cell_difference.max()),
        "redistributed_fraction": float(
            cell_difference.sum() / np.abs(generic.cell_current).sum()
        ),
        "max_abs_density_difference": float(density_difference.max()),
    }


def _violating_modes(generic_plant, plant) -> list[dict]:
    reference = np.asarray(generic_plant.mode_plasma_flux.ravel(), float)
    measured = np.asarray(plant.mode_plasma_flux.ravel(), float)
    relative = np.abs(measured - reference) / np.where(
        np.abs(reference) > 0, np.abs(reference), np.inf
    )
    rows = []
    for index in np.flatnonzero(relative > TOLERANCE):
        rows.append(
            {
                "mode": int(index),
                "tau_s": float(plant.mode_tau[index]),
                "reference_wb": float(reference[index]),
                "presented_wb": float(measured[index]),
                "abs_difference_wb": float(abs(measured[index] - reference[index])),
                "relative": float(relative[index]),
            }
        )
    rows.sort(key=lambda row: row["relative"], reverse=True)
    return rows


def localise(scratch: Path) -> dict:
    """Drive four presentations that differ from the generic one in one input.

    The generic slice, the coil clock alone coarsened, the boundary alone moved
    to the last closed flux surface, and both, each built on the same cached
    machine.  The arm that moves the modes names the input the tolerance
    responds to.
    """
    from nova.circuit.coupled_plant import coupled_plant
    from nova.circuit.passive import load_sectioned_machine

    scratch = Path(scratch)
    generic = corsica_source(provenance="CORSICA 130506/6")
    presented = reconstruction_style(
        corsica_source(provenance="CORSICA 130506/6 as a reconstruction")
    )
    arms = {
        "generic": generic,
        "coil_plasma_clock_only": replace(generic, coils=presented.coils),
        "last_closed_surface_only": replace(generic, slices=presented.slices),
        "both": presented,
    }
    machine = load_sectioned_machine(CACHE / "sectioned-conductor-set" / SECTIONED_KEY)
    baseline = coupled_plant(machine, generic, cache_dir=scratch / "arm-generic")
    table = {}
    for name, source in arms.items():
        plant = (
            baseline
            if name == "generic"
            else coupled_plant(machine, source, cache_dir=scratch / f"arm-{name}")
        )
        table[name] = {
            "section": _relative_error(
                baseline.section_plasma_flux, plant.section_plasma_flux
            ),
            "mode": _relative_error(baseline.mode_plasma_flux, plant.mode_plasma_flux),
        }
        table[name]["section"].pop("relative", None)
        table[name]["mode"].pop("relative", None)
    return {
        "machine": machine.label,
        "n_section": int(machine.section_inductance.shape[0]),
        "n_modes": int(baseline.n_modes),
        "redistribution": _redistribution(generic.prescribe(0), presented.prescribe(0)),
        "arms": table,
        "violating_modes": _violating_modes(
            baseline, coupled_plant(machine, presented, cache_dir=scratch / "arm-both")
        ),
        "tolerance": TOLERANCE,
    }


def _printable(report: dict) -> dict:
    return {key: value for key, value in report.items() if key != "plot"}


def main() -> None:
    scratch = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("scratch")
    arguments = sys.argv[2:]
    if "--localise" in arguments:
        report = localise(scratch)
        print(
            "LOCALISE " + json.dumps(_printable(report), sort_keys=True, default=float)
        )
        return
    one_boundary = "--one-boundary" in arguments
    report = measure(scratch, one_boundary=one_boundary)
    draw(report, Path(report["figure"]))
    print(
        ("ONE_BOUNDARY " if one_boundary else "RECONSTRUCTION ")
        + json.dumps(_printable(report), sort_keys=True, default=float)
    )
    if (
        report["section_plasma_flux"]["violations"]
        or report["mode_plasma_flux"]["violations"]
    ):
        raise AssertionError("reconstruction plant exceeds the returned tolerance")


if __name__ == "__main__":
    main()
