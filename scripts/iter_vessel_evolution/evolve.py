"""Evolve the ITER vessel currents through a DINA scenario.

The passive set of the sectioned conductor set is driven by DINA's coil currents
and the prescribed plasma, integrated with the exact zero-order-hold eigenmode
propagation of :mod:`nova.circuit.propagate`.  The evolved passive currents and
the voltages of the whole coupled flux are compared with DINA's own ``pf_passive``
and ``pf_active`` records.

The ITER windings are superconducting, so their winding resistance is zero and
the coil voltage is ``dPsi/dt`` alone -- DINA's stored zero coil resistance is
the physical value, not a gap in the data.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import imas
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "scripts"
CACHE = Path("/home/ITER/mcintos/Code/nova/.cache/iter-passive-plasma-circuit")
CONDUCTOR_CACHE = CACHE / "sectioned-conductor-set"
OUTPUT = CACHE / "vessel-evolution"
DD_VERSION = "3.38.1"

#: cache key per surface-layer: the sectioned conductor set's two meshes
MESH_KEY = {
    0.012: "68df15665ab2ea45f427",  # 1312 passive circuits
    0.006: "06ff06d3eb7a4448f0fe",  # 2163 passive circuits
}

#: nova's 14 coil channels in the cached system's column order
NOVA_CHANNELS = [
    "CS3U", "CS2U", "CS1U", "CS1L", "CS2L", "CS3L",
    "PF1", "PF2", "PF3", "PF4", "PF5", "PF6", "VSU", "VSL",
]

#: DINA coil name -> nova channels it energises.  CS1U and CS1L are in series on
#: one supply (equal currents); VSU and VSL are one circuit in anti-series
#: (equal and opposite), the sign carried by the drive.
DINA_TO_NOVA = {
    "CS3U": ("CS3U",), "CS2U": ("CS2U",), "CS2L": ("CS2L",), "CS3L": ("CS3L",),
    "PF1": ("PF1",), "PF2": ("PF2",), "PF3": ("PF3",), "PF4": ("PF4",),
    "PF5": ("PF5",), "PF6": ("PF6",),
    "CS1": ("CS1U", "CS1L"), "VS3": ("VSU", "VSL"),
}

#: how many DINA coils a nova channel's voltage is summed into for comparison
NOVA_TO_DINA = {
    "CS1U": "CS1", "CS1L": "CS1", "VSU": "VS3", "VSL": "VS3",
    **{
        name: name
        for name in NOVA_CHANNELS
        if name not in ("CS1U", "CS1L", "VSU", "VSL")
    },
}


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def build_module():
    return _load_module(
        "iter_sectioned_build", SCRIPTS / "iter_sectioned_conductor_set/build.py"
    )


def plasma_module():
    return _load_module(
        "iter_prescribed_measure", SCRIPTS / "iter_prescribed_plasma/measure.py"
    )


def conductor_geometry(surface_layer: float):
    """Rebuild the refined conductor set so its geometry and parent map are in hand."""
    from nova.circuit.conductor import subdivide_passive_thickness

    build = build_module()
    conductors, coil_names, owner, thickness, source_resistance, passive_name = (
        build.source_conductors()
    )
    n_section = len(owner)
    source_passive = np.arange(n_section, n_section + len(thickness))
    refined = subdivide_passive_thickness(
        conductors, source_passive, thickness, 0.04, surface_layer_m=surface_layer
    )
    return conductors, coil_names, owner, refined


def load_system(surface_layer: float):
    """Load the cached circuit system and its full linkage matrix."""
    from nova.circuit.passive import load_circuit_system

    directory = CONDUCTOR_CACHE / MESH_KEY[surface_layer]
    matrix = np.load(directory / "linkage.npy")
    system = load_circuit_system(directory / "system.npz")
    return matrix, system


def read_case(pulse: int):
    uri = f"imas:hdf5?user=mcintos;pulse={pulse};run=1;database=iter;version=3"
    with imas.DBEntry(uri, "r", dd_version=DD_VERSION) as entry:
        equilibrium = entry.get("equilibrium", lazy=False, autoconvert=False)
        active = entry.get("pf_active", lazy=False, autoconvert=False)
        passive = entry.get("pf_passive", lazy=False, autoconvert=False)
    return equilibrium, active, passive


def coil_currents(active, vsl_sign: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """DINA coil currents and voltages on nova's 14 channels.

    CS1U and CS1L carry the CS1 current.  VSU carries the VS3 current and VSL
    carries ``vsl_sign`` times it (``-1`` for the anti-series winding).
    """
    name_index = {
        str(coil.name).strip(): index for index, coil in enumerate(active.coil)
    }
    times = np.asarray(active.time, dtype=np.float64)
    currents = np.zeros((times.size, len(NOVA_CHANNELS)))
    voltages = np.zeros((times.size, len(NOVA_CHANNELS)))
    for dina_name, targets in DINA_TO_NOVA.items():
        coil = active.coil[name_index[dina_name]]
        current = np.asarray(coil.current.data, dtype=np.float64)
        voltage = np.asarray(coil.voltage.data, dtype=np.float64)
        for target in targets:
            sign = vsl_sign if target == "VSL" else 1.0
            column = NOVA_CHANNELS.index(target)
            currents[:, column] = sign * current
            voltages[:, column] = sign * voltage
    return times, currents, voltages


def series_matrix(owner: np.ndarray) -> np.ndarray:
    """One-hot map from the 295 winding sections to nova's 14 coil channels."""
    series = np.zeros((len(owner), len(NOVA_CHANNELS)))
    series[np.arange(len(owner)), owner] = 1.0
    return series


def dina_passive(passive):
    times = np.asarray(passive.time, dtype=np.float64)
    names = [str(loop.name).strip() for loop in passive.loop]
    data = np.stack(
        [np.asarray(loop.current.data, dtype=np.float64) for loop in passive.loop],
        axis=1,
    )
    return times, names, data


def vs3_points(active) -> np.ndarray:
    """Centroids of the VS3 coil's elements, where the psi-map check left a residual."""
    from nova.imas.machine import CrossSection

    name_index = {
        str(coil.name).strip(): index for index, coil in enumerate(active.coil)
    }
    coil = active.coil[name_index["VS3"]]
    points = []
    for element in coil.element:
        centroid = CrossSection(element.geometry).data.poly.centroid
        points.append([float(centroid.x), float(centroid.y)])
    return np.asarray(points)


def prescribed_vs3_history(pulse: int, n_points: int, times: np.ndarray):
    """The psi-map residual the prescribed-plasma node left at VS3, on ``times``.

    Returns ``(residual, tolerance_wb)``: ``residual`` is ``(n_times, n_points)``
    in the VS3 element order, interpolated from the check's sampled slices, and
    the tolerance is the loosest per-row bound the check applied.
    """
    summary = CACHE / "gate-logs/iter-prescribed-plasma" / f"summary-{pulse}.json"
    data = json.loads(summary.read_text())
    rows_by_point: dict[int, list[dict]] = {}
    tolerance = 0.0
    for row in data["linkage"]:
        if row.get("category") != "coil":
            continue
        if str(row.get("name")).strip() != "VS3":
            continue
        rows_by_point.setdefault(int(row["target_index"]), []).append(row)
        tolerance = max(tolerance, float(row["tolerance_wb"]))
    ordered = [sorted(rows_by_point[key], key=lambda row: row["time"])
               for key in sorted(rows_by_point)]
    if len(ordered) != n_points:
        raise ValueError(
            f"prescribed-plasma check has {len(ordered)} VS3 points, "
            f"geometry has {n_points}"
        )
    residual = np.zeros((times.size, n_points))
    for index, rows in enumerate(ordered):
        sample_times = np.array([float(row["time"]) for row in rows])
        sample_values = np.array([float(row["residual_wb"]) for row in rows])
        residual[:, index] = np.interp(times, sample_times, sample_values)
    return residual, tolerance


def dina_passive_flux(active, passive, measure, points, times):
    """The psi-map check's *passive* source term at ``points``, from DINA's currents.

    Section 2 summed one source over coils and passive loops together; the
    evolved flux is passive-only, so the same kernel is restricted to its
    passive columns before comparison.
    """
    kernel, owners = measure.conductor_sources(
        active, passive, points[:, 0], points[:, 1]
    )
    mask = np.array([category == "passive" for category, _ in owners])
    flux = np.zeros((times.size, points.shape[0]))
    for step, time in enumerate(times):
        currents = measure.current_vector(active, passive, owners, float(time))
        flux[step] = kernel[:, mask] @ currents[mask]
    return flux


def coverage_mask(active, passive, times):
    """Keep only the times both the coil and passive records cover."""
    low = max(float(np.min(active.time)), float(np.min(passive.time)))
    high = min(float(np.max(active.time)), float(np.max(passive.time)))
    return (times >= low) & (times <= high)


def interp_to(times, target, values):
    return np.column_stack(
        [
            np.interp(target, times, values[:, column])
            for column in range(values.shape[1])
        ]
    )


def integrate(tau, vectors, times, psi_circuit):
    from nova.circuit.propagate import integrate_eddy_ode

    state, _swing = integrate_eddy_ode(tau, times, psi_circuit @ vectors)
    return state @ vectors.T


def psis_and_voltages(
    times, series, matrix, i_channel, i_passive, psi_plasma_section
):
    """Section-level coupled flux and the per-channel voltage of the whole flux."""
    n_section = series.shape[0]
    Lcc = matrix[:n_section, :n_section]
    Mcp = matrix[:n_section, n_section:]
    i_section = i_channel @ series.T
    psi_section = i_section @ Lcc.T + i_passive @ Mcp.T + psi_plasma_section
    voltage_section = np.gradient(psi_section, times, axis=0)
    psi_coil = psi_section @ series
    voltage_coil = voltage_section @ series
    return psi_coil, voltage_coil


def per_coil_voltage_residual(
    times, nova_voltage, dina_voltage
) -> dict:
    """Rms and peak residual per DINA coil, nova channels summed into it."""
    channels = {}
    for column, name in enumerate(NOVA_CHANNELS):
        channels.setdefault(NOVA_TO_DINA[name], []).append(column)
    rows = {}
    for dina_name, columns in channels.items():
        nova = nova_voltage[:, columns].sum(axis=1)
        reference = dina_voltage[:, columns[0]]
        residual = nova - reference
        rows[dina_name] = {
            "rms_v": float(np.sqrt(np.mean(residual**2))),
            "peak_v": float(np.max(np.abs(residual))),
            "dina_peak_v": float(np.max(np.abs(reference))),
        }
    rms = np.array([row["rms_v"] for row in rows.values()])
    peak = np.array([row["peak_v"] for row in rows.values()])
    return {
        "coils": rows,
        "rms_mean_v": float(rms.mean()),
        "rms_max_v": float(rms.max()),
        "peak_max_v": float(peak.max()),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pulse", type=int, choices=(105014, 105052), required=True)
    parser.add_argument("--surface-layer", type=float, default=0.012)
    parser.add_argument("--plasma-slices", type=int, default=24)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    analysis = _load_module(
        "iter_vessel_measure", Path(__file__).resolve().parent / "measure.py"
    )
    measure = plasma_module()

    matrix, system = load_system(args.surface_layer)
    conductors, coil_names, owner, refined = conductor_geometry(args.surface_layer)
    if list(coil_names) != NOVA_CHANNELS:
        raise ValueError("cached channels disagree with the sectioned coil order")
    if not np.array_equal(refined.circuits, system.circuits):
        raise ValueError("rebuilt conductor set disagrees with the cached circuits")
    series = series_matrix(owner)
    equilibrium, active, passive = read_case(args.pulse)
    centroids = np.column_stack([system.centroid_r, system.centroid_z])
    section_centroids = np.column_stack(
        refined.conductors.centroids(np.arange(len(owner)))
    )

    slice_times, plasma_flux = analysis.plasma_flux_history(
        equilibrium, measure, centroids, args.plasma_slices
    )
    section_slice_times, section_plasma = analysis.plasma_flux_history(
        equilibrium, measure, section_centroids, args.plasma_slices
    )

    tau, vectors = system.mode_system()
    result = {"pulse": args.pulse, "surface_layer_m": args.surface_layer}
    variants = {}
    for vsl_sign in (-1.0, 1.0):
        times, i_channel, dina_voltage = coil_currents(active, vsl_sign)
        keep = coverage_mask(active, passive, times)
        times, i_channel, dina_voltage = (
            times[keep], i_channel[keep], dina_voltage[keep]
        )
        psi_plasma = interp_to(slice_times, times, plasma_flux)
        psi_plasma_section = interp_to(section_slice_times, times, section_plasma)
        drive = i_channel @ system.m_channel.T
        i_passive = integrate(tau, vectors, times, drive + psi_plasma)
        i_noplasma = integrate(tau, vectors, times, drive)

        _, v_full = psis_and_voltages(
            times, series, matrix, i_channel, i_passive, psi_plasma_section
        )
        _, v_passive = psis_and_voltages(
            times, series, matrix, i_channel, i_passive, 0.0 * psi_plasma_section
        )
        _, v_noplasma = psis_and_voltages(
            times, series, matrix, i_channel, i_noplasma, 0.0 * psi_plasma_section
        )
        _, v_bare = psis_and_voltages(
            times, series, matrix, i_channel, 0.0 * i_passive, 0.0 * psi_plasma_section
        )
        variants[vsl_sign] = {
            "times": times,
            "i_channel": i_channel,
            "i_passive": i_passive,
            "i_noplasma": i_noplasma,
            "dina_voltage": dina_voltage,
            "full": per_coil_voltage_residual(times, v_full, dina_voltage),
            "passive": per_coil_voltage_residual(times, v_passive, dina_voltage),
            "noplasma": per_coil_voltage_residual(times, v_noplasma, dina_voltage),
            "bare": per_coil_voltage_residual(times, v_bare, dina_voltage),
        }
    best = min(variants, key=lambda key: variants[key]["full"]["rms_mean_v"])
    chosen = variants[best]
    times = chosen["times"]

    dina_times, loop_names, dina_currents = dina_passive(passive)
    loop_polys = analysis.dina_loop_polygons(passive)
    comparison = analysis.group_comparison(
        times,
        chosen["i_passive"],
        centroids,
        loop_polys,
        loop_names,
        dina_times,
        dina_currents,
        absolute_a=10e3,
        relative=0.25,
    )
    vs3_centroids = vs3_points(active)
    section2_residual, section2_tolerance = prescribed_vs3_history(
        args.pulse, vs3_centroids.shape[0], times
    )
    vs3 = analysis.vs3_residual(
        times,
        section2_residual,
        analysis.passive_flux_at_points(
            refined.conductors, system.circuits, chosen["i_passive"], times,
            vs3_centroids,
        ),
        dina_passive_flux(active, passive, measure, vs3_centroids, times),
        tolerance_wb=section2_tolerance,
    )
    vs3_noplasma = analysis.vs3_residual(
        times,
        section2_residual,
        analysis.passive_flux_at_points(
            refined.conductors, system.circuits, chosen["i_noplasma"], times,
            vs3_centroids,
        ),
        dina_passive_flux(active, passive, measure, vs3_centroids, times),
        tolerance_wb=section2_tolerance,
    )

    result.update(
        {
            "circuits": int(system.n_circuits),
            "coil_sections": len(owner),
            "times": int(times.size),
            "plasma_slices": int(slice_times.size),
            "passive_current_peak_a": float(np.abs(chosen["i_passive"]).max()),
            "passive_current_rms_a": float(
                np.sqrt(np.mean(chosen["i_passive"] ** 2))
            ),
            "plasma_flux_peak_wb": float(np.abs(plasma_flux).max()),
            "positive_control": analysis.positive_control(),
            "vsl_sign_test": {
                str(key): {
                    "full_rms_mean_v": variants[key]["full"]["rms_mean_v"],
                    "bare_rms_mean_v": variants[key]["bare"]["rms_mean_v"],
                }
                for key in variants
            },
            "vsl_sign_chosen": best,
            "voltage_bare": chosen["bare"],
            "voltage_passive": chosen["passive"],
            "voltage_noplasma": chosen["noplasma"],
            "voltage_full": chosen["full"],
            "group_comparison": comparison,
            "vs3": vs3,
            "vs3_noplasma": vs3_noplasma,
        }
    )

    OUTPUT.mkdir(parents=True, exist_ok=True)
    np.savez(
        OUTPUT / f"evolved-{args.pulse}-{int(args.surface_layer * 1000):03d}.npz",
        times=times,
        i_passive=chosen["i_passive"],
        i_noplasma=chosen["i_noplasma"],
    )
    if args.output:
        args.output.write_text(json.dumps(result, indent=2, default=float))
    print("SUMMARY " + json.dumps(result, sort_keys=True, default=float))


if __name__ == "__main__":
    main()