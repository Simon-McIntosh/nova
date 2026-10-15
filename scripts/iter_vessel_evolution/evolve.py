"""Evolve the ITER vessel currents through a DINA scenario.

The passive set of the sectioned conductor set (``iter_sectioned_conductor_set``)
is driven by DINA's coil currents and the prescribed plasma of
``iter_prescribed_plasma``, integrated with the exact zero-order-hold eigenmode
propagation of ``nova.circuit.propagate``.  The evolved passive currents and the
coupled coil voltages are compared with DINA's own ``pf_passive`` and
``pf_active`` records.
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
    "CS3U",
    "CS2U",
    "CS1U",
    "CS1L",
    "CS2L",
    "CS3L",
    "PF1",
    "PF2",
    "PF3",
    "PF4",
    "PF5",
    "PF6",
    "VSU",
    "VSL",
]

#: DINA coil name -> (nova channel, split) : CS1 and VS3 are stored merged in
#: the DINA case and split into upper/lower halves in the machine description.
DINA_TO_NOVA = {
    "CS3U": ("CS3U",),
    "CS2U": ("CS2U",),
    "CS2L": ("CS2L",),
    "CS3L": ("CS3L",),
    "PF1": ("PF1",),
    "PF2": ("PF2",),
    "PF3": ("PF3",),
    "PF4": ("PF4",),
    "PF5": ("PF5",),
    "PF6": ("PF6",),
    "CS1": ("CS1U", "CS1L"),
    "VS3": ("VSU", "VSL"),
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
    """Rebuild §1's refined conductor set so its geometry and parent map are in hand."""
    from nova.circuit.conductor import subdivide_passive_thickness

    build = build_module()
    conductors, coil_names, owner, thickness, source_resistance, passive_name = (
        build.source_conductors()
    )
    # source passive circuits follow the winding sections, in order
    n_section = len(owner)
    source_passive = np.arange(n_section, n_section + len(thickness))
    refined = subdivide_passive_thickness(
        conductors, source_passive, thickness, 0.04, surface_layer_m=surface_layer
    )
    return (
        conductors,
        coil_names,
        owner,
        thickness,
        source_resistance,
        passive_name,
        refined,
    )


def load_system(surface_layer: float):
    """Load §1's cached circuit system and its full linkage matrix."""
    from nova.circuit.passive import load_circuit_system

    key = MESH_KEY[surface_layer]
    directory = CONDUCTOR_CACHE / key
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


def coil_current_matrix(active) -> tuple[np.ndarray, list[str]]:
    """DINA coil currents on nova's 14 channels, in DINA's own time base."""
    name_index = {
        str(coil.name).strip(): index for index, coil in enumerate(active.coil)
    }
    times = np.asarray(active.time, dtype=np.float64)
    currents = np.zeros((times.size, len(NOVA_CHANNELS)))
    for dina_name, targets in DINA_TO_NOVA.items():
        coil = active.coil[name_index[dina_name]]
        signal = np.asarray(coil.current.data, dtype=np.float64)
        for target in targets:
            currents[:, NOVA_CHANNELS.index(target)] = signal
    return times, currents


def dina_passive(active_passive):
    """DINA pf_passive loop currents and names on their own time base."""
    times = np.asarray(active_passive.time, dtype=np.float64)
    names = [str(loop.name).strip() for loop in active_passive.loop]
    data = np.stack(
        [
            np.asarray(loop.current.data, dtype=np.float64)
            for loop in active_passive.loop
        ],
        axis=1,
    )
    return times, names, data


def plasma_flux_history(equilibrium, centroids, n_slices: int, measure):
    """Prescribed plasma flux [Wb] per passive circuit at subsampled slice times."""
    eligible = [
        index
        for index, slice_ in enumerate(equilibrium.time_slice)
        if abs(float(slice_.global_quantities.ip)) >= 1e5
    ]
    if not eligible:
        raise ValueError("no plasma-bearing equilibrium slices")
    selected = np.unique(np.linspace(eligible[0], eligible[-1], n_slices, dtype=int))
    slice_times = np.array(
        [float(equilibrium.time_slice[index].time) for index in selected]
    )
    flux = np.zeros((slice_times.size, centroids.shape[0]))
    for row, index in enumerate(selected):
        current = measure.prescribe_slice(equilibrium.time_slice[index])
        flux[row] = measure_flux(current, centroids)
    return slice_times, flux


def measure_flux(current, centroids):
    from nova.imas.prescribed_plasma import plasma_flux_at_points

    return plasma_flux_at_points(current, centroids[:, 0], centroids[:, 1])


def interp_to(times, target, values):
    return np.column_stack(
        [
            np.interp(target, times, values[:, column])
            for column in range(values.shape[1])
        ]
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pulse", type=int, choices=(105014, 105052), required=True)
    parser.add_argument("--surface-layer", type=float, default=0.012)
    parser.add_argument("--plasma-slices", type=int, default=24)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    from nova.circuit.propagate import integrate_eddy_ode

    measure = plasma_module()
    analysis = _load_module(
        "iter_vessel_measure", Path(__file__).resolve().parent / "measure.py"
    )
    matrix, system = load_system(args.surface_layer)
    equilibrium, active, passive = read_case(args.pulse)
    times, i_channel = coil_current_matrix(active)
    centroids = np.column_stack([system.centroid_r, system.centroid_z])
    slice_times, flux = plasma_flux_history(
        equilibrium, centroids, args.plasma_slices, measure
    )
    psi_plasma = interp_to(slice_times, times, flux)

    tau, vectors = system.mode_system()
    psi_drive = i_channel @ system.m_channel.T
    psi_mode = (psi_drive + psi_plasma) @ vectors
    a, _ = integrate_eddy_ode(tau, times, psi_mode)
    i_passive = a @ vectors.T

    a_noplasma, _ = integrate_eddy_ode(tau, times, psi_drive @ vectors)
    i_noplasma = a_noplasma @ vectors.T

    control = analysis.positive_control()
    dina_times, loop_names, dina_currents = dina_passive(passive)
    loop_centroids, _names = analysis.dina_loop_centroids(passive)
    comparison = analysis.group_comparison(
        times,
        i_passive,
        centroids,
        loop_centroids,
        loop_names,
        dina_times,
        dina_currents,
        absolute_a=10e3,
        relative=0.25,
    )

    result = {
        "pulse": args.pulse,
        "surface_layer_m": args.surface_layer,
        "circuits": int(system.n_circuits),
        "coil_sections": int(matrix.shape[0] - system.n_circuits),
        "times": int(times.size),
        "plasma_slices": int(slice_times.size),
        "channel_current_peak": float(np.abs(i_channel).max()),
        "passive_current_peak": float(np.abs(i_passive).max()),
        "passive_current_rms": float(np.sqrt(np.mean(i_passive**2))),
        "plasma_flux_peak": float(np.abs(psi_plasma).max()),
        "positive_control": control,
        "group_comparison": comparison,
    }
    OUTPUT.mkdir(parents=True, exist_ok=True)
    np.savez(
        OUTPUT / f"evolved-{args.pulse}-{int(args.surface_layer * 1000):03d}.npz",
        times=times,
        i_channel=i_channel,
        i_passive=i_passive,
        i_noplasma=i_noplasma,
        psi_plasma=psi_plasma,
    )
    if args.output:
        args.output.write_text(json.dumps(result, indent=2))
    print("POSITIVE_CONTROL " + json.dumps(control, sort_keys=True))
    print(
        "GROUP_COMPARISON "
        + json.dumps(
            {
                k: result["group_comparison"][k]
                for k in result["group_comparison"]
                if k != "rows"
            },
            sort_keys=True,
        )
    )
    print(
        "SUMMARY "
        + json.dumps(
            {k: v for k, v in result.items() if k not in ("group_comparison",)},
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
