"""Attribute the MAST core change and locate source and map current peaks."""

from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
import subprocess

import jax
import jax.numpy as jnp
import numpy as np
from scipy.signal import find_peaks

from nova.equilibrium import solve_request
from nova.jax.config import configure_dtypes


def peak_locations(radius, values, selected):
    """Resolve peaks with prominence at least one percent of the selected maximum."""
    selected = np.asarray(selected, bool) & np.isfinite(values)
    indices = np.flatnonzero(selected)
    if not len(indices):
        raise ValueError("peak measurement has no selected current samples")
    values = np.asarray(values)[indices]
    peaks, info = find_peaks(values, prominence=0.01 * np.max(values))
    return [
        dict(
            index=int(indices[index]),
            radius_m=float(radius[indices[index]]),
            density_ma_per_m2=float(values[index] / 1e6),
            prominence_ma_per_m2=float(prominence / 1e6),
        )
        for index, prominence in zip(peaks, info["prominences"], strict=True)
    ]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    configure_dtypes()
    assert jax.config.jax_enable_x64
    analyses = {}
    for arm in ("base", "head"):
        folder = args.input / arm / "mast-21978-35"
        analyses[arm] = json.loads((folder / "core-analysis.json").read_text())
    assert analyses["base"]["source_inputs"] == analyses["head"]["source_inputs"]
    assert analyses["base"]["profiles"] == analyses["head"]["profiles"]
    for field in ("source_inputs", "profiles"):
        for name in ("p_prime", "ff_prime"):
            for key in ("coordinate", "values"):
                before = np.asarray(analyses["base"][field][name][key])
                after = np.asarray(analyses["head"][field][name][key])
                np.testing.assert_array_equal(
                    before.view(np.uint64), after.view(np.uint64)
                )

    root = Path(__file__).resolve().parents[3]
    baseline_source = subprocess.check_output(
        [
            "git",
            "show",
            "71d06e032b73106c56386ae3a64fa30605571102:nova/equilibrium/solve_request.py",
        ],
        cwd=root,
        text=True,
    )
    node = next(
        node
        for node in ast.parse(baseline_source).body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_evaluate_sampled_flux_function"
    )
    namespace = dict(vars(solve_request))
    exec(
        compile(ast.Module(body=[node], type_ignores=[]), "baseline-profile", "exec"),
        namespace,
    )
    baseline = jax.jit(namespace["_evaluate_sampled_flux_function"])
    held = jax.jit(solve_request._evaluate_sampled_flux_function)
    coordinates = [0.0, 0.02, 1.0, 1.01, 1.02]
    for analysis in analyses.values():
        coordinates.extend(analysis["midplane"]["source"]["psi_norm"])
        coordinates.extend(
            value
            for record in analysis["negative_source_evaluations"]
            for value in record["psi_norm"]
        )
    coordinate = np.asarray(coordinates)
    lower = coordinate < 0
    assert lower.any() and (~lower).sum() > 100
    function_identity = {}
    for name, data in analyses["base"]["source_inputs"].items():
        profile = solve_request.SampledFluxFunction(data["coordinate"], data["values"])
        leaves = profile.tree_flatten()[0]
        before = np.asarray(baseline(*leaves, jnp.asarray(coordinate)))
        after = np.asarray(held(*leaves, jnp.asarray(coordinate)))
        different = before.view(np.uint64) != after.view(np.uint64)
        assert not different[~lower].any()
        assert different[lower].any()
        assert np.all(after[lower] == data["values"][0])
        function_identity[name] = dict(
            common_coordinates=len(coordinate),
            negative_coordinates=int(lower.sum()),
            changed_negative_coordinates=int(different[lower].sum()),
            changed_nonnegative_coordinates=int(different[~lower].sum()),
            max_abs_nonnegative_difference=float(
                np.max(np.abs(after[~lower] - before[~lower]))
            ),
            held_value=float(data["values"][0]),
        )

    result = dict(
        inputs_bit_identical=True, function_identity=function_identity, arms={}
    )
    for arm, analysis in analyses.items():
        cut = analysis["midplane"]
        radius = np.asarray(cut["radius"])
        selected = np.asarray(cut["source"]["support"], bool)
        pressure = analysis["amplitude"] * np.asarray(cut["source"]["pressure_term"])
        field = analysis["amplitude"] * np.asarray(cut["source"]["field_term"])
        total = np.asarray(cut["source"]["density"])
        np.testing.assert_allclose(
            (pressure + field)[selected], total[selected], rtol=2e-13, atol=1e-8
        )
        source_peaks = peak_locations(radius, total, selected)
        terms = []
        for peak in source_peaks:
            i = peak["index"]
            terms.append(
                dict(
                    radius_m=peak["radius_m"],
                    pressure_ma_per_m2=float(pressure[i] / 1e6),
                    field_ma_per_m2=float(field[i] / 1e6),
                    sum_ma_per_m2=float(total[i] / 1e6),
                )
            )
        records = analysis["negative_source_evaluations"]
        locations = sorted(
            {tuple(point) for record in records for point in record["rz_m"]}
        )
        result["arms"][arm] = dict(
            source_peaks=source_peaks,
            source_peak_components=terms,
            map_peaks=peak_locations(radius, np.asarray(cut["map_density"]), selected),
            efit_peaks=peak_locations(
                radius,
                np.asarray(cut["efit_density"]),
                np.asarray(cut["efit_support"], bool),
            ),
            negative_source_calls=len(records),
            negative_source_node_evaluations=sum(record["count"] for record in records),
            unique_negative_nodes=len(locations),
            negative_rz_m=locations,
            terminal_negative_nodes=analysis["terminal_negative_nodes"],
            amplitude=float(analysis["amplitude"]),
        )
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print("MAST_ATTRIBUTION " + json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
