"""Locate pointwise source-cutoff crossings in the saved direction sweep.

The structured DIII-D builder supplies value-only coupling matrices. Its
five-point cell-average rule has centre weight 5/6 and neighbour weight 1/24.
This independent calculation predicts the finite current jump from the
recorded edge profiles and grid spacing, without evaluating another solve.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.constants import mu_0

from verify import ROWS


def analyze(folder):
    spatial = json.loads((folder / "spatial.json").read_text())
    proposals = json.loads((folder / "proposals.json").read_text())
    proposal = dict(np.load(folder / f"proposal-{proposals[-1]['index']}.npz"))
    data = dict(np.load(folder / "terminal.npz"))
    seed = np.load(folder / "request-seed.npz")["state"]
    scale = np.max(np.abs(seed))
    initial = np.load(folder / "inner-0.npz")["initial"][: seed.size] * scale
    np.testing.assert_allclose(initial, seed, rtol=2e-15, atol=2e-15)
    coordinates = np.load(folder / "spatial-marks.npz")["coordinates"]
    count = len(coordinates)
    tables = json.loads((folder / "source-inputs.json").read_text())
    radius = coordinates[:, 0]
    edge_density = (
        -2
        * np.pi
        * (
            radius * tables["p_prime"]["values"][-1]
            + tables["ff_prime"]["values"][-1] / (mu_0 * radius)
        )
    )
    area = np.diff(data["radius"])[0] * np.diff(data["height"])[0]
    sweep = []
    for row in spatial["directional_discriminator"]:
        factor = row["factor"]
        arrays = dict(np.load(folder / f"discriminator-{factor}.npz"))
        flux = (proposal["state"][:count] + factor * proposal["step"][:count]) * scale
        read = row["reading"]
        norm = (flux - read["axis_flux_wb"]) / (
            read["boundary_flux_wb"] - read["axis_flux_wb"]
        )
        gate = (norm <= 1) & arrays["participation"]
        sweep.append((row, arrays, norm, gate))
    pairs = []
    for large, small in zip(sweep[:-1], sweep[1:], strict=True):
        high, high_arrays, high_norm, high_gate = large
        low, low_arrays, low_norm, low_gate = small
        crossing = high_gate != low_gate
        if not crossing.any():
            continue
        sign = high_gate.astype(int) - low_gate.astype(int)
        point_jump = (edge_density * sign).reshape(data["flux"].shape)
        predicted = point_jump.copy()
        predicted[1:-1, 1:-1] = (5 / 6) * point_jump[1:-1, 1:-1] + (
            point_jump[:-2, 1:-1]
            + point_jump[2:, 1:-1]
            + point_jump[1:-1, :-2]
            + point_jump[1:-1, 2:]
        ) / 24
        predicted = predicted.ravel() * area * high_arrays["participation"]
        measured = high_arrays["current"] - low_arrays["current"]
        affected = np.flatnonzero(predicted)
        pairs.append(
            dict(
                larger_factor=high["factor"],
                smaller_factor=low["factor"],
                source_gate_changes=int(crossing.sum()),
                connectivity_changes=int(
                    np.sum(high_arrays["connected"] != low_arrays["connected"])
                ),
                participation_changes=int(
                    np.sum(high_arrays["participation"] != low_arrays["participation"])
                ),
                residual_shadow_changes=int(
                    np.sum(high_arrays["shadow"] != low_arrays["shadow"])
                ),
                boundary_class_large=high["reading"]["boundary_class"],
                boundary_class_small=low["reading"]["boundary_class"],
                crossing_rz_m=coordinates[crossing].tolist(),
                psi_norm_large=high_norm[crossing].tolist(),
                psi_norm_small=low_norm[crossing].tolist(),
                predicted_jump_a=float(predicted.sum()),
                measured_jump_a=float(measured.sum()),
                measured_l2_a=float(np.linalg.norm(measured)),
                prediction_error_l2_a=float(np.linalg.norm(measured - predicted)),
                affected_rz_m=coordinates[affected].tolist(),
                predicted_cell_jump_a=predicted[affected].tolist(),
                measured_cell_jump_a=measured[affected].tolist(),
            )
        )
    finest, finest_arrays, finest_norm, _ = sweep[-1]
    eligible = np.flatnonzero(finest_arrays["participation"])
    closest = eligible[np.argmin(np.abs(finest_norm[eligible] - 1))]
    result = dict(
        cell_area_m2=float(area),
        pairs=pairs,
        smallest_crossing_pair=pairs[-1] if pairs else None,
        finest_step_factor=finest["factor"],
        closest_source_sample_rz_m=coordinates[closest].tolist(),
        closest_source_sample_psi_norm=float(finest_norm[closest]),
    )
    (folder / "cutoff-jumps.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    args = parser.parse_args()
    for frame, branch in ROWS:
        folder = args.input / "edge" / f"frame-{frame}-{branch}"
        result = analyze(folder)
        print(
            "CUTOFF_JUMP "
            + json.dumps(
                dict(
                    frame=frame,
                    branch=branch,
                    smallest_crossing_pair=result["smallest_crossing_pair"],
                )
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
