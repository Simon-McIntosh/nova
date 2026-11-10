"""Refresh lower-root diagnostics on saved states without building a forward map."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--row", type=Path, required=True)
    args = parser.parse_args()
    import jax
    import jax.numpy as jnp
    import numpy as np

    from nova.equilibrium.flux_surface_connectivity import polish_stationary_points
    from nova.jax.config import configure_dtypes
    from nova.linalg.tensor_spline import fit_tensor_spline
    from nova.media.poloidal import inside_wall_units

    configure_dtypes()
    assert jax.config.jax_enable_x64
    record = json.loads((args.row / "receipt.json").read_text())
    with np.load(args.row / "terminal.npz") as archive:
        data = {key: archive[key] for key in archive.files}
    state_hash = hashlib.sha256(data["state"].tobytes()).hexdigest()
    count = data["flux"].size
    reference = np.asarray(record["efit_x_point_rz_m"])
    terminal_trip = record["inner"][-1]["index"]
    with np.load(args.row / f"inner-{terminal_trip}.npz") as archive:
        normalized = archive["state"][: data["state"].size]
        scale = np.dot(data["state"], normalized) / np.dot(normalized, normalized)

    @jax.jit
    def roots(state, starts):
        field = fit_tensor_spline(
            jnp.asarray(data["radius"]),
            jnp.asarray(data["height"]),
            state[:count].reshape(data["flux"].shape).T,
        )
        return polish_stationary_points(field, starts, jnp.ones(2, bool))

    observations = []

    def refresh(topology, state):
        starts = np.vstack((reference, topology["selected_saddle_rz_m"]))
        found = jax.device_get(roots(jnp.asarray(state), jnp.asarray(starts)))
        positions = np.asarray(found["position_rz"])
        valid = (
            np.asarray(found["converged"])
            & (np.asarray(found["hessian_type"]) == -1)
            & (positions[:, 1] < min(data["reference_axis"][1], data["axis"][1]))
        )
        distance = np.linalg.norm(positions - reference, axis=1)
        index = int(np.argmin(np.where(valid, distance, np.inf)))
        print(
            "LOWER_ROOT_PROBE "
            + json.dumps(
                dict(
                    starts=starts.tolist(),
                    positions=positions.tolist(),
                    converged=np.asarray(found["converged"]).tolist(),
                    gradient_norm=np.asarray(found["gradient_norm"]).tolist(),
                    hessian_type=np.asarray(found["hessian_type"]).tolist(),
                )
            ),
            flush=True,
        )
        assert valid.any(), "known lower saddle was not recovered by either probe"
        position = positions[index]
        topology.update(
            lower_saddle_qualified=True,
            lower_saddle_probe_converged=np.asarray(found["converged"]).tolist(),
            lower_saddle_probe_qualified=valid.tolist(),
            lower_saddle_probe_gradient_norm=np.asarray(
                found["gradient_norm"]
            ).tolist(),
            lower_saddle_probe_positions_rz_m=positions.tolist(),
            lower_saddle_rz_m=position.tolist(),
            lower_saddle_flux_wb=float(found["value"][index]),
            lower_saddle_inside_wall=bool(
                inside_wall_units(position[None], data["limiter"])[0]
            ),
            lower_saddle_distance_m=float(distance[index]),
        )
        observations.append(
            dict(distance_m=float(distance[index]), probes=valid.tolist())
        )

    for trip in record["trip_topology"]:
        with np.load(args.row / f"inner-{trip['trip']}.npz") as archive:
            refresh(trip["initial"], archive["initial"] * scale)
            refresh(trip["returned"], archive["state"] * scale)
    refresh(record["terminal_topology"], data["state"])
    assert observations[0]["distance_m"] < 1e-12
    record["recovery"]["lower_root_diagnostic_refresh"] = observations
    data["continued_saddle"] = np.r_[
        record["terminal_topology"]["lower_saddle_rz_m"],
        record["terminal_topology"]["lower_saddle_flux_wb"],
    ]
    assert hashlib.sha256(data["state"].tobytes()).hexdigest() == state_hash
    np.savez_compressed(args.row / "terminal.npz", **data)
    (args.row / "receipt.json").write_text(json.dumps(record, indent=2) + "\n")
    (args.row / "trips.json").write_text(
        json.dumps(record["trip_topology"], indent=2) + "\n"
    )
    print("REFRESHED_LOWER_ROOTS " + json.dumps(observations), flush=True)
    print("FORWARD_SOLVES_EXECUTED=0; STATE_SHA256=" + state_hash, flush=True)


if __name__ == "__main__":
    main()
