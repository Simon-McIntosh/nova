"""Read saved states through their actual source operator, without a solve."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from scripts.prototypes.diiid_xpoint_attribution.attribute import write


def probe(measurement):
    import jax
    import jax.numpy as jnp
    from nova.jax.config import configure_dtypes

    configure_dtypes()
    assert jax.config.jax_enable_x64
    from benchmarks import diiid_forward_gs_match as bench
    from nova.equilibrium.forward import ForwardProfile

    def refuse(*a, **kw):
        raise AssertionError("operator probe cannot advance a forward solve")

    ForwardProfile.solve = refuse
    for frame in (137, 146):
        result = json.loads((measurement / f"result-{frame}.json").read_text())
        data = dict(np.load(Path(result["folder"]) / "terminal.npz"))
        row = bench._read(
            Path(result["archive"]),
            bench._LABEL_COLUMNS
            + bench._GEOMETRY_COLUMNS
            + bench._CURRENT_COLUMNS
            + bench._PLASMA_CURRENT_COLUMNS,
        )
        row["_source_path"] = result["archive"]
        built = bench._build_profile(row, frame)
        machine = bench.dataset_machine_description(row, source_row=result["archive"])
        shipped = bench.shipped_current_at(
            row, machine.physical, bench.POLOIDAL_CONDUCTORS, result["time_ms"]
        )
        adapter = bench.complete_profile_current_adapter(
            built.profile,
            shipped_names=bench.POLOIDAL_CONDUCTORS,
            shipped_current_a=shipped,
            use_circuit=True,
        )
        op = adapter.profile.operator
        target = result["current"]["prescribed_a"]
        print("PREPARED_OPERATOR", frame, op.use_linear_moments, flush=True)

        @jax.jit
        def read(state):
            masks, topology, connected, admitted = op._fixed_design_read(state)
            moments = op._point_current_moments(masks)
            amplitude = op.current_normalisation_amplitude(
                target, jnp.sum(moments.cell_current)
            )
            imaged = op.current_moment_image(
                op.scaled_current_moments(moments, amplitude)
            )
            return (
                masks,
                topology,
                connected,
                admitted,
                moments.cell_current,
                amplitude,
                imaged,
            )

        states = [("reference", np.asarray(built.seed)), ("solved", data["state"])]
        records = []
        for label, state in states:
            masks, topology, connected, admitted, unscaled, amplitude, imaged = (
                jax.device_get(read(jnp.asarray(state)))
            )
            external = np.asarray(op.external(op.external_current))
            if label == "solved":
                difference = imaged + external - state
            else:
                difference = imaged + external - state
            np.savez_compressed(
                measurement / f"operator-{frame}-{label}.npz",
                state=state,
                source_image=imaged,
                external=external,
                source_current=unscaled * amplitude,
                psi_norm=np.asarray(masks.psi_norm),
                labels=np.asarray(masks.label),
                participation=np.asarray(masks.profile_participation),
                connected=connected,
                average_stencil=np.asarray(op.cell_average_stencil),
                average_weight=np.asarray(op.cell_average_weight),
                radius=np.asarray(op.radius),
            )
            record = dict(
                label=label,
                axis_m=np.asarray(topology.axis),
                saddle_m=np.asarray(topology.x_point),
                axis_flux_wb=float(topology.axis_flux),
                boundary_flux_wb=float(topology.boundary_flux),
                amplitude=float(amplitude),
                admitted=bool(admitted),
                support=int(connected.sum()),
                max_map_difference_wb=float(np.max(np.abs(difference))),
                grid_max_map_difference_wb=float(
                    np.max(np.abs(difference[: data["flux"].size]))
                ),
                total_current_a=float(np.sum(unscaled * amplitude)),
            )
            records.append(record)
            print(
                "OPERATOR_READ",
                frame,
                json.dumps(
                    {k: v for k, v in record.items() if k not in ["axis_m", "saddle_m"]}
                ),
                flush=True,
            )
        write(
            measurement / f"operator-{frame}.json",
            dict(rows=records, forward_solves=0),
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--measurement", type=Path, required=True)
    args = parser.parse_args()
    probe(args.measurement)


if __name__ == "__main__":
    main()
