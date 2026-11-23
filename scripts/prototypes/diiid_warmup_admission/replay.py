"""Read topology on captured operands without repeating a forward solve."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from measure import load
from nova.jax.config import configure_dtypes

ROOT = Path(__file__).resolve().parents[3]


def replay(index, frame, output):
    material = load(
        "material_replay", ROOT / "scripts/prototypes/diiid_material_mask/measure.py"
    )
    print(f"PREPARE frame={frame}", flush=True)
    prepared = material.probe._prepare_frame(frame, None)
    operator = prepared["profile"].operator
    assert int(np.asarray(operator.inside_material).sum()) == 527
    print("PREPARED material=527", flush=True)

    @jax.jit
    def read(physical):
        _masks, topology, connected, admitted = operator._fixed_design_read(physical)
        return topology, connected, admitted

    cache = {}

    def inspect(physical):
        key = hashlib.sha256(physical.tobytes()).hexdigest()
        if key in cache:
            return cache[key]
        if not np.isfinite(physical).all():
            return dict(finite_state=False)
        topology, connected, admitted = jax.device_get(read(jnp.asarray(physical)))
        result = dict(
            finite_state=True,
            support=int(connected.sum()),
            diverted=bool(topology.diverted),
            axis_admitted=bool(admitted),
            axis=np.asarray(topology.axis).tolist(),
            saddle=np.asarray(topology.x_point).tolist(),
            saddle_flux=float(topology.x_point_flux),
            boundary=np.asarray(topology.boundary).tolist(),
            boundary_flux=float(topology.boundary_flux),
            wall_point=np.asarray(topology.wall_point).tolist(),
        )
        cache[key] = result
        return result

    results = {}
    for record in index:
        if record["case"] != f"frame-{frame}-diverted":
            continue
        path = Path(record["source_path"])
        terminal = np.load(path / "terminal.npz")
        count = terminal["state"].size
        last = np.load(path / f"inner-{len(record['inner']) - 1}.npz")["state"][:count]
        scale = float(np.dot(terminal["state"], last) / np.dot(last, last))
        error = float(np.max(np.abs(terminal["state"] - scale * last)))
        assert error < 1e-12
        control = inspect(terminal["state"])
        assert control["support"] == record["support"]
        assert np.allclose(control["axis"], terminal["axis"][:2], atol=1e-12)
        assert np.allclose(control["boundary"], terminal["boundary"][:2], atol=1e-12)
        print(f"CONTROL {record['arm']} {control}", flush=True)
        stages = json.loads((path / "stages.json").read_text())
        trip = 0
        trip_by_stage = {}
        for line in (path / "row.log").open():
            if line.startswith("INNER "):
                trip += 1
            elif line.startswith("STAGE "):
                event = json.loads(line[6:])
                trip_by_stage[event["stage"], event["index"]] = trip
        rows = []
        for event in stages:
            stage, number = event["stage"], event["index"]
            values = np.load(path / f"{stage}-{number}.npz")["state"][:count]
            proposal = stage == "warmup" and record["arm"] == "head" and number % 2 == 1
            row = dict(
                trip=trip_by_stage[stage, number],
                stage=stage,
                index=number,
                role="proposal" if proposal else "incumbent",
                image_finite=event["finite_image"],
                proposal_admitted=(
                    bool(event["finite_image"] and np.isfinite(event["residual"]))
                    if proposal
                    else None
                ),
                **inspect(np.asarray(values * scale)),
            )
            rows.append(row)
            print("REPLAY " + json.dumps(dict(arm=record["arm"], **row)), flush=True)
        trips = []
        for number in range(len(record["inner"])):
            arrays = np.load(path / f"inner-{number}.npz")
            trips.append(
                dict(
                    trip=number,
                    initial=inspect(np.asarray(arrays["initial"][:count] * scale)),
                    returned=inspect(np.asarray(arrays["state"][:count] * scale)),
                    **{
                        key: record["inner"][number][key]
                        for key in ("attempted", "accepted", "reason")
                    },
                )
            )
        results[record["arm"]] = dict(
            source_path=str(path),
            flux_scale=scale,
            scale_max_error=error,
            terminal_control=control,
            stages=rows,
            trips=trips,
            caveat=(
                "Post hoc class-free topology reads of executed operands; "
                "refused proposals are labelled and never treated as "
                "admitted solver states."
            ),
        )
        output.write_text(json.dumps(results, indent=2) + "\n")
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comparison", type=Path, required=True)
    parser.add_argument("--frame", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    configure_dtypes()
    assert jax.config.jax_enable_x64
    replay(json.loads(args.comparison.read_text()), args.frame, args.output)


if __name__ == "__main__":
    main()
