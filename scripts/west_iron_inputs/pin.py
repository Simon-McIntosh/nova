"""Write the WEST 62751 solve-inputs pin from its hexadecimal source.

west_inputs() rebuilds the WEST region, source field and CEA material table
and asserts them against this frozen pin, so a pin whose writing code is
unknown proves nothing. The pin is therefore written from a clean tree and
stamps both the revision and the blob of the script that produced it.

    python scripts/west_iron_inputs/pin.py --output <pin path>
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import pickle
import sys

import numpy as np
import shapely

from nova.imas.machine import StaticMachineDescription
from nova.jax.config import configure_dtypes

configure_dtypes()

from scripts.west_iron_galerkin_pair.measure import (  # noqa: E402
    DISTANCES,
    SOURCE,
    SOURCE_SHA256,
    code_provenance,
    machine_source,
    save,
    source_field,
)

ROOT = Path(__file__).resolve().parents[2]
LAW = (
    "Anhysteretic, single-valued: the model carries no remanent magnetisation "
    "and no minor-loop permeability."
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    revision, blob = code_provenance(Path(__file__))
    print(
        f"# revision: {revision} pin_py_blob: {blob} tree: {ROOT} "
        f"cwd: {Path.cwd().resolve()}",
        flush=True,
    )
    digest = hashlib.sha256(SOURCE.read_bytes()).hexdigest()
    assert digest == SOURCE_SHA256, digest
    with SOURCE.open("rb") as stream:
        data = pickle.load(stream)
    uri, dd_version = machine_source()
    machine = StaticMachineDescription.read_west_iron_outline(
        uri, dd_version=dd_version
    ).description
    assert len(machine.iron_segments) == 6
    table = machine.iron_segments[0].material
    table_b = np.asarray(table.b_field)
    table_mu = np.asarray(table.permeability_relative)
    for segment in machine.iron_segments:
        np.testing.assert_array_equal(segment.material.b_field, table_b)
        np.testing.assert_array_equal(segment.material.permeability_relative, table_mu)
    region = shapely.union_all(
        [
            shapely.Polygon(np.column_stack((s.outline.r, s.outline.z)))
            for s in machine.iron_segments
        ]
    )
    assert region.is_valid
    external = source_field(data)
    probes = np.column_stack((data["edge"] + DISTANCES, np.zeros(3)))
    air = external(probes)
    line_points = np.column_stack((data["radius"], np.zeros_like(data["radius"])))
    known = data["radius"] > data["edge"] + 0.3
    line_air = np.linalg.norm(external(line_points), axis=1)
    control = float(np.max(np.abs(line_air[known] / data["bmag"][0, known] - 1)))
    assert control < 0.002, control
    record = dict(
        revision=revision,
        pin_py_blob=blob,
        pin_py=str(Path(__file__).resolve().relative_to(ROOT)),
        tree=str(ROOT),
        cwd=str(Path.cwd().resolve()),
        source=str(SOURCE),
        source_sha256=digest,
        machine=uri.removeprefix("imas:hdf5?path="),
        dd_version=dd_version,
        segments=len(machine.iron_segments),
        time_s=data["time"],
        source_control_max_relative=control,
        probe_distance_m=DISTANCES.tolist(),
        probes_m=probes.tolist(),
        air_field_T=air.tolist(),
        air_magnitude_T=np.linalg.norm(air, axis=1).tolist(),
        table_b_T=table_b.tolist(),
        table_mu_relative=table_mu.tolist(),
        law=LAW,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    save(args.output, record)
    print(
        "PIN_WRITTEN "
        + json.dumps(
            dict(
                path=str(args.output),
                probes_m=record["probes_m"],
                air_magnitude_T=record["air_magnitude_T"],
                table_knots=int(len(table_b)),
                source_control_max_relative=control,
            )
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
