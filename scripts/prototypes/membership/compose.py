"""Measure the complete Bell read plus fragment patch in one device program."""

# ruff: noqa: E402
from nova.jax.config import configure_dtypes

configure_dtypes()

import argparse
import json
from pathlib import Path
import pickle
import subprocess
import sys
from time import perf_counter

import jax
import jax.numpy as jnp
import numpy as np
import nova

from nova.equilibrium.topology import TopologyConvention
from nova.equilibrium.solve_request import TopologyPolicy
from scripts.prototypes.membership.hybrid import patch_membership
from scripts.prototypes.read_route.bell import field_from_jets, read_bell

assert jax.config.jax_enable_x64 is True


def composed(payload, radius, crossed):
    jets, indices, mesh, geometry, fixed, _, sigma, diverted = payload
    field = field_from_jets(jets, indices, mesh)
    bell = read_bell(
        field,
        geometry,
        TopologyConvention.from_cocos(17, 1.0),
        TopologyPolicy(normal_form_radius=0.054806712567833996),
    )
    result = patch_membership(
        jets, indices, mesh, geometry, fixed, bell, sigma, diverted, radius, crossed
    )
    return result[0], result[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("rows", type=Path)
    args = parser.parse_args()
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    print(
        f"revision={revision} tree={Path.cwd()} command={sys.executable} "
        "-m scripts.prototypes.membership.compose " + str(args.rows),
        flush=True,
    )
    print(
        f"NOVA_MODULE={nova.__file__} MODULE={__file__} CWD={Path.cwd()} "
        f"DEVICES={jax.devices()}",
        flush=True,
    )
    assert jax.default_backend() == "gpu"
    output = args.rows / "composed.json"
    receipt = dict(rows=[], complete=False)
    for cells in (2000, 5000):
        for kind in ("diverted", "limited"):
            with (args.rows / f"{kind}-{cells}-booked-replay.pkl").open("rb") as stream:
                payload = pickle.load(stream)
            operands = payload, jnp.asarray(0.054806712567833996), jnp.asarray(False)
            started = perf_counter()
            program = jax.jit(composed).lower(*operands).compile()
            cold = perf_counter() - started
            for name, flag in (("preserved", False), ("radius", True)):
                operands = payload, jnp.asarray(0.054806712567833996), jnp.asarray(flag)
                jax.block_until_ready(program(*operands))
                walls = []
                for _ in range(3):
                    started = perf_counter()
                    result = program(*operands)
                    jax.block_until_ready(result)
                    walls.append(perf_counter() - started)
                reference = np.load(
                    args.rows / f"{kind}-{cells}-booked-{name}.npz"
                )["membership"]
                difference = float(np.max(np.abs(np.asarray(result[0]) - reference)))
                np.testing.assert_allclose(result[0], reference, rtol=0, atol=1e-12)
                row = dict(
                    kind=kind,
                    cells=cells,
                    arm=name,
                    valid=bool(result[1]),
                    median_seconds=float(np.median(walls)),
                    compile_seconds=cold,
                    seconds=walls,
                    membership_max_delta=difference,
                    cost_boundary=(
                        "full Bell read and patch, including redundant null census"
                    ),
                )
                receipt["rows"].append(row)
                output.write_text(json.dumps(receipt, indent=2) + "\n")
                print("COMPOSED_ROW", json.dumps(row), flush=True)
    receipt["complete"] = True
    output.write_text(json.dumps(receipt, indent=2) + "\n")


if __name__ == "__main__":
    main()
