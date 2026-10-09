"""Exercise the polygon fraction guard on intact and corrupted read receipts."""

# Precision precedes array-valued imports.
# ruff: noqa: E402
from pathlib import Path
import subprocess
import sys

from nova.jax.config import configure_dtypes

configure_dtypes()

import jax
import numpy as np

from nova.equilibrium import topology
from nova.equilibrium.solve_request import TopologyPolicy
from scripts.prototypes.support import geometry
from tests.equilibrium.test_topology_read import (
    _analytic_inputs,
    _realised_hex_geometry,
)


def main():
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    print(
        f"REVISION={revision} TREE={Path.cwd()} "
        f"COMMAND={sys.executable} {Path(__file__).resolve()}",
        flush=True,
    )
    print(f"MODULE={geometry.__file__} CWD={Path.cwd()}", flush=True)
    assert jax.config.jax_enable_x64
    _, field, wall, _, _ = _analytic_inputs("limited")
    carrier = _realised_hex_geometry(wall, 132)
    convention = topology.TopologyConvention.from_cocos(17, 1.0)
    reading = jax.jit(topology.read)(field, carrier, convention, TopologyPolicy())
    assert bool(reading.valid)
    geometry.read_polygons(carrier, reading, convention.sigma)
    print("GUARD_POSITIVE_CONTROL=passed", flush=True)
    index = int(np.flatnonzero(np.asarray(reading.membership) > 0.99)[0])
    corrupted = reading._replace(membership=reading.membership.at[index].set(0.5))
    try:
        geometry.read_polygons(carrier, corrupted, convention.sigma)
    except AssertionError as error:
        assert "read polygons lost fragment membership" in str(error)
        print("GUARD_REFUSAL=" + str(error), flush=True)
    else:
        raise AssertionError("corrupted membership escaped the fraction guard")
    print("GUARD_COMPLETE=passed", flush=True)


if __name__ == "__main__":
    main()
