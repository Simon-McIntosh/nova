"""Measure warm compiled-slice entry overhead against its direct dispatch."""

from __future__ import annotations

import argparse
from pathlib import Path

import jax
import jax.numpy as jnp

from benchmarks import compiled_slice_cache_receipt as receipt
from nova.equilibrium.topology import TopologyClass


ROOT = Path(__file__).resolve().parents[4]
DEFAULT_OUTPUT = Path(__file__).with_name("receipt.json")


def _direct_dispatch(result, member):
    """Dispatch the carried executable with already prepared public inputs."""
    state = jnp.asarray(member.state)
    requested = jnp.asarray(int(TopologyClass.DIVERTED), dtype=jnp.int8)
    shadow = jnp.ravel(
        jnp.asarray(member.operator.residual_shadow_mask(state, requested), dtype=bool)
    )
    output = result.program.slice_solver(state, shadow, result.program.external)
    jax.block_until_ready(output)
    return output


def run(output: Path, cache_root: Path) -> dict:
    """Run both paired arms in one allocation and publish the strict verdict."""
    receipt._direct_dispatch = _direct_dispatch
    result = receipt.run(output, cache_root=cache_root)
    for row in result["members"]:
        row["terminal_flux_zero_ulp"] = (
            row["compiled_host_terminal_flux_ulp"] == 0
            and row["cached_direct_terminal_flux_ulp"] == 0
        )
    result["acceptance"]["terminal_flux_max_ulp"] = 0
    result["verdict"]["terminal_flux_zero_ulp"] = all(
        row["terminal_flux_zero_ulp"] for row in result["members"]
    )
    result["source"]["entry_driver"] = str(Path(__file__).relative_to(ROOT))
    result["source"]["entry_driver_sha256"] = receipt._sha256(Path(__file__))
    result["source"]["direct_dispatch"] = (
        "slice solver with state, shadow, and carried exterior already prepared"
    )
    receipt._write_json(output, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--cache-root", type=Path, required=True)
    arguments = parser.parse_args()
    result = run(arguments.output.resolve(), arguments.cache_root.resolve())
    print(receipt.json.dumps(receipt._strict(result["verdict"]), sort_keys=True))


if __name__ == "__main__":
    main()
