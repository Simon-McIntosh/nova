"""Apply the head current owner to a recorded historical read and profile."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from nova.jax.config import configure_dtypes


def main():
    configure_dtypes()
    assert jax.config.jax_enable_x64 is True
    from benchmarks.diiid_forward_gs_match import _profile_function
    from nova.equilibrium import source as source_module
    from nova.equilibrium.domain import DomainMasks
    from nova.equilibrium.forward import _cubic_cell_average_stencil
    from nova.equilibrium.forward_operator import point_route_current
    from nova.equilibrium.source import DomainProfile, ForwardSource

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    reference = json.loads(arguments.reference.read_text())
    data = reference["partition"]
    source = ForwardSource(
        core=DomainProfile(
            p_prime=_profile_function(data["profile_coordinate"], data["p_prime"]),
            ff_prime=_profile_function(data["profile_coordinate"], data["ff_prime"]),
        )
    )
    masks = DomainMasks(
        jnp.asarray(data["domain_label"]), jnp.asarray(data["psi_norm"])
    )
    stencil, weight = _cubic_cell_average_stencil(
        tuple(reference["label_identity"]["shape"])
    )
    current = np.asarray(
        point_route_current(
            source,
            jnp.asarray(data["radius"]),
            jnp.asarray(data["area"]),
            masks,
            jnp.asarray(stencil),
            jnp.asarray(weight),
        )
    )
    result = {
        "reference_revision": reference["revision"],
        "reference_path": str(arguments.reference),
        "source_module": source_module.__file__,
        "cwd": str(Path.cwd()),
        "restored_current": current.tolist(),
        "restored_current_a": float(np.sum(current)),
        "registered_seed_current_a": reference["registered_seed_current_a"],
        "changes_replayed": [
            "historical domain labels and normalised flux",
            "constant profile endpoint extension",
            "point-current integration",
        ],
    }
    arguments.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(f"MODULE={source_module.__file__} CWD={Path.cwd()}")
    print(f"RESTORED_CURRENT_A={result['restored_current_a']:.12f}")


if __name__ == "__main__":
    main()
