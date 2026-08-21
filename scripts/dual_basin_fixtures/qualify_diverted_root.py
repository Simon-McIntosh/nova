"""Qualify the banked diverted state against the composed forward map."""

from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from nova.equilibrium.forward import ForwardProfile
from nova.equilibrium.source import DomainProfile, ForwardSource
from nova.equilibrium.stencil_mesh import StencilMesh
from nova.equilibrium.topology import TopologyClass
from nova.jax.config import configure_dtypes
from scripts.analytic_oracle_fixtures.measure import (
    FIXTURE_REQUESTS,
    WALL_POINT_COUNT,
    analytic_case,
    cached_machine,
    forward_operator,
)


OUTPUT = Path(__file__).resolve().parent
REPOSITORY_ROOT = OUTPUT.parents[1]
STATE_PATH = OUTPUT / "diverted-state.npz"
ORACLE_RECEIPT_PATH = OUTPUT / "diverted-receipt.json"
ROOT_RECEIPT_PATH = OUTPUT / "diverted-root-receipt.json"
MACHINE_PRECISION_FLOOR = 1.0e-14


def _digest(values: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(values).tobytes()).hexdigest()


def _flat(value: float):
    def profile(psi_norm):
        return jnp.full_like(jnp.asarray(psi_norm), value)

    return profile


def _maximum(values: np.ndarray) -> float:
    return float(np.max(np.abs(values)))


def _strict_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def qualify(*, write: bool = True) -> dict[str, object]:
    """Measure the stored state through one pinned write-then-read cycle."""
    configure_dtypes()
    oracle = json.loads(ORACLE_RECEIPT_PATH.read_text(encoding="utf-8"))
    with np.load(STATE_PATH) as bank:
        state = np.asarray(bank["state"])
    expected_identity = oracle["arrays"]["state"]
    if _digest(state) != expected_identity["sha256"]:
        raise ValueError("the diverted state does not match its banked identity")

    case = analytic_case()
    machine = cached_machine(
        case,
        FIXTURE_REQUESTS["fine"],
        wall_nodes=WALL_POINT_COUNT,
    )
    gradients = oracle["closed_form"]["constant_flux_functions"]
    source = ForwardSource(
        core=DomainProfile(
            p_prime=_flat(gradients["p_prime_pa_per_wb"]),
            ff_prime=_flat(gradients["ff_prime_t2_m2_per_wb"]),
        ),
        boundary_pressure=0.0,
        boundary_field_function=5.0,
    )
    requested_class = TopologyClass.DIVERTED

    source_operator = replace(forward_operator(case, machine), source=source)
    source_image = np.asarray(source_operator.internal(state, requested_class))
    declared_external = state - source_image
    operator = replace(
        forward_operator(case, machine, declared_external),
        source=source,
    )

    external_image = np.asarray(operator.external())
    repeated_source_image = np.asarray(operator.internal(state, requested_class))
    mapped = np.asarray(operator(state, requested_class=requested_class))
    residual = mapped - state
    mapped_scale = max(_maximum(mapped), 1.0e-30)

    lattice = StencilMesh(machine.node, machine.stencil, machine.area)
    profile = ForwardProfile(operator, lattice)
    branch = profile.solve_branch(
        state,
        requested_class,
        route="picard",
        evaluations=1,
        relaxation=1.0,
        tolerance=MACHINE_PRECISION_FLOOR,
    )

    pinned_masks, pinned = operator.read(state, requested_class)
    achieved_masks, achieved = operator.read(state)
    requested = int(branch.requested_class)
    achieved_class = int(branch.achieved_class)
    topology_consistent = bool(branch.topology_consistent)
    pinned_boundary = np.asarray(pinned.boundary)
    achieved_boundary = np.asarray(achieved.boundary)
    pinned_axis = np.asarray(pinned.axis)
    achieved_axis = np.asarray(achieved.axis)

    receipt = {
        "schema": "nova.forward-root-receipt",
        "schema_version": 1,
        "state": {
            "path": str(STATE_PATH.relative_to(REPOSITORY_ROOT)),
            "sha256": _digest(state),
            "shape": list(state.shape),
            "dtype": state.dtype.str,
        },
        "map": {
            "operator": "ForwardFluxOperator",
            "requested_class": requested,
            "achieved_class": achieved_class,
            "topology_consistent": topology_consistent,
            "finite": bool(np.all(np.isfinite(mapped))),
            "converged": bool(branch.converged),
            "iterations": int(branch.iterations),
            "absolute_residual_wb": _maximum(residual),
            "relative_residual": float(branch.residual),
            "mapped_flux_scale_wb": mapped_scale,
            "terminal_state_difference_wb": _maximum(
                np.asarray(branch.equilibrium.flux) - state
            ),
            "machine_precision_floor": MACHINE_PRECISION_FLOOR,
            "at_machine_precision_floor": float(branch.residual)
            <= MACHINE_PRECISION_FLOOR,
        },
        "composition": {
            "external_field": {
                "sha256": _digest(external_image),
                "maximum_absolute_flux_wb": _maximum(external_image),
                "reconstruction_difference_wb": _maximum(
                    external_image - declared_external
                ),
            },
            "source_forcing": {
                "sha256": _digest(repeated_source_image),
                "maximum_absolute_flux_wb": _maximum(repeated_source_image),
                "repeat_difference_wb": _maximum(repeated_source_image - source_image),
                "p_prime_pa_per_wb": gradients["p_prime_pa_per_wb"],
                "ff_prime_t2_m2_per_wb": gradients["ff_prime_t2_m2_per_wb"],
            },
            "normalization_anchor": {
                "pinned_axis_m": pinned_axis.tolist(),
                "unpinned_axis_m": achieved_axis.tolist(),
                "axis_distance_m": float(np.linalg.norm(pinned_axis - achieved_axis)),
                "pinned_boundary_m": pinned_boundary.tolist(),
                "unpinned_boundary_m": achieved_boundary.tolist(),
                "boundary_distance_m": float(
                    np.linalg.norm(pinned_boundary - achieved_boundary)
                ),
                "pinned_axis_flux_wb": float(pinned.axis_flux),
                "unpinned_axis_flux_wb": float(achieved.axis_flux),
                "axis_flux_difference_wb": float(pinned.axis_flux - achieved.axis_flux),
                "pinned_boundary_flux_wb": float(pinned.boundary_flux),
                "unpinned_boundary_flux_wb": float(achieved.boundary_flux),
                "boundary_flux_difference_wb": float(
                    pinned.boundary_flux - achieved.boundary_flux
                ),
                "domain_label_difference_count": int(
                    np.count_nonzero(
                        np.asarray(pinned_masks.label)
                        != np.asarray(achieved_masks.label)
                    )
                ),
            },
            "closure_absolute_residual_wb": _maximum(
                external_image + repeated_source_image - state
            ),
        },
        "evidence": {
            "jax_backend": jax.default_backend(),
            "jax_x64_enabled": bool(jax.config.x64_enabled),
            "carrier_cache_warm_hit": bool(machine.cache["hit"]),
            "verdict": (
                "genuine_machine_precision_root"
                if bool(branch.converged) and topology_consistent
                else "composition_defect"
            ),
        },
    }
    if write:
        _strict_json(ROOT_RECEIPT_PATH, receipt)
    return receipt


def main() -> None:
    receipt = qualify()
    mapped = receipt["map"]
    print(
        f"BANKED residual={mapped['relative_residual']:.17g} "
        f"requested={mapped['requested_class']} "
        f"achieved={mapped['achieved_class']} "
        f"consistent={mapped['topology_consistent']} "
        f"receipt={ROOT_RECEIPT_PATH}"
    )


if __name__ == "__main__":
    main()
