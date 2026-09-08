"""Direct evidence for sampled-profile evaluation inside a real MAST solve."""

from __future__ import annotations

from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from benchmarks import strict_exit_incidence as incidence
from nova.equilibrium.solve_request import SampledFluxFunction
from nova.jax.config import (
    configure_dtypes,
    configure_persistent_compilation_cache,
    default_persistent_compilation_cache_root,
)


def _array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
    digest.update(array.tobytes())
    return digest.hexdigest()


def _recording_function(function, batches: list[np.ndarray]):
    def record(value) -> None:
        batches.append(np.asarray(value).copy())

    def recorded(psi_norm):
        jax.debug.callback(record, psi_norm)
        return function(psi_norm)

    return recorded


def _compare_at_recorded_points(
    closure,
    sampled: SampledFluxFunction,
    batches: list[np.ndarray],
) -> dict[str, object]:
    points = np.concatenate([batch.reshape(-1) for batch in batches])
    closure_values = np.asarray(closure(jnp.asarray(points)))
    sampled_values = np.asarray(sampled(jnp.asarray(points)))
    different = closure_values != sampled_values
    absolute_difference = np.abs(closure_values - sampled_values)
    return {
        "callback_batches": len(batches),
        "point_occurrences": int(points.size),
        "distinct_point_values": int(np.unique(points).size),
        "different_point_occurrences": int(np.count_nonzero(different)),
        "largest_absolute_difference": float(np.max(absolute_difference)),
        "point_sha256": _array_sha256(points),
        "closure_value_sha256": _array_sha256(closure_values),
        "sampled_value_sha256": _array_sha256(sampled_values),
    }


@pytest.mark.slow
def test_sampled_profiles_at_exact_width_one_mast_solve_points(monkeypatch):
    """Compare both representations at every profile point a real solve uses."""
    state_cache = os.environ.get("NOVA_MAST_STATE_CACHE")
    output = os.environ.get("NOVA_PROFILE_POINT_AUDIT_OUTPUT")
    if not state_cache or not output:
        pytest.skip("the exact MAST profile-point audit needs its cache and output")

    configure_dtypes()
    cache_receipt = configure_persistent_compilation_cache(
        default_persistent_compilation_cache_root()
    )
    transform = incidence._mast_source_with_sampled_profiles
    monkeypatch.setattr(
        incidence,
        "_mast_source_with_sampled_profiles",
        lambda source, group, row: source,
    )
    closure_members, _ = incidence._build_mast_members(
        Path(state_cache), member_count=1
    )
    monkeypatch.setattr(incidence, "_mast_source_with_sampled_profiles", transform)
    sampled_members, _ = incidence._build_mast_members(
        Path(state_cache), member_count=1
    )
    closure_member = closure_members[0]
    sampled_member = sampled_members[0]
    assert incidence._array_sha256(closure_member.state) == incidence._array_sha256(
        sampled_member.state
    )

    closure_core = closure_member.profile.operator.source.core
    sampled_core = sampled_member.profile.operator.source.core
    assert isinstance(sampled_core.p_prime, SampledFluxFunction)
    assert isinstance(sampled_core.ff_prime, SampledFluxFunction)
    recorded_points: dict[str, list[np.ndarray]] = {"p_prime": [], "ff_prime": []}
    source = replace(
        closure_member.profile.operator.source,
        core=replace(
            closure_core,
            p_prime=_recording_function(
                closure_core.p_prime, recorded_points["p_prime"]
            ),
            ff_prime=_recording_function(
                closure_core.ff_prime, recorded_points["ff_prime"]
            ),
        ),
    )
    operator = replace(
        closure_member.profile.operator,
        source=source,
        prescribed_current_field=closure_member.profile.operator.prescribed_field,
    )
    recorded_member = replace(
        closure_member,
        profile=replace(closure_member.profile, operator=operator),
    )
    compiled, state, compile_seconds = incidence._compiled_member(recorded_member)
    result = compiled(state, jnp.asarray(False))
    incidence._block(result)
    jax.effects_barrier()

    comparisons = {
        name: _compare_at_recorded_points(
            getattr(closure_core, name), getattr(sampled_core, name), batches
        )
        for name, batches in recorded_points.items()
    }
    payload = {
        "schema": "nova.sampled-flux-solve-point-audit/1",
        "machine": "MAST",
        "member": closure_member.identity,
        "solve_route": "width_one_newton_krylov_without_strict_exit",
        "compile_seconds": compile_seconds,
        "terminal_state_sha256": incidence._array_sha256(result.flux),
        "comparison": comparisons,
        "all_points_bit_identical": all(
            record["different_point_occurrences"] == 0
            for record in comparisons.values()
        ),
        "persistent_compilation_cache": cache_receipt.receipt(),
    }
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(incidence._strict(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print("PROFILE_POINT_AUDIT=" + json.dumps(payload, sort_keys=True))
    assert payload["all_points_bit_identical"]
