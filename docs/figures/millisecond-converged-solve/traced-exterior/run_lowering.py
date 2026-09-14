"""Lower exterior-variable forward programs without compiling or executing."""

from __future__ import annotations

import difflib
import hashlib
import json
import os
from pathlib import Path
import re

import jax.numpy as jnp
import numpy as np

from nova.equilibrium import reduced_newton
from nova.equilibrium.forward_operator import (
    PrescribedCurrentField,
    set_support_clip_mode,
    support_clip_mode,
)
from nova.jax.config import configure_dtypes
from tests.test_forward_compile_identity import (
    CASES,
    _certificate_row,
    _digest,
    _lower_certificate_solve,
)


OUTPUT = Path(__file__).with_name("same-mesh-lowering.json")
RUN_DIRECTORY = Path(os.environ.get("HLO_RUN_DIRECTORY", OUTPUT.parent))
RUN_LABEL = os.environ.get("SLURM_JOB_ID", "local")
_CONSTANT_PATTERN = re.compile(
    r"(?P<symbol>%[A-Za-z0-9_]+)\s*=\s*stablehlo\.constant\s+"
    r"dense<(?P<payload>.*?)>\s*:\s*tensor<(?P<type>[^>]+)>",
    re.DOTALL,
)
_NUMPY_DTYPES = {
    "bf16": None,
    "f16": np.dtype("<f2"),
    "f32": np.dtype("<f4"),
    "f64": np.dtype("<f8"),
    "i8": np.dtype("i1"),
    "i16": np.dtype("<i2"),
    "i32": np.dtype("<i4"),
    "i64": np.dtype("<i8"),
    "ui8": np.dtype("u1"),
    "ui16": np.dtype("<u2"),
    "ui32": np.dtype("<u4"),
    "ui64": np.dtype("<u8"),
}


def _constant_shape_and_dtype(type_text: str) -> tuple[list[int], str]:
    """Split a StableHLO tensor type into its static shape and element type."""
    parts = type_text.split("x")
    dtype = parts[-1]
    shape = [int(part) for part in parts[:-1]]
    return shape, dtype


def _constant_first_values(payload: str, dtype: str) -> list[object]:
    """Decode a short value prefix from a dense StableHLO literal."""
    compact = payload.strip()
    numpy_dtype = _NUMPY_DTYPES.get(dtype)
    if compact.startswith('"0x') and compact.endswith('"') and numpy_dtype:
        raw = bytes.fromhex(compact[3:-1])
        count = min(6, len(raw) // numpy_dtype.itemsize)
        values = np.frombuffer(raw, dtype=numpy_dtype, count=count)
        return values.tolist()
    if compact.startswith("[") and compact.endswith("]"):
        compact = compact[1:-1]
    values = []
    for token in compact.split(",")[:6]:
        token = token.strip()
        if not token:
            continue
        try:
            values.append(float(token))
        except ValueError:
            values.append(token[:80])
    return values


def _constants(text: str) -> list[dict[str, object]]:
    """Return the ordered dense constants from one StableHLO module."""
    result = []
    for ordinal, match in enumerate(_CONSTANT_PATTERN.finditer(text)):
        shape, dtype = _constant_shape_and_dtype(match.group("type"))
        payload = match.group("payload")
        result.append(
            {
                "ordinal": ordinal,
                "symbol": match.group("symbol"),
                "shape": shape,
                "dtype": dtype,
                "first_values": _constant_first_values(payload, dtype),
                "payload": payload,
            }
        )
    return result


def _write_hlo_difference(fixture, scaled) -> tuple[str, str, Path]:
    """Persist both modules, their textual diff and changed constant census."""
    RUN_DIRECTORY.mkdir(parents=True, exist_ok=True)
    fixture_text = fixture.as_text(dialect="stablehlo")
    scaled_text = scaled.as_text(dialect="stablehlo")
    fixture_path = RUN_DIRECTORY / f"certificate-fixture-{RUN_LABEL}.stablehlo"
    scaled_path = RUN_DIRECTORY / f"certificate-scaled-{RUN_LABEL}.stablehlo"
    diff_path = RUN_DIRECTORY / f"certificate-{RUN_LABEL}.stablehlo.diff"
    constants_path = RUN_DIRECTORY / f"certificate-{RUN_LABEL}-constants.json"
    fixture_path.write_text(fixture_text)
    scaled_path.write_text(scaled_text)
    diff_path.write_text(
        "\n".join(
            difflib.unified_diff(
                fixture_text.splitlines(),
                scaled_text.splitlines(),
                fromfile=str(fixture_path),
                tofile=str(scaled_path),
                lineterm="",
            )
        )
        + "\n"
    )

    fixture_constants = _constants(fixture_text)
    scaled_constants = _constants(scaled_text)
    differing = []
    for ordinal in range(max(len(fixture_constants), len(scaled_constants))):
        first = fixture_constants[ordinal] if ordinal < len(fixture_constants) else None
        second = scaled_constants[ordinal] if ordinal < len(scaled_constants) else None
        if first is not None and second is not None:
            unchanged = (
                first["shape"] == second["shape"]
                and first["dtype"] == second["dtype"]
                and first["payload"] == second["payload"]
            )
            if unchanged:
                continue
        differing.append(
            {
                "ordinal": ordinal,
                "fixture": (
                    None
                    if first is None
                    else {k: v for k, v in first.items() if k != "payload"}
                ),
                "scaled": (
                    None
                    if second is None
                    else {k: v for k, v in second.items() if k != "payload"}
                ),
            }
        )
    census = {
        "fixture_constant_count": len(fixture_constants),
        "scaled_constant_count": len(scaled_constants),
        "differing_constant_count": len(differing),
        "differing_constants": differing,
    }
    constants_path.write_text(json.dumps(census, indent=2, sort_keys=True) + "\n")
    print(
        "STAGE StableHLO artifacts written "
        f"fixture={fixture_path} scaled={scaled_path} diff={diff_path} "
        f"constants={constants_path} differing={len(differing)}",
        flush=True,
    )
    return fixture_text, scaled_text, constants_path


def _compiled_slice_digests(row) -> tuple[str, str]:
    """Return StableHLO identities for two prescribed-current vectors."""
    profile, seed, requested_class, target_current, _request = row
    operator = profile.operator
    response = jnp.asarray(operator.external())[:, None]
    operator.prescribed_field = PrescribedCurrentField(
        response=response,
        current=jnp.asarray([1.0], dtype=response.dtype),
    )
    first_current = jnp.asarray([1.0], dtype=response.dtype)
    second_current = jnp.asarray([0.8], dtype=response.dtype)
    first_external = operator.external(prescribed_current=first_current)
    second_external = operator.external(prescribed_current=second_current)
    program, raw_kernels = reduced_newton._compiled_program(
        operator,
        seed,
        requested_class=requested_class,
        target_current=jnp.asarray(target_current),
        external=first_external,
        program=None,
    )
    kernels = reduced_newton._bind_dynamic_arguments(
        raw_kernels,
        first_external,
        jnp.asarray(target_current),
        requested_class,
        bind_external=False,
    )
    solver = reduced_newton._compiled_slice_solver(
        kernels,
        tolerance=1.0e-8,
        newton_steps=1,
        active_set_steps=1,
    )
    shadow = jnp.ravel(
        jnp.asarray(operator.residual_shadow_mask(seed, requested_class), dtype=bool)
    )
    print("STAGE compiled-slice lower first prescribed current", flush=True)
    first = solver.lower(seed, shadow, first_external)
    print("STAGE compiled-slice lower second prescribed current", flush=True)
    second = solver.lower(seed, shadow, second_external)
    assert program.external_shape == first_external.shape
    return _digest(first), _digest(second)


def main() -> int:
    print("STAGE configure CPU extended precision", flush=True)
    configure_dtypes()
    set_support_clip_mode("chord")
    assert support_clip_mode() == "chord"

    print("STAGE load weak cached 300-cell certificate row", flush=True)
    weak = _certificate_row(CASES[0])
    fixture_external = weak[0].operator.external()
    scaled_external = fixture_external * jnp.asarray(0.9, dtype=fixture_external.dtype)

    print("STAGE certificate lower fixture exterior", flush=True)
    fixture_lowered, _ = _lower_certificate_solve(weak, fixture_external)
    print("STAGE certificate lower scaled exterior", flush=True)
    scaled_lowered, _ = _lower_certificate_solve(weak, scaled_external)
    fixture_hlo, scaled_hlo, constants_path = _write_hlo_difference(
        fixture_lowered, scaled_lowered
    )
    certificate_digests = (
        hashlib.sha256(fixture_hlo.encode()).hexdigest(),
        hashlib.sha256(scaled_hlo.encode()).hexdigest(),
    )

    print("STAGE compiled-slice lowering", flush=True)
    compiled_digests = _compiled_slice_digests(weak)
    receipt = {
        "clip_mode": support_clip_mode(),
        "requested_cells": -300,
        "external_shape": list(fixture_external.shape),
        "exterior_scale": 0.9,
        "differing_constants": str(constants_path),
        "certificate": {
            "fixture_sha256": certificate_digests[0],
            "scaled_sha256": certificate_digests[1],
            "identical": certificate_digests[0] == certificate_digests[1],
        },
        "compiled_slice": {
            "first_prescribed_current_sha256": compiled_digests[0],
            "second_prescribed_current_sha256": compiled_digests[1],
            "identical": compiled_digests[0] == compiled_digests[1],
        },
    }
    OUTPUT.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(f"STAGE receipt written {OUTPUT}", flush=True)
    print(json.dumps(receipt, sort_keys=True), flush=True)
    return (
        0
        if all(
            section["identical"]
            for section in (receipt["certificate"], receipt["compiled_slice"])
        )
        else 1
    )


if __name__ == "__main__":
    raise SystemExit(main())
