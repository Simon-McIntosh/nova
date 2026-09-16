"""Whether flux-function profile coefficients remain solve-program operands.

Lower the certificate solve program (no execution) on the weak 300-cell
whole-cell row for the committed source profile and for a second ForwardSource
whose flux-function gradients are scaled (pressure-gradient p' by 0.9 and the
diamagnetic current-profile FF' by 1.1), and compare the resulting StableHLO
modules.  A digest difference with the coefficients among the differing
constants is the evidence that the flux functions are baked into the program
as closure constants rather than passed as traced arguments, so every profile
change recompiles.

The benchmark also inventories the explicit coefficient and normalisation
arguments and records why a fixed-order polynomial is the higher-order
per-slice representation beside the existing piecewise-linear sampled route.
"""

from __future__ import annotations

import dataclasses
import difflib
import hashlib
import json
import os
from pathlib import Path
import re
import time

import jax
import jax.numpy as jnp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from nova.equilibrium import reduced_newton
from nova.equilibrium.forward_operator import (
    PrescribedCurrentField,
    _CallableLayout,
    _SourceLayout,
    set_support_clip_mode,
    support_clip_mode,
)
from nova.equilibrium.rotation import RotatingDomainProfile
from nova.equilibrium.source import (
    DomainProfile,
    ForwardSource,
    PolynomialFluxFunction,
)
from nova.jax.config import configure_dtypes
from tests.test_forward_compile_identity import (
    CASES,
    REQUESTED_CELLS,
    _certificate_row,
)

OUTPUT = Path(
    os.environ.get(
        "PROFILE_RECOMPILE_OUTPUT",
        Path(__file__).with_name("profile-coefficients-recompile.json"),
    )
)
RUN_DIRECTORY = Path(os.environ.get("HLO_RUN_DIRECTORY", OUTPUT.parent))
RUN_LABEL = os.environ.get("SLURM_JOB_ID", "local")
CASE = CASES[0]
PRESSURE_SCALE = 0.9
DIAMAGNETIC_SCALE = 1.1

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


def _constant_locations(debug_text: str) -> dict[str, str]:
    """Map each stablehlo constant symbol to its XLA debug source location.

    The debug module carries ``#locN = loc(...)`` definitions (possibly
    chained as ``loc("name"(#locM))``) and each op annotates ``loc(#locN)``.
    Definitions are parsed with balanced parentheses; a reference resolves to
    the deepest readable ``path:line:column`` span reachable through the
    chain.
    """
    definitions: dict[str, str] = {}
    for match in re.finditer(r"#loc(\d+)\s*=\s*", debug_text):
        cursor = match.end()
        if not debug_text.startswith("loc(", cursor):
            continue
        depth = 0
        close = -1
        for pos in range(cursor, min(cursor + 1000, len(debug_text))):
            ch = debug_text[pos]
            if ch == "(":
                depth += 1
            elif ch == ")":
                depth -= 1
                if depth == 0:
                    close = pos
                    break
        if close != -1:
            definitions[f"#loc{match.group(1)}"] = debug_text[cursor : close + 1]

    resolved: dict[str, str] = {}
    resolving: set[str] = set()

    def resolve(reference: str) -> str:
        if reference in resolved:
            return resolved[reference]
        if reference in resolving:
            return reference
        raw = definitions.get(reference)
        if raw is None:
            return reference
        resolving.add(reference)
        location = ""
        for name, line, column in re.findall(r'loc\(\s*"([^"]+)":(\d+):(\d+)', raw):
            if line and not location and name.endswith(".py"):
                location = f"{name.rsplit('/', 1)[-1]}:{line}:{column}"
        if not location:
            nested = re.search(r"\((#loc\d+)\)", raw)
            if nested:
                location = resolve(nested.group(1))
            else:
                for name in re.findall(r'loc\(\s*"([^"]+)"\s*\)', raw):
                    if not any(char.isdigit() for char in name):
                        location = name
                        break
        resolving.discard(reference)
        resolved[reference] = location or reference
        return resolved[reference]

    locations: dict[str, str] = {}
    for match in re.finditer(
        r"(?P<symbol>%[A-Za-z0-9_]+)\s*=\s*stablehlo\.constant\s+"
        r"dense<.*?>\s*:\s*tensor<[^>]+>[^;]*?loc\((?P<loc>[^)]*)\)",
        debug_text,
        re.DOTALL,
    ):
        reference = match.group("loc").strip()
        location = resolve(reference)
        locations[match.group("symbol")] = (
            location if len(location) <= 120 else location[:117] + "..."
        )
    return locations


def _scaled_source(
    fixture_source: ForwardSource, pressure_scale: float, diamagnetic_scale: float
) -> ForwardSource:
    """Return a source whose flux functions are the fixture's, scaled.

    The fixture's flux functions are analytic ``jnp.full_like`` closures over
    one scalar gradient each; the scaled arm rebuilds those two closures with
    the same bodies and scaled closed-over scalars, so the trace structure is
    identical and only the coefficient constants move.  The
    ``RotatingDomainProfile`` is rebuilt with the fixture's declared rotation
    closure and reference-pressure primitive by identity, so pressure-gradient
    scaling reaches the residual through the gradient alone, as declared.
    """
    core = fixture_source.core
    p_gradient = _cell_contents(core.p_prime)
    f_gradient = _cell_contents(core.ff_prime)

    scaled_p_prime = PolynomialFluxFunction(
        jnp.asarray([pressure_scale * p_gradient], dtype=jnp.float64),
        jnp.asarray(1.0, dtype=jnp.float64),
    )
    scaled_ff_prime = PolynomialFluxFunction(
        jnp.asarray([diamagnetic_scale * f_gradient], dtype=jnp.float64),
        jnp.asarray(1.0, dtype=jnp.float64),
    )

    if type(core) is RotatingDomainProfile:
        scaled_core = RotatingDomainProfile(
            p_prime=scaled_p_prime,
            ff_prime=scaled_ff_prime,
            reference_pressure=core.reference_pressure,
            rotation=core.rotation,
        )
    elif type(core) is DomainProfile:
        scaled_core = DomainProfile(
            p_prime=scaled_p_prime,
            ff_prime=scaled_ff_prime,
        )
    else:
        raise TypeError(f"unsupported core profile {type(core).__qualname__}")
    return ForwardSource(
        core=scaled_core,
        boundary_pressure=float(fixture_source.boundary_pressure),
        boundary_field_function=float(fixture_source.boundary_field_function),
    )


def _scaled_row(fixture_row, scaled_source):
    """Return the same inputs with only the source's profile coefficients moved."""
    profile, seed, requested_class, target_current, _request = fixture_row
    scaled_request = dataclasses.replace(
        _request,
        source_profile=scaled_source,
        carrier_identity=f"solovev:{CASE}:{REQUESTED_CELLS}:scaled-profile",
    )
    return (
        profile,
        seed,
        requested_class,
        target_current,
        scaled_request,
    )


def _lower_certificate(row, external):
    """Return one lowered certificate solve programme as an HLO text pair."""
    active = row[0]._with_source(row[4].source_profile)
    program = active._accelerated_history_program(
        "newton_krylov",
        requested_class=row[2],
        target_current=row[3],
        **row[4].policy.kernel_options(),
    )
    lowered = program.lower(row[1], external, active.operator, jnp.asarray(row[3]))
    return lowered, lowered.as_text(dialect="stablehlo")


def _compiled_slice_module(operator, seed, requested_class, target_current):
    """Lower the compiled slice solver for one exterior current."""
    response = jnp.asarray(operator.external())[:, None]
    operator.prescribed_field = PrescribedCurrentField(
        response=response,
        current=jnp.asarray([1.0], dtype=response.dtype),
    )
    current = jnp.asarray([1.0], dtype=response.dtype)
    external = operator.external(prescribed_current=current)
    program, raw_kernels = reduced_newton._compiled_program(
        operator,
        seed,
        requested_class=requested_class,
        target_current=jnp.asarray(target_current),
        external=external,
        program=None,
    )
    kernels = reduced_newton._bind_dynamic_arguments(
        raw_kernels,
        external,
        operator,
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
    lowered = solver.lower(
        seed,
        shadow,
        external,
        operator,
        jnp.asarray(target_current),
        requested_class,
    )
    return lowered, lowered.as_text(dialect="stablehlo")


def _cell_contents(function: object) -> object:
    """Return the single captured value of a flux-function closure, if any."""
    cells = getattr(function, "__closure__", None)
    return cells[0].cell_contents if cells else None


def _inventory(fixture_source) -> dict[str, object]:
    """Report what the traced residual closes over, classified per closure item.

    Each row names the object, its definition location, and whether it reaches
    the compiled residual as a traced value, a captured constant, or a Python
    callable whose identity participates in the JAX cache key.
    """
    core = fixture_source.core
    _, source_children = _SourceLayout.flatten(fixture_source)
    pressure_layout, pressure_leaves = _CallableLayout.flatten(core.p_prime)
    field_layout, field_leaves = _CallableLayout.flatten(core.ff_prime)
    rows = [
        {
            "slot": "core.p_prime (flux function)",
            "definition": (
                "nova/equilibrium/source.py:285; closure "
                "scripts/analytic_oracle_fixtures/measure.py:161"
            ),
            "classification": "static polynomial evaluator with traced leaves",
            "notes": (
                f"coefficients {tuple(core.p_prime.coefficients.shape)} and scalar "
                "normalisation are explicit operator-pytree leaves"
            ),
        },
        {
            "slot": "core.ff_prime (flux function)",
            "definition": ("nova/equilibrium/source.py:286; closure measure.py:164"),
            "classification": "static polynomial evaluator with traced leaves",
            "notes": (
                f"coefficients {tuple(core.ff_prime.coefficients.shape)} and scalar "
                "normalisation are explicit operator-pytree leaves"
            ),
        },
        {
            "slot": "core.pressure_gradient(radius, psi_norm)",
            "definition": (
                "nova/equilibrium/source.py:297; RotatingDomainProfile "
                "override rotation.py:261"
            ),
            "classification": "method, runs inside the trace",
            "notes": "current_density reads it; the rotation algebra folds the "
            "reference pressure, exponent gradient and centrifugal factor "
            "from the shared rotation closure",
        },
        {
            "slot": "core.current_density(radius, psi_norm)",
            "definition": "nova/equilibrium/source.py:308; drives convention.py:68",
            "classification": "method, runs inside the trace",
            "notes": "enters the residual through support_current_moments "
            "(forward_operator.py:2058) and the moment stencil current_density "
            "(stencil_mesh.py:485)",
        },
        {
            "slot": "core.rotation (IsothermalRotation)",
            "definition": "rotation.py:110; built measure.py:113",
            "classification": "Python object of callables, inlined at trace",
            "notes": "temperature / angular_frequency / gradient closures close over "
            "the axis and boundary temperature, rotation parameter and mass "
            "(measure.py:115-119), captured constants; zero for the static case",
        },
        {
            "slot": "core.reference_pressure",
            "definition": "rotation.py:221; closure measure.py:167",
            "classification": "Python callable, inlined at trace",
            "notes": "closed over axis_pressure (measure.py:168), a captured constant",
        },
        {
            "slot": "core.pressure / field_function_squared primitives",
            "definition": "source.py:314 / rotation.py:277; observation.py:476",
            "classification": "method, host receipt path",
            "notes": "read by the observation receipts, not by the compiled "
            "fixed-point map; PROFILE_NODES = 257 (observation.py:106) fixes "
            "the gradient_tail node count (observation.py:450) there",
        },
        {
            "slot": "source.boundary_pressure",
            "definition": (
                "source.py:495; converted to a jnp scalar forward_operator.py:1361"
            ),
            "classification": "captured constant array (f64[1])",
            "notes": "not read by the compiled map (pressure receipts only)",
        },
        {
            "slot": "source.boundary_field_function",
            "definition": "source.py:496; converted forward_operator.py:1364",
            "classification": "captured constant array (f64[1])",
            "notes": "not read by the compiled map",
        },
        {
            "slot": "source.normalisation (policy)",
            "definition": "source.py:499",
            "classification": "static IntEnum, host side only",
            "notes": "the shipped forward closure preserves the source absolutely; "
            "it never reaches the traced residual",
        },
    ]
    return {
        "source_layout_identity": _SourceLayout.flatten(fixture_source)[0].identity,
        "source_children": [_child_summary(child) for child in source_children],
        "p_prime": {
            "callable_layout_identity": pressure_layout.identity,
            "dynamic_leaves": [
                f"array({leaf.shape}, {leaf.dtype})" for leaf in pressure_leaves
            ],
        },
        "ff_prime": {
            "callable_layout_identity": field_layout.identity,
            "dynamic_leaves": [
                f"array({leaf.shape}, {leaf.dtype})" for leaf in field_leaves
            ],
        },
        "rows": rows,
    }


def _child_summary(child: object) -> object:
    if hasattr(child, "shape") and hasattr(child, "dtype"):
        return f"array(shape={tuple(child.shape)}, dtype={child.dtype})"
    return repr(child)


def _minimal_argument_set(fixture_source, inventory) -> dict[str, object]:
    """Estimate the fixed-shape arrays one shared programme would need."""
    core = fixture_source.core
    _, source_children = _SourceLayout.flatten(fixture_source)
    arguments = [
        {
            "name": "pressure_coefficients",
            "shape": tuple(core.p_prime.coefficients.shape),
            "dtype": str(core.p_prime.coefficients.dtype),
            "bytes": int(core.p_prime.coefficients.nbytes),
        },
        {
            "name": "pressure_normalisation",
            "shape": (),
            "dtype": str(core.p_prime.normalisation.dtype),
            "bytes": int(core.p_prime.normalisation.nbytes),
        },
        {
            "name": "diamagnetic_coefficients",
            "shape": tuple(core.ff_prime.coefficients.shape),
            "dtype": str(core.ff_prime.coefficients.dtype),
            "bytes": int(core.ff_prime.coefficients.nbytes),
        },
        {
            "name": "diamagnetic_normalisation",
            "shape": (),
            "dtype": str(core.ff_prime.normalisation.dtype),
            "bytes": int(core.ff_prime.normalisation.nbytes),
        },
    ]
    total_bytes = sum(item["bytes"] for item in arguments)
    return {
        "minimal_argument_set": arguments,
        "argument_count": len(arguments),
        "argument_bytes": total_bytes,
        "boundary_primitive_bytes": sum(
            np.asarray(child).nbytes for child in source_children
        ),
        "notes": "the two fixed-shape coefficient vectors and their physical "
        "normalisations vary per slice; evaluator code, basis order, mesh and "
        "solve policy remain static",
    }


def _representation_study(fixture_source) -> dict[str, object]:
    """Record the evaluated representation choice and its argument costs."""
    pressure_count = int(fixture_source.core.p_prime.coefficients.size)
    diamagnetic_count = int(fixture_source.core.ff_prime.coefficients.size)
    return {
        "selected": "fixed-order power basis",
        "selected_orders": {
            "pressure": pressure_count - 1,
            "diamagnetic": diamagnetic_count - 1,
        },
        "alternatives": {
            "sampled_nodes": {
                "interpolation": "piecewise linear",
                "coefficient_count": "one value per fixed node plus its coordinate",
                "reading": "already represented by SampledFluxFunction; cheap and "
                "traceable but not higher order",
            },
            "fixed_order_polynomial": {
                "interpolation": "global power basis evaluated by Horner recurrence",
                "coefficient_count": (
                    "order plus one and one normalisation per function"
                ),
                "reading": "selected for analytic profile families: differentiable, "
                "fixed shape and four small per-slice arguments",
            },
            "b_spline": {
                "interpolation": "piecewise higher order",
                "coefficient_count": "control coefficients plus a fixed knot policy",
                "reading": "retained for a future local higher-order family; it adds "
                "basis and knot semantics the constant analytic rows do not need",
            },
        },
    }


def _write_hlo_difference(fixture_text, scaled_text, fixture_debug, scaled_debug):
    """Persist both modules, their textual diff and the differing constants."""
    RUN_DIRECTORY.mkdir(parents=True, exist_ok=True)
    fixture_path = RUN_DIRECTORY / f"profile-fixture-{RUN_LABEL}.stablehlo"
    scaled_path = RUN_DIRECTORY / f"profile-scaled-{RUN_LABEL}.stablehlo"
    diff_path = RUN_DIRECTORY / f"profile-{RUN_LABEL}.stablehlo.diff"
    constants_path = RUN_DIRECTORY / f"profile-{RUN_LABEL}-constants.json"
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
    fixture_locations = _constant_locations(fixture_debug)
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
                "source_location": (
                    fixture_locations.get((first or second)["symbol"], "unresolved")
                    if (first or second) is not None
                    else "unresolved"
                ),
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
        f"STAGE stablehlo artifacts written fixture={fixture_path} "
        f"scaled={scaled_path} diff={diff_path} constants={constants_path} "
        f"differing={len(differing)}",
        flush=True,
    )
    return census, constants_path


def _profile_figure(row, scaled_row) -> Path:
    """Plot the committed against the scaled flux functions over normalised flux."""
    domain = np.linspace(0.0, 1.0, 129)
    source = row[4].source_profile
    scaled_source = scaled_row[4].source_profile
    figure, axes = plt.subplots(1, 2, figsize=(9, 3.6))
    for axis, function, label in (
        (axes[0], "p_prime", "pressure gradient p'"),
        (axes[1], "ff_prime", "diamagnetic gradient FF'"),
    ):
        fixture = np.asarray(getattr(source.core, function)(jnp.asarray(domain)))
        scaled = np.asarray(getattr(scaled_source.core, function)(jnp.asarray(domain)))
        axis.plot(domain, fixture, label="committed")
        if function == "p_prime":
            scale = PRESSURE_SCALE
        else:
            scale = DIAMAGNETIC_SCALE
        axis.plot(domain, scaled, label=f"scaled {scale}x")
        axis.set_xlabel("normalised flux")
        axis.set_title(label)
        axis.legend()
    figure.tight_layout()
    repo_root = Path(__file__).resolve().parents[1]
    directory = Path(
        os.environ.get(
            "PROFILE_RECOMPILE_FIGURE_DIRECTORY",
            repo_root
            / "docs"
            / "figures"
            / "millisecond-converged-solve"
            / "flux-arguments",
        )
    )
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "flux-functions.png"
    figure.savefig(path, dpi=110)
    plt.close(figure)
    return path


def main() -> int:
    configure_dtypes()
    assert jax.config.jax_enable_x64 is True
    set_support_clip_mode("chord")
    assert support_clip_mode() == "chord"

    print("STAGE load weak cached 300-cell certificate row", flush=True)
    closure_row = _certificate_row(CASE)
    fixture_source = _scaled_source(closure_row[0].source, 1.0, 1.0)
    fixture_row = _scaled_row(closure_row, fixture_source)
    fixture_profile, seed, requested_class, target_current, _ = fixture_row
    external = fixture_profile.operator.external()

    print("STAGE build scaled-source profile", flush=True)
    scaled_source = _scaled_source(
        closure_row[0].source, PRESSURE_SCALE, DIAMAGNETIC_SCALE
    )
    scaled_row = _scaled_row(fixture_row, scaled_source)

    print("STAGE certificate lower committed source profile", flush=True)
    fixture_lowered, fixture_text = _lower_certificate(fixture_row, external)
    print("STAGE certificate lower scaled source profile", flush=True)
    scaled_lowered, scaled_text = _lower_certificate(scaled_row, external)
    fixture_debug = fixture_lowered.as_text(dialect="stablehlo", debug_info=True)
    scaled_debug = scaled_lowered.as_text(dialect="stablehlo", debug_info=True)
    census, constants_path = _write_hlo_difference(
        fixture_text, scaled_text, fixture_debug, scaled_debug
    )
    certificate_digests = (
        hashlib.sha256(fixture_text.encode()).hexdigest(),
        hashlib.sha256(scaled_text.encode()).hexdigest(),
    )
    print("STAGE compile committed and scaled profile programs", flush=True)
    compile_walls = []
    for lowered in (fixture_lowered, scaled_lowered):
        started = time.perf_counter()
        lowered.compile()
        compile_walls.append(time.perf_counter() - started)

    print("STAGE compiled-slice lowering, committed and scaled", flush=True)
    fixture_operator = fixture_profile._with_source(
        fixture_row[4].source_profile
    ).operator
    slice_fixture, slice_fixture_text = _compiled_slice_module(
        fixture_operator, seed, requested_class, target_current
    )
    slice_scaled, slice_scaled_text = _compiled_slice_module(
        scaled_row[0]._with_source(scaled_row[4].source_profile).operator,
        seed,
        requested_class,
        target_current,
    )
    slice_digests = (
        hashlib.sha256(slice_fixture_text.encode()).hexdigest(),
        hashlib.sha256(slice_scaled_text.encode()).hexdigest(),
    )

    print("STAGE inventory and minimal argument set", flush=True)
    argument_source = fixture_row[4].source_profile
    inventory = _inventory(argument_source)
    minimal = _minimal_argument_set(argument_source, inventory)

    print("STAGE profile figure", flush=True)
    figure_path = _profile_figure(fixture_row, scaled_row)
    figure_served = (
        "/nova/figures/millisecond-converged-solve/flux-arguments/flux-functions.png"
    )

    receipt = {
        "case": CASE,
        "clip_mode": support_clip_mode(),
        "requested_cells": REQUESTED_CELLS,
        "pressure_scale": PRESSURE_SCALE,
        "diamagnetic_scale": DIAMAGNETIC_SCALE,
        "external_shape": list(external.shape),
        "differing_constants": str(constants_path),
        "figure": figure_served,
        "figure_file": str(figure_path),
        "certificate": {
            "fixture_sha256": certificate_digests[0],
            "scaled_sha256": certificate_digests[1],
            "identical": certificate_digests[0] == certificate_digests[1],
            "fixture_backend_compile_seconds": compile_walls[0],
            "scaled_backend_compile_seconds": compile_walls[1],
        },
        "compiled_slice": {
            "fixture_sha256": slice_digests[0],
            "scaled_sha256": slice_digests[1],
            "identical": slice_digests[0] == slice_digests[1],
        },
        "differing_constant_summary": census,
        "inventory": inventory,
        "minimal_argument_set": minimal,
        "representation_study": _representation_study(argument_source),
    }
    OUTPUT.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(f"STAGE receipt written {OUTPUT}", flush=True)
    print(json.dumps(receipt, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
