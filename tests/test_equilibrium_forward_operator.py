"""The forward operator's exact clip reaches its tracer through the clip.

The exact clip needs an implicit-derivative level-root tangent, which it used
to obtain by cloning the traced clip with a patched copy of another module's
globals dictionary. A clone like that is invisible at the clip's own call site
and its call chain, and the module namespace is the only place a reader can see
it, so these tests pin the parameter route: the operator passes its tracer to
the clip, and no module attribute is a function carrying a foreign globals
dictionary.
"""

from __future__ import annotations

from dataclasses import replace
import os
from pathlib import Path
import subprocess
import sys
import types

import jax
import jax.numpy as jnp
import numpy as np

from nova.equilibrium.forward_operator import ForwardFluxOperator, PrescribedCurrentField
from nova.equilibrium.stencil_mesh import CellCurrentMoments
from nova.jax.config import configure_dtypes

WORKTREE = Path(__file__).resolve().parents[1]

#: Builds the weak-rotation-reactor certificate operator, spies on every traced
#: clip call the exact support raises, and reports the tracer each call carried.
DRIVER = """\
\"\"\"Report the arc tracer the forward operator's exact clip passes.\"\"\"

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def build():
    \"\"\"Return the operator and the exact certificate state for the case.\"\"\"
    from nova.jax.config import configure_dtypes
    from scripts.analytic_oracle_fixtures import measure as oracle_fixture
    from tests.rotating_equilibrium_references import reference_cases

    configure_dtypes()
    case = reference_cases()["weak-rotation-reactor"].static_limit()
    machine = oracle_fixture.cached_machine(
        case,
        -110,
        wall_nodes=oracle_fixture.WALL_POINT_COUNT,
    )
    operator = oracle_fixture.forward_operator(case, machine)
    coordinates = np.vstack(
        (machine.node, machine.wall_node,
         machine.sample_coordinates)
    )
    state = oracle_fixture.exact_state(case, coordinates)
    return operator, np.asarray(state, dtype=np.float64)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    parser.add_argument("--mode", choices=("chord", "exact"), required=True)
    arguments = parser.parse_args()

    import jax.numpy as jnp

    from nova.equilibrium import forward_operator as fo

    operator, state = build()
    fo.set_support_clip_mode(arguments.mode)

    observed: list[str] = []
    original = fo._traced_clip

    def spy(*args, **kwargs):
        tracer = kwargs.get("arc_tracer")
        if tracer is fo._implicit_traced_level_arc:
            observed.append("implicit")
        elif tracer is None:
            observed.append("default")
        else:
            observed.append("other")
        return original(*args, **kwargs)

    fo._traced_clip = spy
    try:
        moments = operator.cell_current_moments(jnp.asarray(state))
    finally:
        fo._traced_clip = original

    np.savez(
        arguments.output,
        tracers=np.asarray(observed, dtype="<U8"),
        call_count=np.asarray(len(observed), dtype=np.int64),
        cell_current=np.asarray(moments.cell_current, dtype=np.float64),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
"""


def _run_driver(tmp_path: Path, mode: str) -> dict:
    """Run the tracer spy against this worktree and return its receipt."""
    driver = tmp_path / "tracer_spy_driver.py"
    driver.write_text(DRIVER, encoding="utf-8")
    output = tmp_path / f"tracer-spy-{mode}.npz"
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(WORKTREE)
    environment["JAX_PLATFORMS"] = "cpu"
    result = subprocess.run(
        [sys.executable, str(driver), str(output), "--mode", mode],
        cwd=str(WORKTREE),
        env=environment,
        capture_output=True,
        text=True,
        timeout=1800,
    )
    assert result.returncode == 0, result.stderr[-4000:]
    with np.load(output) as receipt:
        return {name: receipt[name] for name in receipt.files}


def _stale_clip_clones(namespace) -> list[str]:
    """Names in ``namespace`` holding a patched copy of the clip's namespace.

    A patched-globals clone is exactly a function whose globals dictionary
    claims to be the clip module's namespace and is not that module's own
    dictionary. Functions legitimately imported from the clip carry its real
    dictionary, so a pass here cannot come from an import.
    """
    from nova.equilibrium import separatrix_clip

    live = vars(separatrix_clip)
    return sorted(
        name
        for name, value in namespace.items()
        if isinstance(value, types.FunctionType)
        and value.__globals__.get("__name__") == separatrix_clip.__name__
        and value.__globals__ is not live
    )


def test_no_module_attribute_clones_the_clip_with_patched_globals():
    """The operator module holds no patched-globals clone of the traced clip.

    The detector is exercised on a clone built here first, so a pass cannot
    come from a detector that never fires.
    """
    from nova.equilibrium import forward_operator
    from nova.equilibrium import separatrix_clip

    library = dict(vars(separatrix_clip))
    library["_traced_level_arc"] = separatrix_clip._traced_level_arc
    clone = types.FunctionType(
        separatrix_clip._traced_clip.__code__,
        library,
        name="_cloned_traced_clip",
        argdefs=separatrix_clip._traced_clip.__defaults__,
        closure=separatrix_clip._traced_clip.__closure__,
    )
    assert _stale_clip_clones({"clone": clone}) == ["clone"]

    assert _stale_clip_clones(vars(forward_operator)) == []
    assert "_implicit_traced_level_arc" in vars(forward_operator)


def test_exact_clip_passes_the_implicit_tracer_by_parameter(tmp_path):
    """The exact support reaches the implicit tracer through the clip argument.

    The spy sits on the clip the operator module holds, in a fresh process, so
    it sees the argument the production path passes. The same driver in the
    committed chord mode raises no traced clip call at all, which is the
    control that the client is not reporting a call from another route.
    """
    configure_dtypes()
    chord = _run_driver(tmp_path, "chord")
    exact = _run_driver(tmp_path, "exact")

    assert int(chord["call_count"]) == 0
    tracers = [str(entry) for entry in exact["tracers"]]
    assert tracers, "the exact clip raised no traced clip call"
    assert set(tracers) == {"implicit"}


def test_exact_clip_moments_are_finite_on_a_clipped_state(tmp_path):
    """The measured state carries real currents, not an empty receipt."""
    configure_dtypes()
    exact = _run_driver(tmp_path, "exact")
    current = np.asarray(exact["cell_current"], dtype=np.float64)
    assert current.size > 0
    assert np.all(np.isfinite(current))
    assert np.count_nonzero(current) > 1
    assert float(np.ptp(current)) > 0.0


def _field_response(
    *, rows: int, circuits: int, cells: int, spy, flux_rows: int | None = None
):
    """Build distinguishable field blocks while counting kernel evaluations."""

    def interaction(shape, offset):
        spy.append((shape, offset))
        return jnp.arange(np.prod(shape), dtype=jnp.float64).reshape(shape) + offset

    return PrescribedCurrentField(
        response=interaction(
            (rows + 2 if flux_rows is None else flux_rows, circuits), 1.0
        ),
        current=jnp.arange(1.0, circuits + 1.0),
        radial_response=interaction((rows, circuits), 10.0),
        vertical_response=interaction((rows, circuits), 20.0),
        plasma_radial_response=tuple(
            interaction((rows, cells), offset) for offset in (30.0, 40.0, 50.0)
        ),
        plasma_vertical_response=tuple(
            interaction((rows, cells), offset) for offset in (60.0, 70.0, 80.0)
        ),
    )


def test_grid_field_response_builds_once_and_reuses_exact_interactions():
    """A second field evaluation changes currents without rebuilding kernels."""
    configure_dtypes()
    assert jax.config.jax_enable_x64 is True
    kernel_evaluations = []
    field = _field_response(rows=4, circuits=2, cells=3, spy=kernel_evaluations)
    built_count = len(kernel_evaluations)
    assert built_count == 9, "the positive control did not build every interaction"
    moments = CellCurrentMoments(
        jnp.asarray((1.0, 2.0, 3.0)),
        jnp.asarray((4.0, 5.0, 6.0)),
        jnp.asarray((7.0, 8.0, 9.0)),
    )

    first = field.poloidal_field(moments)
    edited = jnp.asarray((-3.0, 5.0))
    second = field.poloidal_field(moments, edited)

    assert len(kernel_evaluations) == built_count
    expected_first_radial = field.radial_response @ field.current + sum(
        response @ moment
        for response, moment in zip(field.plasma_radial_response, moments, strict=True)
    )
    expected_second_vertical = field.vertical_response @ edited + sum(
        response @ moment
        for response, moment in zip(
            field.plasma_vertical_response, moments, strict=True
        )
    )
    np.testing.assert_array_equal(first.radial, expected_first_radial)
    np.testing.assert_array_equal(second.vertical, expected_second_vertical)


def test_field_responses_are_pytree_children_and_share_one_identity():
    """Flux and field blocks survive a pytree round trip under one digest."""
    configure_dtypes()
    field = _field_response(rows=4, circuits=2, cells=3, spy=[])
    leaves, auxiliary = field.tree_flatten()
    assert len(leaves) == 10
    restored = PrescribedCurrentField.tree_unflatten(auxiliary, leaves)

    assert restored.response_identity == field.response_identity
    np.testing.assert_array_equal(restored.radial_response, field.radial_response)
    np.testing.assert_array_equal(restored.vertical_response, field.vertical_response)
    for rebuilt, original in zip(
        restored.plasma_radial_response, field.plasma_radial_response, strict=True
    ):
        np.testing.assert_array_equal(rebuilt, original)


def test_operator_pytree_threads_every_field_response_and_identity():
    """The operator digest and pytree cover the exact field matrices."""
    from tests.test_forward_operator_arguments import _operator

    configure_dtypes()
    template = _operator()
    field = _field_response(
        rows=template.grid.node_number,
        circuits=2,
        cells=template.grid.node_number,
        flux_rows=template.node_number,
        spy=[],
    )
    operator = replace(template, prescribed_current_field=field)
    children, auxiliary = operator.tree_flatten()
    restored = ForwardFluxOperator.tree_unflatten(auxiliary, children)

    assert operator.geometry_identity != template.geometry_identity
    np.testing.assert_array_equal(
        restored.prescribed_field.radial_response, field.radial_response
    )
    for rebuilt, original in zip(
        restored.prescribed_field.plasma_vertical_response,
        field.plasma_vertical_response,
        strict=True,
    ):
        np.testing.assert_array_equal(rebuilt, original)


def test_field_response_refuses_partial_or_mis_shaped_blocks():
    """A field carrier cannot silently omit one exact interaction family."""
    configure_dtypes()
    with np.testing.assert_raises_regex(ValueError, "supplied together"):
        PrescribedCurrentField(
            response=jnp.ones((5, 2)),
            current=jnp.ones(2),
            radial_response=jnp.ones((3, 2)),
        )
    with np.testing.assert_raises_regex(ValueError, "plasma field response"):
        PrescribedCurrentField(
            response=jnp.ones((5, 2)),
            current=jnp.ones(2),
            radial_response=jnp.ones((3, 2)),
            vertical_response=jnp.ones((3, 2)),
            plasma_radial_response=(
                jnp.ones((3, 4)),
                jnp.ones((3, 4)),
                jnp.ones((2, 4)),
            ),
            plasma_vertical_response=(jnp.ones((3, 4)),) * 3,
        )
