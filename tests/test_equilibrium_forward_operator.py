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

import os
from pathlib import Path
import subprocess
import sys
import types

import numpy as np

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


def _patched_globals_functions(namespace) -> list[str]:
    """Names in ``namespace`` whose function carries a matching module's dict.

    A function's globals dictionary is its own defining module's namespace, so
    a function that resolves its ``__module__`` elsewhere and yet does not
    carry that module's dictionary was built over a copy of it -- a
    patched-globals clone, whatever it is called.
    """
    found = []
    for name, value in namespace.items():
        if not isinstance(value, types.FunctionType):
            continue
        defining = sys.modules.get(getattr(value, "__module__", None) or "")
        if defining is not None and value.__globals__ is not vars(defining):
            found.append(name)
    return sorted(found)


def test_no_module_function_carries_a_foreign_globals_dictionary():
    """The module exposes no patched-globals clone of another module's function.

    The detector is exercised on a clone built here first, so a pass cannot
    come from a detector that never fires.
    """
    from nova.equilibrium import forward_operator
    from nova.equilibrium import separatrix_clip

    library = dict(vars(separatrix_clip))
    clone = types.FunctionType(
        separatrix_clip._traced_clip.__code__,
        library,
        name="_cloned_traced_clip",
        argdefs=separatrix_clip._traced_clip.__defaults__,
        closure=separatrix_clip._traced_clip.__closure__,
    )
    assert _patched_globals_functions({"clone": clone}) == ["clone"]

    assert _patched_globals_functions(vars(forward_operator)) == []
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
