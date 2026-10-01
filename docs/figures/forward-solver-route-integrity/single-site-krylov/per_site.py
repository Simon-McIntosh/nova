"""Restore the per-site Krylov step of the base revision beside the stream.

``install`` rebinds ``_qualified_krylov_step`` in the live fixed-point module
to the base revision's body, which calls the operator directly at the probe,
the condition Arnoldi column, the three GMRES application sites and the
achieved-residual check. ``base_step`` returns that body under its own name so
the two can be compared in one process.
"""

import ast
import subprocess

from nova.equilibrium import fixed_point

BASE = "52bfefc0a0f02412d8910aaccb632bfbd9fd43b5"
MUTATION = "call the operator directly at each GMRES application site again"


def _base_source():
    text = subprocess.check_output(
        ["git", "show", f"{BASE}:nova/equilibrium/fixed_point.py"], text=True
    )
    for node in ast.parse(text).body:
        if isinstance(node, ast.FunctionDef) and node.name == "_qualified_krylov_step":
            return ast.get_source_segment(text, node)
    raise LookupError("base revision has no _qualified_krylov_step")


def base_step():
    namespace = dict(fixed_point.__dict__)
    exec(compile(_base_source(), "<base _qualified_krylov_step>", "exec"), namespace)
    return namespace["_qualified_krylov_step"]


def install():
    step = base_step()
    step.__globals__.update(
        {k: v for k, v in fixed_point.__dict__.items() if k != "_qualified_krylov_step"}
    )
    fixed_point._qualified_krylov_step = step
    print(f"MUTATION_APPLIED {MUTATION}", flush=True)
    return step
