"""Restore the fixed-capacity, cond-gated scan stream beside the exit loop.

``install`` rebinds ``_single_site_krylov`` in the live fixed-point module to
the body at the base revision, which serves the operator from a
``lax.scan`` of fixed capacity and gates each slot with ``lax.cond``. Under
``vmap`` that cond becomes a select, so every slot applies the operator.
"""

import ast
import subprocess

from nova.equilibrium import fixed_point

BASE = "83e226f7c72d5bdf2a207591e6f40466953178a4"
MUTATION = "restore the fixed-capacity cond-gated scan so every slot runs under vmap"


def _base_source():
    text = subprocess.check_output(
        ["git", "show", f"{BASE}:nova/equilibrium/fixed_point.py"], text=True
    )
    for node in ast.parse(text).body:
        if isinstance(node, ast.FunctionDef) and node.name == "_single_site_krylov":
            return ast.get_source_segment(text, node)
    raise LookupError("base revision has no _single_site_krylov")


def install():
    namespace = fixed_point.__dict__
    exec(compile(_base_source(), fixed_point.__file__, "exec"), namespace)
    print(f"MUTATION_APPLIED {MUTATION}", flush=True)
    return namespace["_single_site_krylov"]
