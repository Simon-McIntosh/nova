"""Restore the exit-loop stream whose default batching selects the carry.

``install`` rebinds ``_single_site_krylov`` in the live fixed-point module to
its body at the exit-loop revision: one ``lax.while_loop`` with no batching
rule of its own, so under ``vmap`` the loop's default rule selects the whole
carry against every member's exit on every slot.
"""

import ast
import subprocess

from nova.equilibrium import fixed_point

BASE = "63a32bb4a041682c1d4fe4e41cd7528fefdb8d91"
MUTATION = "restore the per-slot select of the whole carry under vmap"


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
