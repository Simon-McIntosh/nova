"""Shared machinery for the hand-written tangent identity and compile rows.

The three tangent test modules load their base revision's source through
``git show``, compare this revision's tangents against ``jax.jvp`` of that base.
They apply the ``NOVA_TANGENT_TRUNCATION`` control switch, and they measure a
cold, cache-disabled compile in fresh processes against the primal.  All of
that lives here once; each module supplies only its cases, its truncations and a
``compile_arm`` selector.

The cold compile is measured on the median of ``COMPILE_REPEATS`` fresh
processes per arm, because a single process's wall on a program this small moves
by tens of percent with scheduler noise, and the expanded equation count is
recorded beside it as the deterministic measure of the program's size.  A row
whose median ratio exceeds ``COMPILE_REPORT`` is named in the evidence fragment
even when it passes ``COMPILE_BOUND``.
"""

import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

import jax
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SAMPLES = 10_000
EXACT_TOLERANCE = 1e-13
COMPILE_REPEATS = 7
COMPILE_BOUND = 3.0
COMPILE_REPORT = 2.7
_TRUNCATION_ENV = "NOVA_TANGENT_TRUNCATION"


def truncation_active():
    """Return whether the declared negative control is switched on."""
    return os.environ.get(_TRUNCATION_ENV) == "1"


def load_module(name, path):
    """Load a source file under ``name``, distinct from the package module."""
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_base_module(relative_path, revision, module_name=None):
    """Load a base revision's source file under its own distinct module name.

    The source is read out of git into a fresh scratch directory, so the module
    the identity row differentiates is the base revision's own code and not this
    worktree's.
    """
    stem = Path(relative_path).stem
    module_name = module_name or f"base_{stem}"
    directory = Path(tempfile.mkdtemp(prefix=f"{stem}-base-"))
    path = directory / Path(relative_path).name
    path.write_bytes(
        subprocess.check_output(
            ["git", "-C", str(ROOT), "show", f"{revision}:{relative_path}"]
        )
    )
    module = load_module(module_name, path)
    print(f"BASE_MODULE {stem}={module.__file__}", flush=True)
    return module


def relative_error(got, reference):
    """Return the per-element relative error, with an exact-zero scale of one."""
    got, reference = np.asarray(got), np.asarray(reference)
    agree = (np.isnan(got) & np.isnan(reference)) | (got == reference)
    scale = np.where(reference == 0.0, 1.0, np.abs(reference))
    error = np.where(agree, 0.0, np.abs(got - reference) / scale)
    return np.nan_to_num(error, nan=np.inf)


def leaves(tree):
    return [np.asarray(leaf) for leaf in jax.tree.leaves(tree)]


def per_sample_worst(got, reference):
    """Return the largest relative error over leaves, per sample."""
    errors = [
        relative_error(a, b)
        for a, b in zip(jax.tree.leaves(got), jax.tree.leaves(reference), strict=True)
    ]
    return np.max(np.stack(errors), axis=0)


def worst(got, reference):
    """Return the largest relative error over every leaf and sample."""
    return float(per_sample_worst(got, reference).max())


def finite_fraction(tree):
    """Return the fraction of samples on which every leaf of ``tree`` is finite."""
    values = [np.isfinite(np.asarray(leaf)) for leaf in jax.tree.leaves(tree)]
    return float(np.mean(np.all(np.stack([np.ravel(v) for v in values]), axis=0)))


def identity_row(name, primal, tangent, bound, *, extra=""):
    """Assert and report one identity row from its computed primal/tangent errors.

    Each module computes its own errors -- masked, or against a reference other
    than the base jvp -- but the row's print and its two assertions live here so
    every module reports and enforces the identity the same way.
    """
    print(
        f"IDENTITY {name} primal_max_relative={primal:.3e} "
        f"tangent_max_relative={tangent:.3e} bound={bound:.0e} {extra}"
    )
    assert primal == 0.0
    assert tangent <= bound
    return tangent


_COMPILE_PROBE = r"""
import json, sys, time
import jax
jax.config.update("jax_enable_compilation_cache", False)
from nova.jax.config import configure_dtypes
configure_dtypes()
assert jax.config.jax_enable_x64 is True
hits = []
jax.monitoring.register_event_listener(
    lambda event, **kw: hits.append(event) if "cache_hit" in event else None
)
directory, module_name, name, arm = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
sys.path.insert(0, directory)
module = __import__(module_name)
case = module.CASES[name]
arguments = (case[-2], case[-1])
function = module.compile_arm(name, arm)
def count(jaxpr):
    total = 0
    for equation in jaxpr.eqns:
        total += 1
        for value in equation.params.values():
            for sub in value if isinstance(value, (list, tuple)) else [value]:
                inner = getattr(sub, "jaxpr", sub)
                if hasattr(inner, "eqns"):
                    total += count(inner)
    return total
equations = count(jax.make_jaxpr(function)(*arguments).jaxpr)
lowered = jax.jit(function).lower(*arguments)
start = time.perf_counter()
lowered.compile()
wall = time.perf_counter() - start
print(json.dumps({"name": name, "arm": arm, "compile_seconds": wall,
                  "equations": equations, "cache_hits": len(hits),
                  "cache_enabled": jax.config.jax_enable_compilation_cache}))
"""


def compile_row(module_name, name, arm):
    """Return one cold, cache-disabled compile row from a fresh process."""
    environment = dict(os.environ, JAX_ENABLE_COMPILATION_CACHE="false")
    environment.pop(_TRUNCATION_ENV, None)
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            _COMPILE_PROBE,
            str(Path(__file__).parent),
            module_name,
            name,
            arm,
        ],
        capture_output=True,
        text=True,
        env=environment,
        check=True,
    )  # fmt: skip
    row = json.loads(result.stdout.strip().splitlines()[-1])
    assert row["cache_hits"] == 0 and row["cache_enabled"] is False
    return row


def compile_ratio(module_name, name, repeats=COMPILE_REPEATS):
    """Return the median tangent-over-primal compile ratio, printing every arm."""
    rows = {
        arm: [compile_row(module_name, name, arm) for _ in range(repeats)]
        for arm in ("primal", "tangent", "jvp")
    }
    median = {}
    for arm, runs in rows.items():
        walls = sorted(row["compile_seconds"] for row in runs)
        equations = {row["equations"] for row in runs}
        assert len(equations) == 1
        median[arm] = float(np.median(walls))
        print(
            f"COMPILE {name} {arm} median_seconds={median[arm]:.3f} "
            f"equations={equations.pop()} "
            f"walls={','.join(f'{wall:.3f}' for wall in walls)} "
            f"hits={sum(row['cache_hits'] for row in runs)}"
        )  # fmt: skip
    ratio = median["tangent"] / median["primal"]
    print(f"COMPILE {name} median_tangent_over_primal={ratio:.2f}")
    return ratio