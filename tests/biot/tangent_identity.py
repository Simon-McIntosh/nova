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

The ratio is taken on CPU seconds rather than wall: under concurrent load a
tangent and a primal are descheduled by unrelated work, so their wall times move
together and the ratio crosses the bound without the program having changed.
CPU seconds count only the work the process did, so the same bound holds whether
the machine is idle or saturated.  Each fresh process reports the CPU seconds it
spent inside ``lowered.compile()`` (``time.process_time()`` before and after, all
threads), which is the compile's own cost isolated from the interpreter start.

That figure is process-wide CPU, not single-thread CPU: XLA compiles on several
threads at once, so it sums the work those threads do, and a host with a
different core count can shift it in absolute terms, so a rebaseline on another
machine is not by itself a regression.  What the four passes establish is that
the measure is insensitive to unrelated load: every row moves by at most 0.2 in
tangent-over-primal ratio between an idle sequential pass and a pass under eight
spinning processes, while the wall seconds of the same rows rise by about a
third.  A future change to the ratio should be read beside the equation counts,
which are deterministic and machine-independent.

The whole child's CPU seconds, from ``getrusage(RUSAGE_CHILDREN)`` deltas around
the spawn, is recorded beside it but is not the asserted measure: a fresh child
spends several seconds of CPU importing the interpreter and the module before it
compiles anything, and that constant swamps the sub-second compile, so its ratio
would sit near one and the row would measure the import rather than the program.
"""

import importlib.util
import json
import os
from pathlib import Path
import resource
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
# The negative-control switch: with NOVA_TANGENT_COMPILE_HEAVY=1 the tangent
# program is replaced by HEAVY_COPIES independent copies of itself summed.  Each
# copy sees its arguments scaled by a different factor, so the copies are not
# the same computation and the compiler cannot fold them into one; four
# identical copies would be common-subexpression-eliminated and would barely
# move the ratio at all.
HEAVY_COPIES = 4
import json, os, sys, time
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
copies = 1
if arm == "tangent":
    switch = os.environ.get("NOVA_TANGENT_COMPILE_HEAVY", "")
    if switch == "1":
        copies = HEAVY_COPIES
    elif switch:
        copies = int(switch)
    if copies > 1:
        inner = function

        def function(*arguments):
            pieces = []
            for index in range(copies):
                factor = 1.0 + 0.1 * index
                scaled = jax.tree.map(lambda leaf: leaf * factor, arguments)
                pieces.append(inner(*scaled))
            return jax.tree.map(lambda *leaves: sum(leaves), *pieces)
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
cpu_start = time.process_time()
start = time.perf_counter()
lowered.compile()
# process-wide CPU summed over XLA's compile threads, so the absolute seconds
# move with the host's core count; the ratio between the two arms is what the
# bound is asserted on, and it is load-insensitive (see the module docstring).
wall = time.perf_counter() - start
cpu = time.process_time() - cpu_start
print(json.dumps({"name": name, "arm": arm, "compile_seconds": wall,
                  "compile_cpu_seconds": cpu,
                  "equations": equations, "copies": copies,
                  "cache_hits": len(hits),
                  "cache_enabled": jax.config.jax_enable_compilation_cache}))
"""


def _children_cpu_seconds(before, after):
    """Return the CPU seconds (user plus system) a spawned child consumed."""
    return (after.ru_utime - before.ru_utime) + (after.ru_stime - before.ru_stime)


def compile_row(module_name, name, arm):
    before = resource.getrusage(resource.RUSAGE_CHILDREN)
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
    after = resource.getrusage(resource.RUSAGE_CHILDREN)
    row = json.loads(result.stdout.strip().splitlines()[-1])
    row["process_cpu_seconds"] = _children_cpu_seconds(before, after)
    assert row["cache_hits"] == 0 and row["cache_enabled"] is False
    return row


def _median(values):
    return float(np.median(sorted(values)))


def compile_ratio(module_name, name, repeats=COMPILE_REPEATS):
    """Return the median tangent-over-primal compile CPU ratio, printing every arm.

    The ratio is taken on the CPU seconds each fresh process spent compiling --
    load-insensitive, and the measure the row asserts on.  The wall and the whole
    child's CPU are printed beside it for the record.
    """

    rows = {
        arm: [compile_row(module_name, name, arm) for _ in range(repeats)]
        for arm in ("primal", "tangent", "jvp")
    }
    median = {}
    for arm, runs in rows.items():
        equations = {row["equations"] for row in runs}
        assert len(equations) == 1
        cpu = [row["compile_cpu_seconds"] for row in runs]
        walls = [row["compile_seconds"] for row in runs]
        processes = [row["process_cpu_seconds"] for row in runs]
        median[arm] = _median(cpu)
        print(
            f"COMPILE {name} {arm} median_cpu_seconds={median[arm]:.3f} "
            f"median_wall_seconds={_median(walls):.3f} "
            f"median_process_cpu_seconds={_median(processes):.3f} "
            f"equations={equations.pop()} "
            f"cpu={','.join(f'{value:.3f}' for value in cpu)} "
            f"walls={','.join(f'{value:.3f}' for value in walls)} "
            f"process_cpu={','.join(f'{value:.3f}' for value in processes)} "
            f"hits={sum(row['cache_hits'] for row in runs)}"
        )  # fmt: skip
    ratio = median["tangent"] / median["primal"]
    print(f"COMPILE {name} median_tangent_over_primal_cpu={ratio:.2f}")
    return ratio
