"""Print resident memory around physical fixture stages without changing arithmetic."""

from functools import wraps
import json
from pathlib import Path
import resource
import threading

from nova.jax.config import configure_dtypes

configure_dtypes()

import jax  # noqa: E402
import nova  # noqa: E402

_stage = "pytest"
_stop = threading.Event()


def emit(label):
    status = Path("/proc/self/status").read_text().splitlines()
    rss = next(int(line.split()[1]) for line in status if line.startswith("VmRSS:"))
    print(
        "MEMORY",
        json.dumps(
            dict(
                stage=label,
                resident_mib=rss / 1024,
                peak_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
            )
        ),
        flush=True,
    )


def wrap(owner, name, label=None):
    original = getattr(owner, name)

    @wraps(original)
    def measured(*args, **kwargs):
        global _stage
        previous, _stage = _stage, label or name
        emit(_stage + ":enter")
        result = original(*args, **kwargs)
        jax.block_until_ready(result)
        emit(_stage + ":exit")
        _stage = previous
        return result

    setattr(owner, name, measured)


def pytest_sessionstart(session):
    from nova.equilibrium import current, topology
    from scripts.prototypes.read_route import jets, measure
    from tests.equilibrium import test_topology_read as fixture

    print(
        "PROVENANCE", nova.__file__, topology.__file__, Path.cwd().resolve(), flush=True
    )
    assert jax.config.jax_enable_x64 and jax.default_backend() == "gpu"
    for name in ("_analytic_inputs", "_realised_hex_geometry", "_kernel_backed_field"):
        wrap(fixture, name)
    wrap(measure, "sample_field")
    wrap(jets, "build_jet_matrix")
    wrap(current, "book")
    original = jax.jit

    def observed_jit(function, *args, **kwargs):
        program = original(function, *args, **kwargs)
        if function is not topology.read:
            return program

        def measured(*values, **options):
            global _stage
            previous, _stage = _stage, "topology-read:" + type(values[0]).__name__
            emit(_stage + ":enter")
            result = jax.block_until_ready(program(*values, **options))
            emit(_stage + ":exit")
            _stage = previous
            return result

        return measured

    jax.jit = observed_jit

    def monitor():
        while not _stop.wait(10):
            emit(_stage + ":sample")

    threading.Thread(target=monitor, daemon=True).start()
    emit("session:start")


def pytest_sessionfinish(session, exitstatus):
    _stop.set()
    emit("session:finish")
