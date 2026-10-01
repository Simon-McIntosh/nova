"""Record the process memory-map count and resident memory at every test boundary.

Loaded with ``-p abort_timeline``. Each row is appended and flushed to the file
named by ``ABORT_TIMELINE`` so rows written before a native abort survive it.
With ``ABORT_TIMELINE_CLEAR=1`` every JAX compilation cache is cleared and the
garbage collector run after each test, which releases executables that only
those caches keep alive.
"""

import gc
import os
import time


def _row(phase, nodeid):
    with open("/proc/self/maps") as maps:
        count = sum(1 for _ in maps)
    rss = hwm = "na"
    with open("/proc/self/status") as status:
        for line in status:
            if line.startswith("VmRSS:"):
                rss = line.split()[1]
            elif line.startswith("VmHWM:"):
                hwm = line.split()[1]
    path = os.environ.get("ABORT_TIMELINE")
    if path:
        with open(path, "a") as out:
            out.write(f"{time.time():.1f} {phase} {count} {rss} {hwm} {nodeid}\n")
            out.flush()
            os.fsync(out.fileno())


def pytest_runtest_logstart(nodeid, location):
    _row("start", nodeid)


def pytest_runtest_logfinish(nodeid, location):
    if os.environ.get("ABORT_TIMELINE_CLEAR") == "1":
        import jax

        jax.clear_caches()
        gc.collect()
    _row("finish", nodeid)
