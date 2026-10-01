"""Capture production trip operands without changing the solve state."""
# ruff: noqa: E501 -- Captions and persisted HTML retain their literal text.

import importlib.abc
import importlib.util
from pathlib import Path
import runpy
import sys

ROOT = Path(__file__).resolve().parent
WORKTREE = ROOT.parents[3]


class TripLoader(importlib.abc.Loader):
    def create_module(self, spec):
        return None

    def exec_module(self, module):
        path = WORKTREE / "nova/equilibrium/fixed_point.py"
        source = path.read_text()
        marker = "            next_presettlement = ("
        assert source.count(marker) == 1
        callback = """            jax.debug.callback(
                _capture_trip_state,
                index, carry.state, state, carry.mask, mask, live_residual,
                inner_result.residual, inner_result.accepted_newton_promotions,
                inner_result.attempted_newton_promotions,
                cycle_detected, stagnated, settled, continue_trajectory,
                inner_result.trajectory_state, ordered=True,
            )
"""
        source = source.replace(marker, callback + marker)
        source += """
def _capture_trip_state(*values):
    import os
    destination = Path(os.environ["TRIP_RECORD_DIRECTORY"])
    names = ("index", "incoming", "state", "incoming_mask", "mask",
             "live_residual", "inner_residual", "accepted", "attempted",
             "cycle", "stagnated", "settled", "continue_trajectory", "trajectory")
    record = {name: np.asarray(value) for name, value in zip(names, values, strict=True)}
    trip = int(record["index"]) + 1
    np.savez(destination / f"trip-{trip:02d}.npz", **record)
    print(f"CAPTURED_TRIP {trip} residual={float(record['live_residual']):.10g}", flush=True)
"""
        exec(compile(source, str(path), "exec"), module.__dict__)


class TripFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path, target=None):
        if fullname == "nova.equilibrium.fixed_point":
            return importlib.util.spec_from_loader(fullname, TripLoader())
        return None


sys.meta_path.insert(0, TripFinder())
runpy.run_path(str(ROOT / "measure_trips.py"), run_name="__main__")
