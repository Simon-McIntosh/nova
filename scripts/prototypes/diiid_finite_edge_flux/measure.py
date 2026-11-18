"""Solve one DIII-D row on the tree the launcher names.

The solve, branch seeding and analysis are the established cohort's; this driver
adds one record: the prescribed profile's boundary value, read from the profile
the solve was handed, so the receipt carries the value at psi_norm = 1 the solve
actually used rather than the table it was meant to use.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
HELD = ROOT / "scripts/prototypes/diiid_xpoint_held_nodes"
sys.path.insert(0, str(HELD))

SPEC = importlib.util.spec_from_file_location("held_measure", HELD / "measure.py")
held_measure = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = held_measure
SPEC.loader.exec_module(held_measure)

from benchmarks import diiid_forward_gs_match as bench  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tree", type=Path, required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--frame", type=int, required=True)
    parser.add_argument("--branch", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--trace-helper", type=Path, required=True)
    args = parser.parse_args()
    assert Path(bench.__file__).resolve().is_relative_to(ROOT), bench.__file__
    args.output.mkdir(parents=True, exist_ok=True)
    import jax.numpy as jnp
    from nova.equilibrium.forward import ForwardProfile

    solve = ForwardProfile.solve
    captured = {}

    def recorded(profile, request=None, *positional, **keywords):
        captured.setdefault("profile", profile)
        return solve(profile, request, *positional, **keywords)

    ForwardProfile.solve = recorded
    held_measure.core.install_analysis = held_measure.install_analysis
    sys.argv = [
        sys.argv[0],
        "--tree",
        str(args.tree),
        "--revision",
        args.revision,
        "--frame",
        str(args.frame),
        "--branch",
        args.branch,
        "--output",
        str(args.output),
        "--trace-helper",
        str(args.trace_helper),
    ]
    held_measure.core.main()
    profile = captured.get("profile")
    if profile is None:
        raise RuntimeError("the solve never handed its profile to the recorder")
    core = profile.operator.source.core
    record = {
        "tree": str(args.tree),
        "revision": args.revision,
        "has_finite_edge": bool(getattr(core.p_prime, "has_finite_edge", False)),
        "p_prime_at_boundary": float(core.p_prime(jnp.asarray(1.0))),
        "ff_prime_at_boundary": float(core.ff_prime(jnp.asarray(1.0))),
        "coordinate_first": float(core.p_prime.coordinate[0]),
        "coordinate_last": float(core.p_prime.coordinate[-1]),
        "coordinate_count": int(core.p_prime.coordinate.size),
    }
    (args.output / "built-profile.json").write_text(json.dumps(record, indent=2) + "\n")
    print("BUILT_PROFILE " + json.dumps(record), flush=True)


if __name__ == "__main__":
    main()
