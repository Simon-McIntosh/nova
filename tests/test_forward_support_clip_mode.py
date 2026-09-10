"""The support clip mode defaults to the committed chord clip.

The chord clip is the committed production behaviour of the forward
operator; exact participation and the chord-cells hybrid are opt-in
through ``set_support_clip_mode``.  These tests pin the import default,
prove it reproduces the main checkout's forward operator bit-for-bit on
the weak-rotation-reactor-static certificate state, and prove the exact
opt-in path changes cell currents.
"""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from nova.jax.config import configure_dtypes

#: The repository whose committed forward operator is the identity reference.
MAIN_CHECKOUT = Path("/home/ITER/mcintos/Code/nova")
WORKTREE = Path(__file__).resolve().parents[1]

DRIVER = """\
\"\"\"Compute unit-amplitude zeroth current moments on a certificate state.\"\"\"
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def build():
    \"\"\"Return the operator and the exact certificate state for the case.\"\"\"
    import jax.numpy as jnp

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
        (machine.node, machine.wall_node, machine.sample_coordinates)
    )
    state = oracle_fixture.exact_state(case, coordinates)
    return operator, np.asarray(state, dtype=np.float64)


def zeroth_moments(operator, state):
    \"\"\"Return raw and unit-sum per-cell zeroth current moments.\"\"\"
    import jax.numpy as jnp

    moments = operator.cell_current_moments(jnp.asarray(state))
    raw = np.asarray(moments.cell_current, dtype=np.float64)
    return raw, raw / np.sum(raw)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "--mode",
        choices=("chord", "exact", "chord_cells"),
        default=None,
        help="explicit clip mode; the module default when omitted",
    )
    args = parser.parse_args()
    operator, state = build()
    if args.mode is not None:
        from nova.equilibrium.forward_operator import set_support_clip_mode

        set_support_clip_mode(args.mode)
    raw, unit = zeroth_moments(operator, state)
    np.savez(args.output, raw=raw, unit=unit)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
"""


def _write_driver(tmp_path: Path) -> Path:
    driver = tmp_path / "zeroth_moments_driver.py"
    driver.write_text(DRIVER, encoding="utf-8")
    return driver


def _run_driver(checkout: Path, driver: Path, output: Path, mode: str | None) -> None:
    """Run the moment driver against one checkout and wait for its receipt."""
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(checkout)
    command = [sys.executable, str(driver), str(output)]
    if mode is not None:
        command.extend(("--mode", mode))
    result = subprocess.run(
        command,
        cwd=str(WORKTREE),
        env=environment,
        capture_output=True,
        text=True,
        timeout=900,
    )
    if result.returncode != 0 or not output.is_file():
        raise AssertionError(
            f"driver failed against {checkout}:\nstdout:\n{result.stdout}"
            f"\nstderr:\n{result.stderr}"
        )


def _load_arrays(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with np.load(path) as receipt:
        return np.asarray(receipt["raw"]), np.asarray(receipt["unit"])


def test_support_clip_mode_defaults_to_chord_on_import():
    """A fresh interpreter reports the committed chord default."""
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(WORKTREE)
    probe = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "from nova.equilibrium.forward_operator import support_clip_mode;"
                "assert support_clip_mode() == 'chord'"
            ),
        ],
        cwd=str(WORKTREE),
        env=environment,
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert probe.returncode == 0, probe.stderr


def test_default_chord_moments_match_main_checkout_bit_for_bit(tmp_path):
    """The default mode reproduces the main checkout's moments to the bit.

    The same deterministic driver builds the weak-rotation-reactor-static
    -110 certificate machine and state once against this worktree and once
    against the main checkout (PYTHONPATH swapped in a subprocess).  The
    per-cell zeroth current moments at unit amplitude under the default,
    and their sum, must be bit-identical: the held stack must merge without
    changing any production result.
    """
    configure_dtypes()
    driver = _write_driver(tmp_path)
    worktree_out = tmp_path / "worktree.npz"
    main_out = tmp_path / "main.npz"
    _run_driver(WORKTREE, driver, worktree_out, mode=None)
    _run_driver(MAIN_CHECKOUT, driver, main_out, mode=None)
    worktree_raw, worktree_unit = _load_arrays(worktree_out)
    main_raw, main_unit = _load_arrays(main_out)

    np.testing.assert_array_equal(worktree_unit, main_unit)
    assert np.sum(worktree_unit) == np.sum(main_unit)
    assert float(np.sum(worktree_unit)) == pytest.approx(1.0)
    np.testing.assert_array_equal(worktree_raw, main_raw)
    assert np.sum(worktree_raw) == np.sum(main_raw)


def test_exact_mode_changes_cell_currents(tmp_path):
    """The exact opt-in path is live: it moves at least one cell's current."""
    from nova.equilibrium.forward_operator import (
        set_support_clip_mode,
        support_clip_mode,
    )

    configure_dtypes()
    previous = support_clip_mode()
    try:
        assert set_support_clip_mode("exact") == "exact"
        assert support_clip_mode() == "exact"

        driver = _write_driver(tmp_path)
        chord_out = tmp_path / "chord.npz"
        exact_out = tmp_path / "exact.npz"
        _run_driver(WORKTREE, driver, chord_out, mode="chord")
        _run_driver(WORKTREE, driver, exact_out, mode="exact")
        chord_raw, _chord_unit = _load_arrays(chord_out)
        exact_raw, _exact_unit = _load_arrays(exact_out)

        assert not np.array_equal(chord_raw, exact_raw)
        assert np.any(chord_raw != exact_raw)
    finally:
        set_support_clip_mode(previous)
