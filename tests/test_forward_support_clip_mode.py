"""The support clip mode defaults to the committed signed-flux clip.

The committed default clips every atomic cell against the signed flux of
the current iterate; exact participation and the chord-cells hybrid are
opt-in through ``set_support_clip_mode``.  These tests pin the import
default, record the receipt the reinstated clip produces against the
whole-cell booking of the main checkout on the weak-rotation-reactor-static
certificate state, and prove the exact opt-in path changes cell currents.
"""

from __future__ import annotations

import json
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


def test_default_chord_clip_changes_moments_with_a_recorded_receipt(tmp_path):
    """The reinstated clip is a measured before/after change, not an identity.

    The same deterministic driver builds the weak-rotation-reactor-static
    -110 certificate machine and state once against this worktree (the
    signed-flux clip default) and once against the main checkout (the prior
    whole-cell booking by partition label), with PYTHONPATH swapped in a
    subprocess.  The default is meant to change here, so the test records
    both moment arrays and their unit-amplitude totals as a receipt and
    asserts the clip booked the current the whole-cell line dropped.
    """
    configure_dtypes()
    driver = _write_driver(tmp_path)
    worktree_out = tmp_path / "worktree.npz"
    main_out = tmp_path / "main.npz"
    _run_driver(WORKTREE, driver, worktree_out, mode=None)
    _run_driver(MAIN_CHECKOUT, driver, main_out, mode=None)
    worktree_raw, worktree_unit = _load_arrays(worktree_out)
    main_raw, main_unit = _load_arrays(main_out)

    # The clip is live: the committed default differs cell by cell and in
    # total from the whole-cell booking of the same state.
    assert not np.array_equal(worktree_raw, main_raw)
    assert np.sum(worktree_raw) != np.sum(main_raw)
    before_total = float(np.sum(main_raw))
    after_total = float(np.sum(worktree_raw))

    # On this exact certificate state the clipped support keeps the cut-cell
    # current the whole-cell line dropped, so the total moves toward the
    # analytic current rather than staying suppressed.
    assert after_total > before_total
    assert float(np.sum(worktree_unit)) == pytest.approx(1.0)

    receipt = {
        "default_before_total_a": before_total,
        "default_before_total_from_unit_sum": float(np.sum(main_unit)),
        "default_after_total_a": after_total,
        "default_after_total_from_unit_sum": float(np.sum(worktree_unit)),
        "per_cell_before_count_changed": int(np.sum(worktree_raw != main_raw)),
        "before_array": main_raw.tolist(),
        "after_array": worktree_raw.tolist(),
    }
    receipt_path = tmp_path / "clip-receipt.json"
    receipt_path.write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    assert receipt_path.is_file()


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
