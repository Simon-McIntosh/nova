"""Read one run's receipts in a written order, and read a refused history once.

Two orderings used to come from the filesystem: the merge order of the per-run
``measurement.json`` files, and the order of the generating-failure receipts.
Both are out-of-band facts, so re-copying or re-touching a run directory
silently flipped which measurement of an arm won, and an archive restore lost
the ordering entirely. Every receipt therefore carries a written ``sequence``
that this module orders by, and the newest write of a key wins.

The rejected-history reader lives here once. Both ``report.py`` and
``run_cases.py`` call it, so a refused solve whose last iterate it must read is
described in exactly one place.
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np

PROTOTYPE = Path(__file__).resolve().parent


def profile_coefficients(source) -> np.ndarray:
    """Return the pressure and field coefficients of a solved profile."""
    core = source.core
    pressure = np.asarray(core.p_prime.coefficients, dtype=float)
    field = np.asarray(core.ff_prime.coefficients, dtype=float)
    return np.r_[pressure, field]


def mesh_record(data) -> dict:
    """Return what a receipt records about the mesh a solve was differentiated on."""
    lattice = data["profile"].lattice
    return dict(
        kind=type(lattice).__name__,
        nodes=int(np.asarray(lattice.coordinate).shape[0]),
        stencil_rows=int(np.asarray(lattice.stencil).shape[0]),
        machine_nodes=int(np.asarray(data["machine"].node).shape[0]),
    )


def order_key(path: Path) -> tuple[int, str, str]:
    """Return the written order of one receipt, never its filesystem time.

    ``sequence`` is absent on a receipt written before this module existed; such
    a receipt sorts below every sequenced one and keeps a stable order among its
    peers from its own ``source_revision`` and path.
    """
    payload = json.loads(path.read_text())
    sequence = payload.get("sequence")
    return (
        int(sequence) if isinstance(sequence, int) else 0,
        str(payload.get("source_revision") or ""),
        str(path),
    )


def measurement_receipts(directory: Path) -> list[Path]:
    """Return every measurement receipt that shares ``directory``.

    The run directory holds one subdirectory per job, and each job writes its
    receipt at that subdirectory's top level. ``directory``'s own receipt is
    included as well, because a receipt written directly into the directory it
    is counted against is the shape the job's own output directory has.
    """
    return sorted(
        [*directory.glob("measurement.json"), *directory.glob("*/measurement.json")],
        key=order_key,
    )


def next_sequence(directory: Path) -> int:
    """Return the next written sequence among the receipts sharing ``directory``.

    The caller passes the run directory, which holds every job, not the job's
    own output directory: a job sees only itself through its own directory, so
    a fresh job would write sequence 1 every time and a re-measured arm would
    tie the run it replaces instead of superseding it.
    """
    sequences = [
        int(json.loads(path.read_text()).get("sequence") or 0)
        for path in measurement_receipts(directory)
    ]
    return max(sequences, default=0) + 1


def ordered(run: Path, pattern: str) -> list[Path]:
    """Return the receipts matching ``pattern``, oldest write first."""
    return sorted(run.glob(pattern), key=order_key)


def measurements(run: Path) -> dict:
    """Merge every per-run measurement, oldest write first.

    A later write of one key replaces the earlier arms of that key and every
    case an older run measured still survives, so a re-measured arm supersedes
    the arm it replaced.
    """
    merged: dict = {}
    for path in measurement_receipts(run):
        for key, value in json.loads(path.read_text()).items():
            if isinstance(value, dict):
                merged.setdefault(key, {})
                merged[key].update(value)
            else:
                merged[key] = value
    return merged


def owner(run: Path, label: str) -> Path | None:
    """Return the receipt that holds the newest write of one arm, if any."""
    holder = None
    for path in measurement_receipts(run):
        if label in json.loads(path.read_text()):
            holder = path
    return holder


def failure_receipts(run: Path, label: str) -> list[dict]:
    """Return one arm's generating-failure receipts, oldest write first."""
    return [
        json.loads(path.read_text())
        for path in ordered(run, f"*/comparison/*{label}-generating-failure.json")
    ]


def residual_fields(path) -> dict:
    """Return the residuals a refused solve kept in its rejected history.

    A refusal writes no receipt, so the fixed-point history it rejected is the
    only record of the iterates it did produce. ``finite_residual_entries`` is
    always reported, and is zero when no iterate was finite, so a reader can
    tell "the seed refused before any trip" from "the history was unreadable".
    """
    if not path:
        return {}
    history_path = Path(str(path))
    if not history_path.is_file():
        return {}
    import nova.equilibrium.fixed_point  # noqa: F401 - resolves the pickle type

    try:
        with history_path.open("rb") as stream:
            history = pickle.load(stream)
        values = np.asarray(history.active_set_residuals, dtype=float)
    except OSError, AttributeError, ValueError, pickle.UnpicklingError:
        return {}
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return dict(finite_residual_entries=0)
    return dict(
        last_finite_residual=float(finite[-1]),
        lowest_recorded_residual=float(finite.min()),
        finite_residual_entries=int(finite.size),
    )


def suite_arm_log(path: Path) -> dict:
    """Read a suite arm's log into the facts promotion checks it for.

    The log carries its revision and the exact command on the first two lines
    and its exit status on the last, so the ``exit_status`` a manifest reports
    can be read back from the log it cites rather than taken on trust.
    """
    lines = Path(path).read_text().splitlines()
    if len(lines) < 2:
        raise ValueError(f"{path}: a suite arm log needs a header and a body")
    header, command = lines[0], lines[1]
    if not header.startswith("# revision "):
        raise ValueError(f"{path}: first line is not '# revision <sha>'")
    if not command.startswith("# command: "):
        raise ValueError(f"{path}: second line is not '# command: <command>'")
    tail = lines[-1]
    if not tail.startswith("EXIT="):
        raise ValueError(f"{path}: last line is not 'EXIT=<code>'")
    return dict(
        revision=header.removeprefix("# revision ").strip(),
        command=command.removeprefix("# command: ").strip(),
        exit_status=int(tail.removeprefix("EXIT=")),
        body=lines[2:-1],
    )
