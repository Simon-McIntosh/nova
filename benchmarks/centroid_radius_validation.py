"""Validate Nova's solved current-centroid major radius against EFIT.

The forward labeller labels a slice with the plasma current centroid that its
own moment observation reads off the converged flux.  That major-radius
component was never compared with anything -- the height has a branch guard and
a conditioning target, the major radius has none -- so a several-centimetre
systematic inboard offset survived a corpus of 923 shots.  This benchmark turns
the persisted labeller solves into a measured distribution and states a guard
tolerance argued from the grid spacing, never from the present miss.

Measurement basis: the persisted six-carrier labeller corpus produced by the
reserved-card batches (one manifest per shot, each slice recording the solved
``achieved_current_centroid_r``).  EFIT's published current centroid radius is
read from the ``efm`` store as ``current_centrd_r`` on the same slice index.
Every slice with a finite solved centroid contributes one signed delta
``nova_r - efit_r``.

The tolerance argument: the solved centroid is a cell-current-weighted mean of
the lattice cell centres.  Representing each cell's current at its centre
rather than at its true in-cell distribution displaces the moment integral by
at most half a cell, so the discretisation floor on the centroid major radius
is ``dr / 2`` where ``dr`` is the radial step of the lattice the solve ran on
(the stride-2 decimation the labeller uses).  A displacement beyond that floor
on a maintained sign is a resolved systematic, not discretisation scatter.  A
guard set to pass today's miss would ratify the defect; ``dr / 2`` is argued
from the grid alone.

Receipts: median, mean, standard deviation, per-shot medians, inboard fraction
and per-row pass/fail on the twelve bank rows and on the carrier sample,
persisted to ``docs/figures/playable-forward-solve/centroid-radius/``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import platform
import subprocess
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import zarr

from nova.imas.mast_solve_inputs import SHOT_STORE

ROOT = Path(__file__).resolve().parents[1]
#: The deterministic dozen of the labeller's driven 22086 rows that the
#: topology and forward-solve suites use as their referee slice set.
MAST_BANK_ROWS = (1, 6, 12, 18, 24, 30, 36, 43, 45, 50, 54, 57)
#: The frozen reference cohort whose response carrier the labeller shares.
CARRIER_SHOTS = (21978, 21983, 21985, 21986, 21989, 22086)
#: The stride applied to the stored efm axis by the labeller's operator build.
GRID_STRIDE = 2
#: Persisted solved sessions: one manifest per shot on the reserved-card runs.
DEFAULT_CORPUS_ROOT = Path(
    "/work/projects/imas_gpu/sophelio/labeller_sessions/76906a29"
)
DEFAULT_OUTPUT = ROOT / "docs/figures/playable-forward-solve/centroid-radius"
RECEIPT_NAME = "centroid-radius-validation.json"
FIGURE_NAME = "centroid-radius-distribution.png"


def _source_revision() -> str:
    """Return the revision this measurement runs from."""
    return subprocess.run(
        ["git", "-C", str(ROOT), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _strict_float(value: Any) -> float | None:
    """Return one finite host float, or None where the value is absent."""
    if value is None:
        return None
    result = float(np.asarray(value))
    return result if np.isfinite(result) else None


def grid_radial_step(shot: int) -> float:
    """Return the radial step of the labeller's solved lattice [m]."""
    group = zarr.open_group(str(SHOT_STORE / f"{shot}.zarr"), mode="r")["efm"]
    full_r = np.asarray(group["gridr"], dtype=np.float64)
    return float(np.diff(full_r[::GRID_STRIDE]).mean())


def discretisation_tolerance_m(radial_step: float) -> float:
    """Return the discretisation floor on the centroid major radius [m].

    The solved centroid is a cell-current-weighted mean of lattice cell
    centres; representing each cell's current at its centre rather than at its
    true in-cell distribution displaces the moment by at most half a cell.  The
    floor is therefore one half of the radial cell step.
    """
    return radial_step / 2.0


def load_measurements(corpus_root: Path, shot: int) -> dict[str, Any]:
    """Return aligned solved-centroid and EFIT radii for one shot.

    ``rows`` names the slices with a finite solved centroid, ``nova_r`` their
    solved radii and ``efit_r`` the published ``efm/current_centrd_r`` on the
    same slice index.  Provenance carries the corpus's recorded revision,
    policy digest and carrier identity so the measurement stays attributable
    across code movement.
    """
    manifest_path = corpus_root / f"{shot}.manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"no solved session manifest for shot {shot}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    group = zarr.open_group(str(SHOT_STORE / f"{shot}.zarr"), mode="r")["efm"]
    efit_r = np.asarray(group["current_centrd_r"], dtype=np.float64)
    rows: list[int] = []
    nova_r: list[float] = []
    converged: list[bool] = []
    for entry in manifest["slices"]:
        achieved = _strict_float(entry.get("achieved_current_centroid_r"))
        if achieved is None:
            continue
        rows.append(int(entry["row"]))
        nova_r.append(achieved)
        converged.append(bool(entry.get("converged", False)))
    return {
        "rows": np.asarray(rows, dtype=np.int64),
        "nova_r": np.asarray(nova_r, dtype=np.float64),
        "efit_r": efit_r[np.asarray(rows, dtype=np.int64)],
        "converged": np.asarray(converged, dtype=bool),
        "provenance": {
            "nova_revision": manifest.get("nova_revision"),
            "policy_digest": manifest.get("policy_digest"),
            "carrier_identity": manifest.get("carrier_identity"),
            "status": manifest.get("status"),
            "radial_step_m": grid_radial_step(shot),
        },
    }


def _distribution(deltas_cm: np.ndarray) -> dict[str, float]:
    """Return median, mean, standard deviation and inboard fraction."""
    return {
        "count": int(deltas_cm.size),
        "median_cm": float(np.median(deltas_cm)),
        "mean_cm": float(deltas_cm.mean()),
        "std_cm": float(deltas_cm.std()),
        "inboard_fraction": float((deltas_cm < 0.0).mean()),
    }


def _bank_rows(
    measurements: dict[int, dict[str, Any]], floor_m: float
) -> list[dict[str, Any]]:
    """Return per-bank-row records for shot 22086, measured and missing."""
    bank = {int(row): i for i, row in enumerate(measurements[22086]["rows"])}
    records: list[dict[str, Any]] = []
    for row in MAST_BANK_ROWS:
        index = bank.get(int(row))
        if index is None:
            records.append(
                {"row": int(row), "measured": False, "reason": "no solved centroid"}
            )
            continue
        entry = measurements[22086]
        delta_cm = float(entry["nova_r"][index] - entry["efit_r"][index]) * 100.0
        records.append(
            {
                "row": int(row),
                "measured": True,
                "nova_centroid_r_m": float(entry["nova_r"][index]),
                "efit_centroid_r_m": float(entry["efit_r"][index]),
                "delta_cm": delta_cm,
                "pass": bool(abs(delta_cm) <= floor_m * 100.0),
            }
        )
    return records


def measure(
    corpus_root: Path,
    output: Path,
    *,
    tolerance_m: float | None = None,
) -> dict[str, Any]:
    """Validate solved centroid radii against EFIT and persist the receipt."""
    measurements: dict[int, dict[str, Any]] = {}
    carrier_deltas: list[np.ndarray] = []
    for shot in CARRIER_SHOTS:
        entry = load_measurements(corpus_root, shot)
        entry["deltas_cm"] = (entry["nova_r"] - entry["efit_r"]) * 100.0
        entry["per_shot"] = _distribution(entry["deltas_cm"])
        measurements[shot] = entry
        carrier_deltas.append(entry["deltas_cm"])
    all_cm = np.concatenate(carrier_deltas)

    radial_step = measurements[22086]["provenance"]["radial_step_m"]
    floor = (
        tolerance_m
        if tolerance_m is not None
        else discretisation_tolerance_m(radial_step)
    )
    bank_rows = _bank_rows(measurements, floor)
    measured_bank = np.asarray(
        [row["delta_cm"] for row in bank_rows if row["measured"]],
        dtype=np.float64,
    )
    receipt = {
        "receipt": "current-centroid major radius validation against EFIT",
        "source": {
            "revision": _source_revision(),
            "python": platform.python_version(),
        },
        "corpus": {
            "root": str(corpus_root),
            "shots": list(CARRIER_SHOTS),
            "bank_rows": list(MAST_BANK_ROWS),
        },
        "tolerance": {
            "floor_m": floor,
            "radial_step_m": radial_step,
            "argument": (
                "the solved centroid is a cell-current-weighted mean of lattice "
                "cell centres; representing each cell's current at its centre "
                "displaces the moment by at most half a cell, so the discretisation "
                "floor is dr/2 of the labeller's lattice, never fitted to the "
                "observed miss"
            ),
        },
        "bank_rows": bank_rows,
        "bank_distribution": (
            _distribution(measured_bank) if measured_bank.size else {}
        ),
        "carrier_distribution": _distribution(all_cm),
        "carrier_per_shot": {
            str(shot): measurements[shot]["per_shot"] for shot in CARRIER_SHOTS
        },
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / RECEIPT_NAME).write_text(
        json.dumps(receipt, indent=2, allow_nan=False) + "\n"
    )
    _figure(measurements, bank_rows, floor, output / FIGURE_NAME)
    return receipt


def _figure(
    measurements: dict[int, dict[str, Any]],
    bank_rows: list[dict[str, Any]],
    floor_m: float,
    path: Path,
) -> None:
    """Draw per-shot delta spreads, the bank rows, and the tolerance floor."""
    figure, axes = plt.subplots(
        1, 2, figsize=(11, 4.4), gridspec_kw={"width_ratios": [1.3, 2.0]}
    )
    floor_cm = floor_m * 100.0
    for shot in CARRIER_SHOTS:
        entry = measurements[shot]
        sample = entry["deltas_cm"]
        rng = np.random.default_rng(int(shot))
        axes[0].scatter(
            np.full(sample.size, shot) + rng.uniform(-0.18, 0.18, sample.size),
            sample,
            s=14,
            alpha=0.55,
            edgecolor="none",
        )
        axes[0].plot(
            [shot],
            [entry["per_shot"]["median_cm"]],
            marker="D",
            color="k",
            markersize=5,
        )
    axes[0].axhline(0.0, color="0.4", linewidth=0.8)
    axes[0].axhspan(-floor_cm, floor_cm, color="0.85", alpha=0.6)
    axes[0].axhline(-floor_cm, color="0.3", linewidth=0.7, linestyle="--")
    axes[0].set_xticks(list(CARRIER_SHOTS))
    axes[0].set_xlabel("Carrier shot")
    axes[0].set_ylabel("Solved centroid R $-$ EFIT R [cm]")
    axes[0].set_title(
        f"Carrier sample (inboard negative, floor $\\pm${floor_cm:.2f} cm)"
    )
    axes[0].grid(axis="y", alpha=0.2)

    measured = [(row["row"], row["delta_cm"]) for row in bank_rows if row["measured"]]
    missing = [row["row"] for row in bank_rows if not row["measured"]]
    if measured:
        rows = [row for row, _ in measured]
        deltas = [delta for _, delta in measured]
        axes[1].bar(
            np.arange(len(measured)),
            deltas,
            color=["#d62728" if delta < -floor_cm else "#1f77b4" for delta in deltas],
            width=0.6,
        )
        axes[1].axhline(0.0, color="0.4", linewidth=0.8)
        axes[1].axhspan(-floor_cm, floor_cm, color="0.85", alpha=0.6)
        axes[1].set_xticks(np.arange(len(measured)), [str(row) for row in rows])
        axes[1].set_xlabel("22086 bank row")
        axes[1].set_ylabel("Delta [cm]")
        axes[1].set_title("Twelve bank rows against the discretisation floor")
        axes[1].grid(axis="y", alpha=0.2)
        for position, (row, delta) in enumerate(zip(rows, deltas)):
            label = f"{delta:+.1f}"
            axes[1].text(
                position,
                delta + (0.25 if delta >= 0 else -0.6),
                label,
                ha="center",
                fontsize=7,
            )
    axes[1].text(
        0.97,
        0.03,
        (
            f"missing: {', '.join(str(row) for row in missing)}"
            if missing
            else "all measured"
        ),
        transform=axes[1].transAxes,
        ha="right",
        va="bottom",
        fontsize=8,
        color="0.35",
    )
    figure.suptitle("Current-centroid major radius: solved vs EFIT", y=0.98)
    figure.tight_layout(rect=(0, 0, 1, 0.95))
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180)
    plt.close(figure)


def main(argv: list[str] | None = None) -> Path:
    """Run the validation and return the receipt path."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS_ROOT)
    parser.add_argument(
        "--tolerance-m",
        type=float,
        default=None,
        help="explicit discretisation-floor tolerance [m] (default dr/2)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="receipt and figure output directory",
    )
    arguments = parser.parse_args(argv)
    measure(arguments.corpus, arguments.output, tolerance_m=arguments.tolerance_m)
    return arguments.output / RECEIPT_NAME


if __name__ == "__main__":
    main()
