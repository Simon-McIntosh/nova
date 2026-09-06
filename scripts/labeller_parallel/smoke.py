#!/usr/bin/env python3
"""Run the corpus scheduler smoke on sixteen ranked decoder shots."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys
from typing import Any, Sequence

import numpy as np
import xarray as xr

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.labeller_batch.shard import (  # noqa: E402
    DEFAULT_COHORT_REPORT,
    DEFAULT_MANIFEST,
    decoder_corpus,
)
from scripts.labeller_parallel.scheduler import (  # noqa: E402
    ArrayBatchEngine,
    AssemblyRequest,
    CorpusScheduler,
    SliceInput,
    assemble_stub_frame,
    write_shot,
)


SHOT_COUNT = 16
SLICES_PER_SHOT = 4


def _synthetic_slices(shot: int) -> list[SliceInput]:
    """Build deterministic inputs keyed only by a real corpus shot identity."""
    random = np.random.default_rng(shot)
    state = random.normal(scale=1e-3, size=1_126)
    current = random.normal(scale=2e3, size=101)
    result = []
    for row in range(SLICES_PER_SHOT):
        result.append(
            SliceInput(
                shot=shot,
                row=row,
                time=row / 200.0,
                initial_state=state + row * 1e-8,
                prescribed_current=current + row,
                target_current=500_000.0 + shot,
                requested_class=1 + shot % 2,
                centroid_target_z=(shot % 17 - 8) * 1e-3,
            )
        )
    return result


def _sequential_reference(
    ranked: Sequence[tuple[int, Sequence[SliceInput]]], output_root: Path
) -> None:
    """Write the same shots through an intentionally sequential engine loop."""
    engine = ArrayBatchEngine(device_count=1)
    for shot, slices in ranked:
        warm_state = None
        assembled = []
        for item in slices:
            if warm_state is not None:
                item = SliceInput(
                    shot=item.shot,
                    row=item.row,
                    time=item.time,
                    initial_state=warm_state,
                    prescribed_current=item.prescribed_current,
                    target_current=item.target_current,
                    requested_class=item.requested_class,
                    centroid_target_z=item.centroid_target_z,
                )
            scheduler = CorpusScheduler(
                engine=engine,
                device_count=1,
                batch_per_device=1,
                host_workers=1,
            )
            batch = scheduler._pack(
                [
                    type(
                        "ReferenceSlot",
                        (),
                        {"current": lambda self, value=item: value},
                    )()
                ]
            )
            result = engine.step(batch)
            warm_state = np.asarray(result.state[0])
            assembled.append(
                assemble_stub_frame(
                    AssemblyRequest(
                        shot=shot,
                        row=item.row,
                        time=item.time,
                        state=warm_state,
                        converged=bool(result.converged[0]),
                        termination=int(result.termination[0]),
                        trips=int(result.trips[0]),
                        terminal_residual=float(result.terminal_residual[0]),
                        centroid=np.asarray(result.centroid[0]),
                        conditioned=bool(result.conditioned[0]),
                        labelled_fields={
                            name: np.asarray(value[0])
                            for name, value in result.labelled_fields.items()
                        },
                    )
                )
            )
        write_shot(
            output_root,
            shot,
            assembled,
            run_metadata={"engine": "array-contract-stub"},
        )


def _manifest_rows(path: Path) -> list[dict[str, Any]]:
    return json.loads(path.read_text(encoding="utf-8"))["slices"]


def _assert_identical(reference: Path, candidate: Path, shots: Sequence[int]) -> int:
    compared = 0
    for shot in shots:
        reference_rows = _manifest_rows(reference / f"{shot}.manifest.json")
        candidate_rows = _manifest_rows(candidate / f"{shot}.manifest.json")
        if reference_rows != candidate_rows:
            raise AssertionError(f"per-slice manifest records differ for shot {shot}")
        with xr.open_dataset(
            reference / f"{shot}.nc", group="steering_frames"
        ) as expected:
            with xr.open_dataset(
                candidate / f"{shot}.nc", group="steering_frames"
            ) as actual:
                xr.testing.assert_identical(expected, actual)
        compared += len(reference_rows)
    return compared


def run_smoke(output: Path) -> dict[str, Any]:
    """Execute one- and three-device scheduler arms and persist their receipt."""
    corpus = decoder_corpus(DEFAULT_MANIFEST, DEFAULT_COHORT_REPORT)
    ranked = [(item.shot, _synthetic_slices(item.shot)) for item in corpus[:SHOT_COUNT]]
    shots = [shot for shot, _slices in ranked]
    if output.exists():
        shutil.rmtree(output)
    reference = output / "sequential"
    _sequential_reference(ranked, reference)
    arms = []
    for devices in (1, 3):
        arm_root = output / f"devices-{devices}"
        scheduler = CorpusScheduler(
            engine=ArrayBatchEngine(device_count=devices),
            device_count=devices,
            batch_per_device=2,
            host_workers=max(1, 4 * devices - 1),
        )
        receipt = scheduler.run(
            ranked,
            arm_root,
            run_metadata={"engine": "array-contract-stub"},
        )
        receipt["identical_slice_records"] = _assert_identical(
            reference, arm_root, shots
        )
        arms.append(receipt)
    result = {
        "schema": "nova-forward-labeller-parallel-smoke",
        "corpus_source": str(DEFAULT_MANIFEST),
        "cohort_source": str(DEFAULT_COHORT_REPORT),
        "shot_count": SHOT_COUNT,
        "slices_per_shot": SLICES_PER_SHOT,
        "shot_ids": shots,
        "engine": "array-contract-stub",
        "arms": arms,
        "passed": all(
            arm["identical_slice_records"] == SHOT_COUNT * SLICES_PER_SHOT
            for arm in arms
        ),
    }
    receipt_path = output / "parallel-smoke-receipt.json"
    receipt_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    result = run_smoke(_parser().parse_args(argv).output.resolve())
    print(json.dumps(result, sort_keys=True))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
