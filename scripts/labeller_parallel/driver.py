#!/usr/bin/env python3
"""Run one resumable scheduler job over the ranked decoder corpus."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import traceback
from typing import Iterator, Sequence

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.labeller_batch.shard import (  # noqa: E402
    DEFAULT_COHORT_REPORT,
    DEFAULT_MANIFEST,
    ShotWork,
    _write_json,
    decoder_corpus,
    prepare_labeller,
)
from scripts.labeller_parallel.scheduler import (  # noqa: E402
    CorpusScheduler,
    HostRouteEngine,
    SequentialCompiledEngine,
    ShotInput,
    SourceIdentity,
    load_shot,
)


def _is_written(output_root: Path, shot: int) -> bool:
    """Return whether the per-shot session and manifest already exist."""
    return (output_root / f"{shot}.nc").is_file() and (
        output_root / f"{shot}.manifest.json"
    ).is_file()


def _default_host_workers() -> int:
    """Return the default host-assembly worker pool size.

    The pool defaults to the allocated cores minus one: SLURM_CPUS_PER_TASK
    minus one when the scheduler sets it, the logical CPU count minus one
    otherwise, floored at one worker.
    """
    if "SLURM_CPUS_PER_TASK" in os.environ:
        return max(1, int(os.environ["SLURM_CPUS_PER_TASK"]) - 1)
    return max(1, int(os.cpu_count() or 1) - 1)


def resolve_host_workers(requested: int | None) -> int:
    """Return the host-assembly worker pool size for a run.

    A positive explicit request wins; otherwise the pool defaults to the
    machine's allocated cores minus one.
    """
    if requested is not None:
        return requested
    return _default_host_workers()


def _load_ranked_shots(
    work: Sequence[ShotWork],
    output_root: Path,
    source_identity: SourceIdentity,
    failures: list[int],
) -> Iterator[ShotInput]:
    """Load one ranked shot at refill time so the corpus is never resident."""
    for item in work:
        try:
            yield load_shot(item.shot, max_slices=None)
        except Exception as error:
            failure = {
                "schema": "nova-forward-labeller-shot",
                "shot": item.shot,
                "status": "failed",
                "nova_revision": source_identity.nova_revision,
                "nova_equilibrium_tree": source_identity.nova_equilibrium_tree,
                "labeller_batch_tree": source_identity.labeller_batch_tree,
                "declared_additions": [
                    "manifest.nova_equilibrium_tree",
                    "manifest.labeller_batch_tree",
                    "manifest.declared_additions",
                    "manifest.slices.requested_class",
                ],
                "failure": f"{type(error).__name__}: {error}",
                "traceback": traceback.format_exc(),
                "slices": [],
            }
            _write_json(failure, output_root / f"{item.shot}.manifest.json")
            failures.append(item.shot)
            print(json.dumps(failure, sort_keys=True), flush=True)


def run_corpus(arguments: argparse.Namespace) -> dict[str, object]:
    """Prepare once, walk the ranked shots lazily, and persist a final receipt."""
    output_root = arguments.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    source_identity = SourceIdentity.capture()
    corpus = decoder_corpus(arguments.manifest, arguments.cohort_report)
    if arguments.max_shots is not None:
        corpus = corpus[: arguments.max_shots]
    pending = [item for item in corpus if not _is_written(output_root, item.shot)]
    previously_written = len(corpus) - len(pending)
    prepared = prepare_labeller()
    engine_type = {
        "host": HostRouteEngine,
        "compiled": SequentialCompiledEngine,
    }[arguments.engine]
    engine = engine_type(
        prepared,
        device_count=arguments.devices,
        condition_on_guard_failure=arguments.condition_on_guard_failure,
    )
    scheduler = CorpusScheduler(
        engine=engine,
        device_count=arguments.devices,
        batch_per_device=arguments.batch_per_device,
        host_workers=arguments.host_workers,
        include_raster=arguments.include_raster,
        condition_on_guard_failure=arguments.condition_on_guard_failure,
    )
    failures: list[int] = []
    receipt = scheduler.run(
        _load_ranked_shots(pending, output_root, source_identity, failures),
        output_root,
        prepared=prepared,
        source_identity=source_identity,
        previously_written_shots=previously_written,
    )
    receipt.update(
        corpus_shot_count=len(corpus),
        failed_shot_count=len(failures),
        failed_shots=failures,
        corpus_source=str(arguments.manifest),
        cohort_source=str(arguments.cohort_report),
        source_identity={
            "nova_revision": source_identity.nova_revision,
            "nova_equilibrium_tree": source_identity.nova_equilibrium_tree,
            "labeller_batch_tree": source_identity.labeller_batch_tree,
        },
        condition_on_guard_failure=arguments.condition_on_guard_failure,
        include_raster=arguments.include_raster,
    )
    _write_json(receipt, output_root / "parallel-receipt.json")
    return receipt


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_root", type=Path)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--cohort-report", type=Path, default=DEFAULT_COHORT_REPORT)
    parser.add_argument("--engine", choices=("host", "compiled"), default="host")
    parser.add_argument("--devices", type=int, default=1)
    parser.add_argument("--batch-per-device", type=int, default=1)
    parser.add_argument("--host-workers", type=int)
    parser.add_argument("--max-shots", type=int)
    parser.add_argument("--include-raster", action="store_true")
    parser.add_argument("--condition-on-guard-failure", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    host_workers = resolve_host_workers(arguments.host_workers)
    arguments.host_workers = host_workers
    if min(arguments.devices, arguments.batch_per_device, host_workers) < 1:
        raise ValueError("device, batch and host worker counts must be positive")
    if arguments.max_shots is not None and arguments.max_shots < 1:
        raise ValueError("--max-shots must be positive")
    print(f"host_workers={host_workers}", flush=True)
    receipt = run_corpus(arguments)
    print(json.dumps(receipt, sort_keys=True), flush=True)
    return 1 if receipt["failed_shot_count"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
