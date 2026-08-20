"""Audit persistent reuse of the expensive forward-equilibrium build products."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


FIXTURE_RE = re.compile(r"BUILD fixture=(coarse|fine)\b")
COMPLETION_RE = re.compile(r"(?:plasmagrid|plasmawall): 100%.*?\[(\d+):(\d+)<")


def _fixture_build_times(path: Path) -> dict[str, int]:
    """Return summed operator-build progress times from one completed run log."""
    text = path.read_text(encoding="utf-8", errors="replace")
    starts = list(FIXTURE_RE.finditer(text))
    elapsed: dict[str, int] = {}
    for position, start in enumerate(starts):
        end = starts[position + 1].start() if position + 1 < len(starts) else len(text)
        section = text[start.start() : end]
        samples = [
            int(minutes) * 60 + int(seconds)
            for minutes, seconds in COMPLETION_RE.findall(section)
        ]
        unique: list[int] = []
        for sample in samples:
            if not unique or sample != unique[-1]:
                unique.append(sample)
        if len(unique) == 3:
            elapsed[start.group(1)] = sum(unique)
    return elapsed


def _require_source_contract(source_root: Path) -> dict[str, bool]:
    """Assert the cache and bypass mechanisms this audit reports are present."""
    machine = (source_root / "nova/imas/machine.py").read_text(encoding="utf-8")
    fixture = (source_root / "tests/test_equilibrium_forward_reference.py").read_text(
        encoding="utf-8"
    )
    consumers = {
        name: (source_root / relative).read_text(encoding="utf-8")
        for name, relative in {
            "root_attribution": (
                "scripts/root_gate_attribution/measure_root_attribution.py"
            ),
            "observation_rescore": "scripts/observation_clip_rescore/measure.py",
            "seed_adjudication": "scripts/analytic_seed_adjudication/run.py",
        }.items()
    }
    checks = {
        "machine_group_uses_semantic_hash": "self.hash_attrs(self.group_attrs)"
        in machine,
        "machine_store_rejects_identity_drift": (
            "machine cache identity changed after its group was selected" in machine
        ),
        "fixture_has_process_local_machine_memo": (
            "@lru_cache(maxsize=4)\ndef _machine" in fixture
        ),
        "fixture_builder_is_direct": "def build_machine(" in fixture,
        **{
            f"{name}_calls_direct_builder": "reference.build_machine(" in text
            for name, text in consumers.items()
        },
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise AssertionError(f"source contract changed: {', '.join(failed)}")
    return checks


def _summary(source_root: Path, logs: list[Path]) -> dict[str, object]:
    """Build the machine-readable audit summary."""
    checks = _require_source_contract(source_root)
    samples: dict[str, list[dict[str, object]]] = {"coarse": [], "fine": []}
    for path in logs:
        for fixture, seconds in _fixture_build_times(path).items():
            samples[fixture].append({"seconds": seconds, "log": str(path)})
    timings = {}
    for fixture, records in samples.items():
        ordered = sorted(int(record["seconds"]) for record in records)
        if not ordered:
            timings[fixture] = {"cold_samples": [], "warm_samples": []}
            continue
        timings[fixture] = {
            "cold_samples": records,
            "cold_min_seconds": ordered[0],
            "cold_median_seconds": ordered[len(ordered) // 2],
            "cold_max_seconds": ordered[-1],
            "warm_samples": [],
            "warm_measurement_blocker": (
                "the standalone consumers call build_machine directly and no "
                "persistent fixture entry exists to load"
            ),
        }
    return {
        "source_contract": checks,
        "stores": {
            "production_machine": {
                "status": "cached_and_reused",
                "store": (
                    "${user_data_dir}/nova/${nova_version}/${machine_filename}.zarr"
                ),
                "key": "xxh64(canonical_key(Machine.group_attrs))",
            },
            "production_coupling": {
                "status": "cached_and_reused_with_machine",
                "store": "<machine-store>/<machine-key>/<method-name>",
                "key": "machine semantic identity plus method group name",
            },
            "production_plasma_grid": {
                "status": "cached_and_reused_with_machine",
                "store": "<machine-store>/<machine-key>/plasmagrid and plasmawall",
                "key": "machine semantic identity plus method group name",
            },
            "reference_coarse": {
                "status": "rebuilt_per_standalone_run",
                "store": None,
                "key": None,
            },
            "reference_fine": {
                "status": "rebuilt_per_standalone_run",
                "store": None,
                "key": None,
            },
        },
        "fixture_miss_mechanism": "bypassed existing persistent cache architecture",
        "timings": timings,
    }


def main() -> None:
    """Write a reproducible JSON audit from source and banked timing logs."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, default=Path.cwd())
    parser.add_argument("--timing-log", type=Path, action="append", default=[])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = _summary(args.source_root.resolve(), args.timing_log)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
