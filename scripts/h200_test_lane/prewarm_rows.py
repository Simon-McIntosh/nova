"""Run the certificate fixture floor and refuse XLA autotune cache files."""

from __future__ import annotations

import argparse
from pathlib import Path


def autotune_file_count(directory: Path) -> int:
    """Count persisted per-fusion autotune files in a served cache directory."""
    autotune = directory / "xla_gpu_per_fusion_autotune_cache_dir"
    return sum(path.is_file() for path in autotune.rglob("*"))


def fixture_floor(output_directory: Path) -> None:
    """Compile all four fixture-floor cases while keeping outputs local."""
    from benchmarks import solovev_certificate as certificate

    certificate.REPOSED_FIXTURE_ROOT = output_directory
    certificate._measure_reposed_fixture(output_directory / "fixture-floor.json")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("fixture-floor", "count-autotune"))
    parser.add_argument("directory", type=Path)
    arguments = parser.parse_args()
    if arguments.action == "fixture-floor":
        fixture_floor(arguments.directory)
        return 0
    count = autotune_file_count(arguments.directory)
    print(f"PREWARM_AUTOTUNE_FILES={count}", flush=True)
    return int(count != 0)


if __name__ == "__main__":
    raise SystemExit(main())
