"""The pre-warm publishes only a cache without per-fusion autotune files."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

LANE = Path(__file__).resolve().parents[1] / "scripts" / "h200_test_lane"
sys.path.insert(0, str(LANE))

from prewarm_rows import autotune_file_count  # noqa: E402


def test_autotune_count_accepts_a_compile_only_directory(tmp_path: Path) -> None:
    (tmp_path / "jit_solve-123-cache").touch()
    assert autotune_file_count(tmp_path) == 0


def test_autotune_count_refuses_one_persisted_file(tmp_path: Path) -> None:
    autotune = tmp_path / "xla_gpu_per_fusion_autotune_cache_dir"
    autotune.mkdir()
    (autotune / "record.textproto").touch()
    result = subprocess.run(
        [
            sys.executable,
            str(LANE / "prewarm_rows.py"),
            "count-autotune",
            str(tmp_path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode != 0
    assert result.stdout.strip() == "PREWARM_AUTOTUNE_FILES=1"


def test_identity_row_allows_numerical_failure_after_cache_write() -> None:
    launcher = (LANE / "prewarm.sh").read_text(encoding="utf-8")
    assert "compile_row forward-compile-identity --allow-pytest-failures pytest" in (
        launcher
    )
