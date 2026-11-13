"""Accept only cold production-row receipts that meet the splitter budget."""

import json
import os
from pathlib import Path

import pytest

from scripts.prototypes.compile_pass_profile.summarise import summarise


@pytest.fixture(scope="module")
def compile_profiles():
    paths = [
        os.environ.get(f"NOVA_PRODUCTION_COMPILE_{arm}") for arm in ("BASE", "HEAD")
    ]
    if not all(paths):
        pytest.skip("requires paired cold H200 production-row receipts")
    profiles = []
    for path in paths:
        directory = Path(path)
        profile = json.loads((directory / "profile.json").read_text())
        assert profile["phase"] == "done"
        assert profile["program"] == "production_point_batch"
        assert profile["size"] == 132
        assert "H200" in profile["device"]
        assert profile["cache_enabled"] is False
        assert profile["cache_guard_hits"] == 0
        assert profile["cache_guard_misses"] > 0
        attribution = json.loads((directory / "attribution.json").read_text())
        assert profile["revision"] == attribution["revision"]
        assert profile["compile_seconds"] == attribution["compile_seconds"]
        profiles.append(summarise(attribution))
    return profiles


def _splitter(profile):
    rows = [
        row for row in profile["table"] if row["pass"].endswith("/variadic-op-splitter")
    ]
    assert len(rows) == 1
    row = rows[0]
    assert row["instructions_after"] > row["instructions_entry"] > 0
    return row


def test_production_row_splitter_instruction_budget(compile_profiles):
    base, head = map(_splitter, compile_profiles)
    print(
        f"SPLITTER base={base['instructions_after']} head={head['instructions_after']}"
    )
    assert head["instructions_after"] * 2 <= base["instructions_after"]


def test_production_row_cold_compile_savings(compile_profiles):
    base, head = compile_profiles
    seconds = sum(
        row["seconds"]
        for row in base["table"]
        if row["pass"].endswith(("/variadic-op-splitter", "/multi_output_fusion"))
    )
    assert seconds > 0
    bound = base["compile_seconds"] - seconds
    print(
        f"COMPILE base={base['compile_seconds']} "
        f"head={head['compile_seconds']} bound={bound}"
    )
    assert head["compile_seconds"] <= bound
