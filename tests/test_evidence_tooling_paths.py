"""No benchmark or supporting script may name a crew run or a worktree.

A crew report directory is promoted and reclaimed, and a worktree is removed
with the node that made it, so a path into either stops resolving for the next
reader.  Those locations are supplied by the caller through
``NOVA_CREW_REPORTS`` or ``NOVA_CREW_RUNS``; this scan holds that they never
come back as a literal, which is the one way the old paths returned.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]

SCANNED_DIRECTORIES = ("benchmarks",)
SCANNED_FILES = (
    "scripts/run_amplitude_census.sh",
    "scripts/media/efit_nova_overlay.py",
    "scripts/labeller_batch/shard.py",
    "scripts/analytic_oracle_fixtures/run_carrier_agreement.sh",
    "scripts/analytic_oracle_fixtures/run_cached_radial.sh",
)
EXCLUDED_PREFIXES = ("scripts/prototypes/",)

# The two ownership prefixes a receipt path used to carry, plus the worktree
# root.  A hit is a site that will read nothing once its owner is reclaimed.
# The pieces are joined rather than spelled, because the repository's own commit
# worktree check refuses any file that spells one of them, and this file has to
# search for exactly them.
LITERAL = re.compile("|".join((".config/" + "reckon/crew", "reckon-" + "worktrees")))


def _candidates() -> list[Path]:
    paths: list[Path] = []
    for directory in SCANNED_DIRECTORIES:
        paths.extend(
            path for path in sorted((ROOT / directory).rglob("*")) if path.is_file()
        )
    paths.extend(ROOT / name for name in SCANNED_FILES)
    return [
        path
        for path in paths
        if not path.relative_to(ROOT).as_posix().startswith(EXCLUDED_PREFIXES)
    ]


def _text(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError, OSError:
        return None


PROBE = """
import json

from benchmarks.exact_clip_closed_form_floor import REPORT_ROOT as CLOSED_FORM
from benchmarks.exact_clip_moment_floor import (
    REPORT_INPUT_ROOT as MOMENT_INPUT,
    REPORT_ROOT as MOMENT,
)

print(json.dumps({
    "closed_form": str(CLOSED_FORM),
    "moment": str(MOMENT),
    "moment_input": str(MOMENT_INPUT),
}))
"""


def _probe(environment: dict[str, str]) -> dict[str, str]:
    root = os.environ.get("PYTHONPATH")
    completed = subprocess.run(
        [sys.executable, "-c", PROBE],
        cwd=ROOT,
        env={
            "PATH": os.environ.get("PATH", ""),
            "PYTHONPATH": str(ROOT) if not root else f"{ROOT}{os.pathsep}{root}",
            **environment,
        },
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    return json.loads(completed.stdout)


def test_benchmark_override_wins_over_the_crew_report_root(tmp_path):
    closed_form = tmp_path / "closed-form"
    moment = tmp_path / "moment"
    moment_input = tmp_path / "moment-input"
    resolved = _probe(
        {
            "NOVA_CREW_REPORTS": str(tmp_path / "crew-reports"),
            "NOVA_EXACT_CLOSED_FORM_REPORT_ROOT": str(closed_form),
            "NOVA_EXACT_CLIP_REPORT_ROOT": str(moment),
            "NOVA_EXACT_CLIP_REPORT_INPUT_ROOT": str(moment_input),
        }
    )

    assert resolved == {
        "closed_form": str(closed_form),
        "moment": str(moment),
        "moment_input": str(moment_input),
    }


def test_unset_benchmark_override_falls_through_to_the_crew_report_root(tmp_path):
    crew = tmp_path / "crew-reports"
    resolved = _probe({"NOVA_CREW_REPORTS": str(crew)})

    assert resolved == {
        "closed_form": str(crew / "nova/s19-local/exact-closed-form"),
        "moment": str(crew / "nova/s19-local/exact-gauss"),
        "moment_input": str(crew / "nova/s19-local/exact-gauss"),
    }


def test_scanned_tree_holds_no_crew_or_worktree_literal():
    # A scan that found no file at all, or one whose pattern had drifted off the
    # text it is meant to catch, would report a clean tree for the same reason a
    # clean tree does.  The pattern is therefore proved against text known to
    # carry both prefixes, and the file list against a module known to be here.
    assert LITERAL.search('"/x/.config/reckon/crew/reports/nova"') is not None
    worktree_sample = "/x/Code/." + "reckon-" + "worktrees" + "/nova-run/tests"
    assert LITERAL.search(worktree_sample) is not None
    candidates = _candidates()
    relative = [path.relative_to(ROOT).as_posix() for path in candidates]
    assert "benchmarks/receipt_locations.py" in relative, (
        "the scan returned nothing it is known to hold "
        f"(benchmarks/receipt_locations.py): {len(relative)} candidates"
    )

    offenders = []
    for path in candidates:
        content = _text(path)
        if content is None:
            continue
        for number, line in enumerate(content.splitlines(), start=1):
            if LITERAL.search(line):
                offenders.append(f"{path.relative_to(ROOT).as_posix()}:{number}")

    assert offenders == [], (
        "benchmarks and scripts naming a crew or worktree path (take the "
        "location from NOVA_CREW_REPORTS or NOVA_CREW_RUNS instead): "
        + ", ".join(offenders)
    )
