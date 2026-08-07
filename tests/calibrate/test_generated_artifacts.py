"""Whether the committed models still are what the schema generates.

The Pydantic models and the JSON-Schema export are committed so that reading a
correction document needs neither the LinkML toolchain nor a build step.  The cost
of committing generated code is that it can be edited, and an edit is invisible: the
models keep working, they simply stop describing the schema everything else is
validated against.

So the schema is regenerated here and compared byte for byte.  The toolchain lives
in an optional extra, so this skips where it is absent -- which is most lanes, and
the reason the committed files exist at all.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from nova.calibrate.correction_set import SCHEMA_PATH

REPOSITORY = SCHEMA_PATH.parents[3]
MODEL_PATH = SCHEMA_PATH.parents[1] / "correction_model.py"
JSON_SCHEMA_PATH = SCHEMA_PATH.with_suffix(".schema.json")


def generate(generator: str) -> str:
    """Return what a LinkML generator makes of the authored schema.

    The generator is invoked as the README documents it -- by console script, from
    the repository root, on the repository-relative schema path -- so the command
    this test holds the committed files to is the command a developer runs.  The
    path matters: the Pydantic generator records the schema path it was handed, so
    generating through an absolute path would embed one developer's checkout.
    """

    script = Path(sys.executable).with_name(generator)
    if not script.exists():
        pytest.skip(f"{generator} is in the schema extra, which is not installed")
    finished = subprocess.run(
        [str(script), str(SCHEMA_PATH.relative_to(REPOSITORY))],
        capture_output=True,
        text=True,
        check=False,
        cwd=REPOSITORY,
    )
    if finished.returncode != 0:
        pytest.fail(f"{generator} failed: {finished.stderr[-2000:]}")
    return finished.stdout


def test_the_committed_models_match_the_schema():
    assert generate("gen-pydantic") == MODEL_PATH.read_text()


def test_the_committed_json_schema_matches_the_schema():
    assert generate("gen-json-schema") == JSON_SCHEMA_PATH.read_text()


def test_the_generated_artifacts_are_committed():
    for path in (MODEL_PATH, JSON_SCHEMA_PATH):
        assert Path(path).exists()
