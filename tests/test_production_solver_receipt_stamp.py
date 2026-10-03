"""The production-solver receipt writer stamps its payload with its revision."""

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import re

from benchmarks import solovev_certificate as certificate


ROOT = Path(__file__).resolve().parents[1]
MEASURE = (
    ROOT / "docs/figures/gs-absolute-accuracy/production-solver-receipt/measure.py"
)
SOURCE_REVISION = re.compile(r"[0-9a-f]{40}")


def _measure_module():
    spec = spec_from_file_location("production_solver_receipt_measure", MEASURE)
    assert spec is not None and spec.loader is not None
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _payload():
    """Build the receipt payload without running the solve."""
    module = _measure_module()
    return module.build_payload(
        revision=certificate._source_revision(),
        case="weak-rotation-reactor-static",
        requested_cells=300,
        realised_cells=300,
        clip_mode="chord",
        job_id=None,
        platform="cpu",
        x64=True,
        history_shapes={"residual": [], "converged": []},
        residual=1.0e-13,
        converged=True,
        production_solver={},
        elapsed_seconds=0.0,
    )


def test_payload_carries_a_source_revision():
    payload = _payload()
    revision = payload.get("source_revision")
    assert isinstance(revision, str), (
        "the production-solver receipt must name the revision that produced it"
    )
    assert SOURCE_REVISION.fullmatch(revision), (
        f"source_revision is not a 40-hex sha: {revision!r}"
    )


def test_source_revision_equals_the_recorded_revision():
    payload = _payload()
    assert payload["source_revision"] == payload["revision"]
