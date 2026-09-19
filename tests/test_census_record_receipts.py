"""The program-shape census records stay tied to their receipts and their figure.

The census receipt and the bank identity receipt each carry their own cache
capture, the bucket figure's bars are drawn from the census receipt, and the
evidence anchor repeats both. These assertions hold those four statements to
each other, so a later edit that walks one of the repairs back fails here.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SHAPES = ROOT / "docs/figures/millisecond-converged-solve/program-shapes"
BANK = SHAPES / "bank-shape-identity.json"
CENSUS = SHAPES / "program-shape-census.json"
BUCKETS_PNG = SHAPES / "program-shape-buckets.png"
BUCKETS_SVG = SHAPES / "program-shape-buckets.svg"
EVIDENCE = ROOT / "docs/evidence/archive/millisecond-converged-solve-landed.html"
OPERAND_CACHE = ROOT / "logs/exact-operand-cache.npz"

BANK_CACHE_BYTES = 20_044_389_103
CENSUS_CACHE_BYTES = 20_093_357_205
WORD = {1: "one", 2: "two", 5: "five", 12: "twelve", 16: "sixteen"}


def _load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def _gib(byte_count: int) -> float:
    return round(byte_count / 2**30, 3)


def test_each_capture_divides_to_its_own_gib_reading():
    bank = _load(BANK)["cache_inventory"]
    assert abs(_gib(int(bank["bytes"])) - float(bank["gib"])) < 5e-4
    census = _load(CENSUS)["cache"]
    assert abs(_gib(int(census["total_bytes"])) - float(census["total_gib"])) < 5e-4


def test_the_two_captures_are_distinct_and_the_anchor_states_each():
    bank = _load(BANK)["cache_inventory"]
    census = _load(CENSUS)["cache"]
    assert int(bank["bytes"]) == BANK_CACHE_BYTES
    assert int(census["total_bytes"]) == CENSUS_CACHE_BYTES
    assert bank["gib"] != census["total_gib"]
    evidence = EVIDENCE.read_text(encoding="utf-8")
    assert "3,989 entries at 18.713 GiB" in evidence
    assert "reads 18.668 GiB for its byte count" in evidence
    assert census["total_entries"] == 3_989
    assert census["runtime_keys"] == 12


def test_the_bank_receipt_records_the_measured_flux_difference():
    bank = _load(BANK)["result"]["geometry_value_agreement"]
    census = _load(CENSUS)["bank"]["geometry_value_agreement"]
    assert float(bank["flux_max_abs_difference"]) == round(
        float(census["operand_only_arrays"]["flux"]), 3
    )
    assert float(bank["flux_max_abs_difference"]) == 0.482
    for field in ("radius", "height", "wall"):
        assert bank[f"{field}_max_abs_difference"] == 0.0
    evidence = EVIDENCE.read_text(encoding="utf-8")
    assert "at max absolute difference 0.0" in evidence
    assert "the only array that differs is <code>flux</code>, at 0.482" in evidence


def test_the_operand_cache_cites_its_own_size():
    assert OPERAND_CACHE.is_file()
    cited = _load(BANK)["source"]["operand_cache"]
    match = re.search(r"\((\d+) bytes", cited)
    assert match is not None
    assert int(match.group(1)) == OPERAND_CACHE.stat().st_size == 180_876
    assert "89486" not in cited


def test_the_cited_paths_exist():
    instrument = _load(BANK)["instrument"].split()[0]
    for path in (
        ROOT / instrument,
        BANK,
        CENSUS,
        SHAPES / "program-key-and-pruner.json",
        BUCKETS_PNG,
        BUCKETS_SVG,
        ROOT
        / "docs/figures/millisecond-converged-solve/compiled-slice-cache/receipt.json",
    ):
        assert path.is_file(), path
    assert BUCKETS_PNG.stat().st_size > 10_000
    assert BUCKETS_SVG.stat().st_size > 10_000


def test_the_bucket_bar_values_match_the_census_receipt_and_the_anchor():
    census = _load(CENSUS)
    today = (
        census["bank"]["program_count_today"]
        + census["certificate"]["distinct_capacity_vectors_today"]
    )
    design = 1 + census["certificate"]["distinct_bucketed_vectors"]
    assert today == census["combined"]["distinct_capacity_vectors_today"] == 5
    assert design == census["combined"]["distinct_bucketed_vectors"] == 2

    svg = BUCKETS_SVG.read_text(encoding="utf-8", errors="replace")
    assert f"today ({today} programs)" in svg
    assert f"one per bucket ({design})" in svg
    assert "padding waste per capacity axis" in svg
    assert "program count" in svg

    evidence = EVIDENCE.read_text(encoding="utf-8")
    assert f"present {WORD[today]} capacity vectors today" in evidence
    assert f"collapse to {WORD[design]} bucketed programs" in evidence
    assert (
        f"The {WORD[census['bank']['identities']]} bank arms already share one program"
        in evidence
    )
    assert census["bank"]["identities"] == 12
    assert census["combined"]["identities"] == 16


def test_the_anchor_carries_the_census_section_figure_and_no_shim_entry():
    census = _load(CENSUS)
    evidence = EVIDENCE.read_text(encoding="utf-8")
    assert 'id="program-shape-census"' in evidence
    assert "program-shape-buckets.png" in evidence
    assert "SVG twin" in evidence
    assert "program-shape-census.json" in evidence
    assert "framework_compat_shims" not in census, (
        "the census must carry no compatibility shim"
    )
    assert census["certificate_source"].startswith("benchmarks/solovev_certificate")
