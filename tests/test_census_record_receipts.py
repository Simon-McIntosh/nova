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
SVG_PATH = re.compile(r'<path d="([^"]*)"[^>]*style="([^"]*)"')
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


def _numbers(path_data: str) -> list[float]:
    return [float(value) for value in re.findall(r"-?\d+\.?\d*", path_data)]


def _left_panel_geometry(svg_text: str):
    """Read back the left panel's bar right edges, its origin and its ratio-one line.

    The renderer draws one horizontal bar per observed capacity, every bar
    starting at data zero, and marks ratio one with a dashed vertical line. Both
    the origin and the unit of the horizontal axis are read from the SVG rather
    than assumed, so a bar drawn from a different ratio than the receipt states
    cannot pass by sharing the receipt's own scale.
    """
    rectangles = []
    verticals = []
    dashed = []
    for data, style in SVG_PATH.findall(svg_text):
        values = _numbers(data)
        fill = re.search(r"fill:\s*(#[0-9a-fA-F]{6})", style)
        if fill is not None and len(values) == 8:
            xs, ys = values[0::2], values[1::2]
            if len(set(xs)) == 2 and len(set(ys)) == 2:
                rectangles.append((min(xs), max(xs), max(ys), fill.group(1)))
        elif len(values) == 4 and values[0] == values[2]:
            (dashed if "stroke-dasharray" in style else verticals).append(values[0])

    spine_xs = sorted(set(verticals))
    origin = min(
        x for x in spine_xs if any(abs(box[0] - x) < 1e-6 for box in rectangles)
    )
    edge = min(x for x in spine_xs if x > origin)
    unit = [x for x in dashed if origin < x < edge]
    assert len(unit) == 1, f"expected one ratio-one line in the left panel, saw {unit}"
    # The axes patch shares the bars' origin and the white figure patch does not
    # reach it, so a colour is what separates a bar from the panel background.
    bars = [
        box for box in rectangles if abs(box[0] - origin) < 1e-6 and box[3] != "#ffffff"
    ]
    return origin, unit[0], bars


def _census_ratio_sequence(census: dict):
    """The receipt's padding ratios in the order the renderer draws them.

    The renderer orders the axes by their largest ratio and the bars inside an
    axis by capacity, lowest first, so the panel's bottom-up order is the
    receipt's own order rather than a figure-side convention.
    """
    waste = census["combined"]["padding_waste"]
    axes = sorted(waste, key=lambda name: -max(waste[name]["padding_ratio"].values()))
    sequence = []
    for name in axes:
        ratios = sorted(
            waste[name]["padding_ratio"].items(), key=lambda pair: float(pair[0])
        )
        sequence.extend((name, float(ratio)) for _, ratio in ratios)
    return sequence


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

    origin, unit_x, bars = _left_panel_geometry(svg)
    sequence = _census_ratio_sequence(census)
    bars.sort(key=lambda bar: -bar[2])
    assert len(bars) == len(sequence), (
        f"the left panel draws {len(bars)} bars against {len(sequence)} receipt ratios"
    )
    scale = unit_x - origin
    assert scale > 0
    for (axis, ratio), (_, right, _, _) in zip(sequence, bars):
        drawn = (right - origin) / scale
        assert abs(drawn - ratio) <= 1e-3 * ratio, (
            f"{axis} bar drawn at ratio {drawn:.4f} against the receipt's {ratio}"
        )

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
