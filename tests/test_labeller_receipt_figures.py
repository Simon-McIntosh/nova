"""The three labeller-receipt figures report one corpus and no fraction above one.

The figures under ``docs/figures/playable-forward-solve/labeller-receipt`` are
regenerated from their ``receipt.json`` by
``benchmarks/batched_labeller_receipt.py render``, which also writes
``render-receipt.json`` beside them. These tests read that render receipt and
the source receipt only: they prove that every figure header states the same
corpus, that every number a figure prints is read from the receipt by the named
JSON path the render receipt carries, that every fraction drawn lies in
[0, 1], and that the PNG on disk is the regenerated one (its embedded title
equals the render receipt's title). Set ``NOVA_LABELLER_RECEIPT_FIG_DIR`` to a
figure directory to run against another copy.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest


DEFAULT_FIGURE_DIR = (
    Path(__file__).resolve().parents[1]
    / "docs/figures/playable-forward-solve/labeller-receipt"
)

FIGURE_NAMES = (
    "labeller-receipt-adjudication",
    "pin-tail-crosstab",
    "signed-error-summaries",
)


def _figure_dir() -> Path:
    return Path(os.environ.get("NOVA_LABELLER_RECEIPT_FIG_DIR", DEFAULT_FIGURE_DIR))


def _at(receipt: dict, path: str):
    value = receipt
    for key in path.split("."):
        value = value[int(key)] if isinstance(value, list) else value[key]
    return value


@pytest.fixture(scope="module")
def render_receipt() -> dict:
    directory = _figure_dir()
    return json.loads((directory / "render-receipt.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def source_receipt() -> dict:
    directory = _figure_dir()
    return json.loads((directory / "receipt.json").read_text(encoding="utf-8"))


def _figures(render_receipt: dict) -> dict[str, dict]:
    return {figure["name"]: figure for figure in render_receipt["figures"]}


def _corpus_string(shots: int, slices: int) -> tuple[str, str]:
    return (f"{shots} completed shots", f"{slices} written slices")


def test_the_three_named_figures_are_present(render_receipt: dict) -> None:
    assert set(_figures(render_receipt)) == set(FIGURE_NAMES)


def test_every_figure_header_states_the_same_one_corpus(
    render_receipt: dict, source_receipt: dict
) -> None:
    corpus = render_receipt["corpus"]
    shots = int(_at(source_receipt, corpus["shots_json_path"]))
    slices = int(_at(source_receipt, corpus["slices_json_path"]))
    assert corpus["completed_shots"] == shots
    assert corpus["slices"] == slices
    shots_text, slices_text = _corpus_string(shots, slices)
    titles = {figure["title"] for figure in render_receipt["figures"]}
    assert len(titles) == 1, f"three corpus headers, not one: {sorted(titles)}"
    title = titles.pop()
    assert shots_text in title and slices_text in title


def test_every_printed_number_matches_its_named_receipt_path(
    render_receipt: dict, source_receipt: dict
) -> None:
    for figure in render_receipt["figures"]:
        for declaration in figure["declarations"]:
            value = declaration["value"]
            if "json_path" in declaration:
                expected = float(_at(source_receipt, declaration["json_path"]))
                assert value == expected, (figure["name"], declaration)
            elif "sum_json_paths" in declaration:
                expected = float(
                    sum(
                        _at(source_receipt, path)
                        for path in declaration["sum_json_paths"]
                    )
                )
                assert value == expected, (figure["name"], declaration)
            else:
                numerical = float(
                    _at(source_receipt, declaration["numerator_json_path"])
                )
                if "denominator_json_path" in declaration:
                    denominator = float(
                        _at(source_receipt, declaration["denominator_json_path"])
                    )
                else:
                    denominator = float(
                        sum(
                            _at(source_receipt, path)
                            for path in declaration["denominator_json_paths"]
                        )
                    )
                assert value == pytest.approx(numerical / denominator), (
                    figure["name"],
                    declaration,
                )


def test_every_printed_fraction_lies_in_zero_to_one(render_receipt: dict) -> None:
    offenders = [
        (figure["name"], declaration["label"], declaration["value"])
        for figure in render_receipt["figures"]
        for declaration in figure["declarations"]
        if declaration["kind"] == "fraction" and not 0.0 <= declaration["value"] <= 1.0
    ]
    assert not offenders, f"fractions outside [0, 1]: {offenders}"


def test_the_figure_on_disk_carries_the_regenerated_title(render_receipt: dict) -> None:
    from PIL import Image

    directory = _figure_dir()
    for figure in render_receipt["figures"]:
        image = Image.open(directory / f"{figure['name']}.png")
        assert image.text.get("Title") == figure["title"], figure["name"]
