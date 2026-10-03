"""Reproduce the frozen-current figure pair from its complete measured receipt."""

from __future__ import annotations

import importlib
import json
import os
from pathlib import Path
import re
import xml.etree.ElementTree as ET

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
FIGURES = ROOT / "docs/figures/cut-cell-current-attribution/gate-b"
SVG = "{http://www.w3.org/2000/svg}"
PANELS = {
    "absolute_error": ("absolute-flux-error.svg", 6),
    "weak_110_comparison": ("weak-110-frozen-prediction-analytic-terminal.svg", 3),
}


def _driver():
    return importlib.import_module("benchmarks.frozen_current_flux_image_gate")


def _receipt_path():
    return Path(os.environ.get("NOVA_FROZEN_CURRENT_RECEIPT", FIGURES / "receipt.json"))


def _drawing(path):
    text = path.read_text(encoding="utf-8")
    root = ET.fromstring(text)
    axes = [
        group
        for group in root.iter(SVG + "g")
        if group.get("id", "").startswith("axes_")
    ]
    # Contour and wall paths carry the numerical drawing independently of SVG ids.
    paths = [
        [(item.get("d"), item.get("style")) for item in group.findall(SVG + "path")]
        for group in root.iter(SVG + "g")
        if group.get("id", "").startswith(("TriContourSet_", "line2d_"))
        and group.findall(SVG + "path")
    ]
    markers = [
        (item.get("x"), item.get("y"))
        for group in root.iter(SVG + "g")
        if group.get("id", "").startswith("line2d_")
        for item in group.findall(".//" + SVG + "use")
    ]
    return len(axes), paths, re.findall(r"<!-- (.*?) -->", text, re.DOTALL), markers


def test_driver_imports_from_this_checkout():
    driver = _driver()
    assert (
        Path(driver.__file__).resolve()
        == ROOT / "benchmarks/frozen_current_flux_image_gate.py"
    )
    assert driver.jax.default_backend() == "cpu"


def test_render_only_reproduces_receipted_panel_set(tmp_path, monkeypatch):
    driver = _driver()

    def refuse_measurement(*args, **kwargs):
        raise AssertionError("render-only entered the measurement path")

    for name in ("measure_rung", "aggregate", "_run_all"):
        monkeypatch.setattr(driver, name, refuse_measurement)
    monkeypatch.setattr(driver.certificate, "_case_machine", refuse_measurement)
    receipt = json.loads((FIGURES / "receipt.json").read_text(encoding="utf-8"))
    before = {path: path.read_bytes() for path in (FIGURES / "parts").glob("*.json")}
    figures = driver.render_from_receipt(
        _receipt_path(), tmp_path, parts_directory=FIGURES / "parts"
    )
    assert driver.jax.config.jax_enable_x64 is True
    assert set(figures) == set(PANELS)
    assert {path.name for path in tmp_path.iterdir()} == {
        name for name, _ in PANELS.values()
    }
    for key, (name, count) in PANELS.items():
        output = tmp_path / name
        actual = _drawing(output)
        expected = _drawing(FIGURES / name)
        assert actual[0] == expected[0] == count
        assert actual[1] and any(d for group in actual[1] for d, _ in group)
        assert actual[1:] == expected[1:]
        assert len(actual[3]) == (8 if key == "absolute_error" else 3)
        level_key = next(
            key for key in receipt["figures"][key] if key.startswith("shared_")
        )
        np.testing.assert_array_equal(
            figures[key][level_key], receipt["figures"][key][level_key]
        )
    assert before == {path: path.read_bytes() for path in before}


def test_render_refuses_a_missing_receipt_row_before_drawing(tmp_path):
    receipt = json.loads((FIGURES / "receipt.json").read_text(encoding="utf-8"))
    receipt["rows"].pop()
    partial = tmp_path / "receipt.json"
    partial.write_text(json.dumps(receipt), encoding="utf-8")
    output = tmp_path / "figures"
    with pytest.raises(ValueError, match="all six declared rows"):
        _driver().render_from_receipt(
            partial, output, parts_directory=FIGURES / "parts"
        )
    assert not output.exists()
