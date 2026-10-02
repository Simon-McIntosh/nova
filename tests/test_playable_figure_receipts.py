"""Every number a playable figure draws is the receipt field it was read from.

The playable render drivers are scripts: they open a committed receipt, draw a
panel, and write a PNG. Nothing about that pipeline pins the drawn text to the
receipt it came from, so a drifted caption, or a receipt replaced under a figure
that still shows the old state, stays green in both directions.

This module re-runs the drivers with the figure writer stubbed out, captures the
title and label strings the drivers actually draw, and asserts each against the
field the driver read it from. The drivers read their receipts under
``PLAYABLE_FIGURE_RECEIPTS_ROOT`` when that names a tree, so a drift can be
produced against a perturbed copy of a receipt while the expectation is still
computed from the committed one; with the variable unset both resolve to the
repository, which is the head case.

The drivers are imported unchanged and no image is written to the repository
tree: ``Figure.savefig`` is replaced with a no-op for the drivers' duration.
"""

from __future__ import annotations

import importlib.util
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace

import matplotlib

matplotlib.use("Agg", force=True)

import matplotlib.axes  # noqa: E402
import matplotlib.figure  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
PFS = "docs/figures/playable-forward-solve"
SUPPLEMENT = (
    REPO_ROOT
    / "docs/figures/null-identification-authority/convergence-atlas/supplement"
)

_OVERRIDE = os.environ.get("PLAYABLE_FIGURE_RECEIPTS_ROOT")
RECEIPTS_ROOT = Path(_OVERRIDE).resolve() if _OVERRIDE else REPO_ROOT

_DRIVER_PATHS = {
    "panels": f"{PFS}/pfs-anchor-and-panel-repairs/render_panels.py",
    "attribution": f"{PFS}/pfs-anchor-and-panel-repairs/render_attribution.py",
    "figures": f"{PFS}/pfs-keyframes-and-compensator/render_figures.py",
}


@dataclass
class Drawn:
    """The title and label strings a driver drew, in call order."""

    titles: list[str] = field(default_factory=list)
    texts: list[str] = field(default_factory=list)


def _load(name: str, relative: str):
    spec = importlib.util.spec_from_file_location(name, REPO_ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _read(root: Path, *parts: str) -> dict:
    return json.loads(root.joinpath(*parts).read_text(encoding="utf-8"))


def _expected_receipt(*parts: str) -> dict:
    return _read(REPO_ROOT, *parts)


@pytest.fixture()
def driven(monkeypatch):
    """Load the three drivers, point them at the receipt tree, capture drawing."""

    panels = _load("pfs_render_panels", _DRIVER_PATHS["panels"])
    attribution = _load("pfs_render_attribution", _DRIVER_PATHS["attribution"])
    figures = _load("pfs_render_figures", _DRIVER_PATHS["figures"])

    monkeypatch.setattr(panels, "LIMITED", RECEIPTS_ROOT / PFS / "limited-anchor")
    monkeypatch.setattr(panels, "EARLY", RECEIPTS_ROOT / PFS / "early-frame-placement")
    monkeypatch.setattr(panels, "SUPPLEMENT", SUPPLEMENT)
    monkeypatch.setattr(
        attribution, "CENTROID", RECEIPTS_ROOT / PFS / "centroid-attribution"
    )
    monkeypatch.setattr(
        attribution, "BATCHED", RECEIPTS_ROOT / PFS / "batched-labeller"
    )

    drawn = Drawn()
    original_title = matplotlib.axes.Axes.set_title
    original_text = matplotlib.axes.Axes.text
    original_annotate = matplotlib.axes.Axes.annotate

    def set_title(self, label, *args, **kwargs):
        drawn.titles.append(str(label))
        return original_title(self, label, *args, **kwargs)

    def text(self, x, y, string, *args, **kwargs):
        drawn.texts.append(str(string))
        return original_text(self, x, y, string, *args, **kwargs)

    def annotate(self, string, *args, **kwargs):
        drawn.texts.append(str(string))
        return original_annotate(self, string, *args, **kwargs)

    monkeypatch.setattr(matplotlib.axes.Axes, "set_title", set_title)
    monkeypatch.setattr(matplotlib.axes.Axes, "text", text)
    monkeypatch.setattr(matplotlib.axes.Axes, "annotate", annotate)
    monkeypatch.setattr(matplotlib.figure.Figure, "savefig", lambda self, *a, **k: None)

    return SimpleNamespace(
        panels=panels, attribution=attribution, figures=figures, drawn=drawn
    )


def _ms(time_s: float, digits: int) -> str:
    return f"{1e3 * time_s:.{digits}f}"


def test_limited_anchor_titles_state_the_receipt(driven):
    """Rows 11 to 17: the refusal word for a refused arm, boundary flux otherwise."""

    expected = _expected_receipt(
        PFS, "limited-anchor", "limited-anchor-containment.json"
    )

    driven.panels.render_limited_anchor()
    titles = driven.drawn.titles
    frames = sorted(expected["frames"], key=lambda item: item["manifest_row"])
    assert len(titles) == len(frames), "one panel title per census frame"

    shot = expected["shot"]
    refused_expected = 0
    for title, frame in zip(titles, frames):
        public = frame["public_read"]
        row = frame["manifest_row"]
        if frame["converged"] and public["contour_closed"]:
            fraction = (public["boundary_flux_wb"] - public["axis_flux_wb"]) / (
                public["x_point_flux_wb"] - public["axis_flux_wb"]
            )
            separation_mm = 1e3 * float(
                np.hypot(
                    *(
                        np.asarray(
                            public["selected_anchor_node_position_m"], dtype=float
                        )
                        - np.asarray(public["boundary_position_m"], dtype=float)
                    )
                )
            )
            assert (
                f"MAST {shot}, row {row}, t = {_ms(frame['time_s'], 0)} ms — converged"
                in title
            )
            assert f"drawn boundary level psi_N = {fraction:.4f} of axis->X" in title
            assert f"anchor->boundary separation {separation_mm:.1f} mm" in title
        else:
            refused_expected += 1
            header = (
                f"MAST {shot}, row {row}, t = {_ms(frame['time_s'], 1)} ms "
                "— converged=False"
            )
            assert header in title
            assert (
                f"max containment {frame['maximum_containment_fraction']:.3f}" in title
            )
            assert "contour closed=False" in title

    refused = [
        t for t in driven.drawn.texts if t == "REFUSED: no closed solved contour"
    ]
    assert len(refused) == refused_expected, (
        "each refused panel draws the refusal word once"
    )


def test_early_frame_placement_titles_state_the_receipt(driven):
    """The grid title carries each arm's convergence state from the receipt."""

    expected = _expected_receipt(
        PFS, "early-frame-placement", "early-frame-placement.json"
    )

    driven.panels.render_early_frame_placement()
    titles = driven.drawn.titles
    rows = sorted(expected["rows"], key=lambda item: item["manifest_row"])
    assert len(titles) == len(rows), "one grid title per placement row"

    codes = {"free": "free", "conditioned": "cold", "conditioned_warm": "warm"}
    for title, frame in zip(titles, rows):
        assert (
            f"row {frame['manifest_row']}, t = {_ms(frame['time_s'], 0)} ms, "
            f"{frame['requested_class']}" in title
        )
        arms = [arm for arm in codes if frame.get(arm) is not None]
        for arm in arms:
            entry = frame[arm]
            ratio = entry.get("enclosed_area_ratio")
            ratio_text = "—" if ratio is None else f"{ratio:.3f}"
            assert (
                f"{codes[arm]} n{entry.get('first_contact_node')} r{ratio_text}"
                in title
            )
        unconverged = [
            codes[arm] for arm in arms if not frame[arm].get("converged", False)
        ]
        refused_line = (
            "unconverged arms: " + ", ".join(unconverged)
            if unconverged
            else "all arms converged"
        )
        assert refused_line in title


def test_centroid_attribution_class_panel_states_the_receipt(driven):
    """The class panel's agree count and 2x2 counts are the receipt's frozen rows."""

    expected = _expected_receipt(
        PFS, "centroid-attribution", "centroid-radius-attribution.json"
    )
    frozen = [row for row in expected["rows"] if row["operator_anchor"] == "frozen"]
    agree = sum(int(row["classes_agree"]) for row in frozen)

    driven.attribution.render_centroid_attribution()
    assert (
        f"Requested vs emergent class (frozen anchor; {agree}/{len(frozen)} agree)"
        in driven.drawn.titles
    )
    assert (
        f"Offset attribution by candidate (frozen anchor; {len(frozen)} rows)"
        in driven.drawn.titles
    )

    counts = np.zeros((2, 2), dtype=int)
    for row in frozen:
        counts[
            int(bool(row["requested_diverted"])), int(bool(row["emergent_diverted"]))
        ] += 1
    for i in range(2):
        for j in range(2):
            assert str(counts[i, j]) in driven.drawn.texts


def test_batched_labeller_throughput_panel_states_the_receipt(driven):
    """Each arm's drawn rate is the receipt's attempted slices per second."""

    expected = _expected_receipt(
        PFS, "batched-labeller", "h200-throughput-receipt.json"
    )

    driven.attribution.render_h200_throughput()
    for arm in expected["arms"]:
        assert f"{arm['attempted_slices_per_second']:.2f}" in driven.drawn.texts
    reference = expected["sequential_compiled_reference"]
    assert f"{reference['slice_count']}-slice reference" in driven.drawn.titles[0]


def test_batched_labeller_parity_panel_states_the_receipt(driven):
    """The parity title carries the receipt's denominator and parity flag."""

    expected = _expected_receipt(PFS, "batched-labeller", "acceptance-receipt.json")
    parity = expected["fraction_parity"]

    driven.attribution.render_terminal_state_parity()
    assert (
        f"({parity['denominator']} slices; parity holds: "
        f"{expected['acceptance_contract']['fraction_parity']})"
        in driven.drawn.titles[0]
    )


def test_keyframes_throughput_series_states_the_receipt(driven):
    """Each drawn keyframe rate is one over the receipt's press wall."""

    expected = _expected_receipt(PFS, "keyframes", "h200-press-stage-receipt.json")

    driven.figures.render_keyframes_throughput(RECEIPTS_ROOT)
    for press in expected["presses"]:
        label = "prime (cold)" if press.get("press") is None else "first moved (warm)"
        assert f"{label}\n{1.0 / press['wall']:.4f}" in driven.drawn.texts
    assert (
        f"({expected['verdict']['presses_measured']} warm press measured;"
        in driven.drawn.titles[0]
    )


def test_compensator_authority_groups_state_the_receipt(driven):
    """Rows 15-17 and 95-97: each drawn derivative is the receipt's absolute value."""

    expected = _expected_receipt(
        PFS, "compensator-authority", "compensator-authority.json"
    )
    rows = {row["row"]: row for row in expected["rows"]}

    driven.figures.render_compensator_authority(RECEIPTS_ROOT)
    for group in (expected["measured_early_rows"], expected["flat_top_rows"]):
        for row in group:
            derivative = abs(rows[row]["selected_derivative_m_per_a"])
            assert f"{derivative:.2e}" in driven.drawn.texts
    row_96 = rows[expected["flat_top_rows"][1]]
    assert f"row 96 selects {row_96['selected_circuit']}" in driven.drawn.texts


def test_axis_configuration_panel_states_the_receipt(driven):
    """The axis figure's identity line and per-arm offsets are the receipt's."""

    expected = _expected_receipt(
        PFS, "axis-configuration", "axis-configuration-receipt.json"
    )

    driven.figures.render_axis_configuration(RECEIPTS_ROOT)
    assert (
        f"MAST {expected['shot']}, row {expected['slice_index']}, "
        f"t = {expected['time_s']:.3f} s" in driven.drawn.titles[0]
    )
    for arm in expected["arms"]:
        assert f"{arm['offset_from_efit_cm']:+.2f} cm" in driven.drawn.texts
    wide = [arm for arm in expected["arms"] if arm["arm"] in ("P", "W")]
    magnitude = abs(wide[0]["offset_from_efit_cm"])
    assert f"P and W are {magnitude:.2f} cm" in driven.drawn.titles[1]
