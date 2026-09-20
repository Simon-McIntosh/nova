"""A panel's content is pinned, because its container is not reproducible.

A digest pin on a rendered figure is only useful if it fires when the drawing
moves and stays quiet when the writer merely rewrites the wrapper. That is not
true of these panels as shipped: the PNG bytes are reproducible, but the SVG
carries a wall-clock ``<dc:date>``, the Matplotlib version token, and generated
element ids, so its file digest advances on every render while the drawing stays
put. A file digest therefore cannot tell a regression from a library upgrade on
the vector file at all, and the natural repair — restoring the bytes so the pin
looks green — is indistinguishable from hiding a real move.

The pin is restated as content: decoded pixels for the bitmap, and the vector
text with the three volatile fields normalised away. The cases below exercise
the property rather than assert it: two renders of one revision are compared,
a faithful re-render is required to satisfy the pin, and a deliberately altered
panel is required not to.
"""

from __future__ import annotations

import copy
import hashlib
import io
import json
import re
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")

import pytest
from PIL import Image

from benchmarks import solovev_certificate as certificate

PIN = Path("docs/figures/gs-absolute-accuracy/figure-panel-pin.json")
NAME = "diverted-single-null-production-route-cells-300"
FIGURE_DIRECTORY = "docs/figures/gs-absolute-accuracy/solovev"

_DATE = re.compile(r"(<dc:date>)[^<]*(</dc:date>)")
_MATPLOTLIB = re.compile(r"Matplotlib v[0-9][0-9A-Za-z.\-]*")
_GENERATED_ID = re.compile(r"(?<=#)[A-Za-z0-9]{8,}|(?<=id=\")[A-Za-z0-9]{8,}")


def _bitmap_content_digest(raw: bytes) -> str:
    """Digest the decoded pixels, ignoring every container field."""

    pixels = np.asarray(Image.open(io.BytesIO(raw)).convert("RGBA"), dtype=np.uint8)
    digest = hashlib.sha256()
    digest.update(f"{pixels.shape}{pixels.dtype.str}".encode())
    digest.update(pixels.tobytes())
    return digest.hexdigest()


def _vector_content_digest(raw: bytes) -> str:
    """Digest the drawing, with the writer's volatile fields normalised away."""

    text = _DATE.sub(r"\1<normalised>\2", raw.decode("utf-8"))
    text = _MATPLOTLIB.sub("Matplotlib v<normalised>", text)
    seen: dict[str, str] = {}

    def rename(match: re.Match[str]) -> str:
        token = match.group(0)
        if token not in seen:
            seen[token] = f"ID{len(seen)}"
        return seen[token]

    return hashlib.sha256(_GENERATED_ID.sub(rename, text).encode("utf-8")).hexdigest()


def _pin() -> dict:
    assert PIN.is_file(), f"the panel content pin is missing: {PIN}"
    pin = json.loads(PIN.read_text(encoding="utf-8"))
    assert pin["schema"] == "nova.figure-panel-content-pin"
    return pin


def _pinned_panel() -> dict:
    panels = [p for p in _pin()["panels"] if p["case"] == "diverted-single-null"]
    assert len(panels) == 1, "the pin must carry exactly one diverted-single-null panel"
    return panels[0]


def _part() -> dict:
    panel = _pinned_panel()
    path = Path(panel["part"])
    assert path.is_file(), f"the persisted part the pin cites is missing: {path}"
    return json.loads(path.read_text(encoding="utf-8"))


def _render(row: dict, root: Path, directory: str) -> tuple[bytes, bytes]:
    """Render one panel under a private root, returning its two files' bytes."""

    relative = f"{directory}/panel.png"
    row = copy.deepcopy(row)
    row["stage_wall_seconds"] = {}
    row["figure"] = {"filesystem_path": relative}
    certificate._render_persisted_row(row)
    target = root / relative
    return target.read_bytes(), target.with_suffix(".svg").read_bytes()


@pytest.fixture()
def private_root(tmp_path, monkeypatch) -> Path:
    monkeypatch.setattr(certificate, "ROOT", tmp_path)
    return tmp_path


def test_two_renders_of_one_revision_move_the_container_and_not_the_panel(private_root):
    """The writer's determinism profile, measured rather than assumed."""

    part = _part()
    first_bitmap, first_vector = _render(
        part, private_root, "docs/figures/gs-absolute-accuracy/first"
    )
    second_bitmap, second_vector = _render(
        part, private_root, "docs/figures/gs-absolute-accuracy/second"
    )
    assert (
        hashlib.sha256(first_bitmap).hexdigest()
        == hashlib.sha256(second_bitmap).hexdigest()
    ), "the bitmap writer is not byte-reproducible at one revision"
    assert _bitmap_content_digest(first_bitmap) == _bitmap_content_digest(
        second_bitmap
    ), "two renders of one revision drew different pixels"
    assert _vector_content_digest(first_vector) == _vector_content_digest(
        second_vector
    ), "two renders of one revision drew different vector content"
    assert (
        hashlib.sha256(first_vector).hexdigest()
        != hashlib.sha256(second_vector).hexdigest()
    ), (
        "the vector container was expected to advance every render; if this case "
        "fails the content pin may be replaceable by a file digest again"
    )


def test_an_unchanged_panel_re_render_satisfies_the_content_pin(private_root):
    """The positively-controlled half: an unchanged panel must keep the pin."""

    panel = _pinned_panel()
    bitmap, vector = _render(
        _part(), private_root, "docs/figures/gs-absolute-accuracy/re-render"
    )
    assert _bitmap_content_digest(bitmap) == panel["bitmap_content_sha256"]
    assert _vector_content_digest(vector) == panel["vector_content_sha256"]


def test_a_deliberately_altered_panel_does_not_satisfy_the_content_pin(private_root):
    """The negatively-controlled half: a moved panel must break the pin.

    The alteration is a five-percent rescale of the solved flux, which moves the
    drawn contours while leaving the container's own fields untouched — exactly
    the move a file digest on the vector file cannot be trusted to catch.
    """

    panel = _pinned_panel()
    altered = _part()
    altered["render_data"]["terminal_flux_wb"] = list(
        1.05 * np.asarray(altered["render_data"]["terminal_flux_wb"], dtype=float)
    )
    bitmap, vector = _render(
        altered, private_root, "docs/figures/gs-absolute-accuracy/altered"
    )
    assert _bitmap_content_digest(bitmap) != panel["bitmap_content_sha256"], (
        "the content pin accepted a panel whose solved contours were rescaled"
    )
    assert _vector_content_digest(vector) != panel["vector_content_sha256"], (
        "the content pin accepted a rescaled panel on the vector file"
    )


def test_the_committed_panel_satisfies_the_pin_it_ships_with():
    """The pin is green on the shipped artifact, not only on a fresh render."""

    panel = _pinned_panel()
    bitmap = (certificate.ROOT / panel["bitmap_filesystem_path"]).read_bytes()
    vector = (certificate.ROOT / panel["vector_filesystem_path"]).read_bytes()
    assert hashlib.sha256(bitmap).hexdigest() == panel["file_sha256"]
    assert _bitmap_content_digest(bitmap) == panel["bitmap_content_sha256"]
    assert _vector_content_digest(vector) == panel["vector_content_sha256"]


def test_the_render_stage_writes_into_the_served_figure_tree(private_root):
    """The receipt stage mutates a published artifact as a side effect of measuring.

    ``_render_persisted_row`` resolves its output under ``ROOT/docs`` and writes
    both files there, so a measurement run rewrites the very artifact whose digest
    the receipt it produces is compared against. Recorded as the property the
    pin design has to survive, not as a behaviour to keep.
    """

    row = _part()
    row["stage_wall_seconds"] = {}
    row["figure"] = {"filesystem_path": f"{FIGURE_DIRECTORY}/{NAME}.png"}
    certificate._render_persisted_row(row)
    written = private_root / FIGURE_DIRECTORY / f"{NAME}.png"
    assert written.is_file(), (
        "the render stage should write into the served figure tree; if it no "
        "longer does, the pin and the driver can be separated by construction"
    )
    assert written.parent == private_root / FIGURE_DIRECTORY
