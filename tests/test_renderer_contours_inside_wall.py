"""Tree-wide guard: no renderer contours a scattered field outside the wall.

Every renderer under ``benchmarks/`` and ``docs/figures/`` that draws a field
interpolated from scattered nodes must mask it to the first wall first: a field
sampled at scattered nodes is finite across the whole node convex hull, so a
contour drawn from it crosses a concave wall unless the raster or the node
triangulation is blanked. ``tests/renderer_wall_scan.py`` reports any call that
reaches a contour routine on scattered data without a wall, so this guard
asserts that scan is empty over the renderer trees and names every offending
file and line when it is not.

The roots default to ``benchmarks`` and ``docs/figures`` relative to the
repository root. ``NOVA_RENDERER_SCAN_ROOTS`` overrides them with an
``os.pathsep``-separated list, so a scratch tree holding a deliberately
unguarded driver can be scanned through this same entry and must make the
guard fail naming that file and line.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import renderer_wall_scan as scan  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
DEFAULT_ROOTS = ("benchmarks", "docs/figures")


def _roots() -> list[Path]:
    override = os.environ.get("NOVA_RENDERER_SCAN_ROOTS")
    names = override.split(os.pathsep) if override else list(DEFAULT_ROOTS)
    return [Path(name) if Path(name).is_absolute() else REPO / name for name in names]


def unguarded_renderers(roots: list[Path] | None = None) -> list[scan.UnmaskedCall]:
    """Return every unguarded scattered-field renderer call under ``roots``.

    ``roots`` defaults to the renderer trees; the negative control passes a
    scratch directory through this same entry.
    """
    targets = [str(path) for path in (roots if roots is not None else _roots())]
    return scan.scan_paths(targets)


def test_no_renderer_contours_scattered_data_outside_the_wall():
    offenders = unguarded_renderers()
    if offenders:
        listing = "\n".join(f"  {record}" for record in offenders)
        raise AssertionError(
            f"{len(offenders)} unguarded scattered-field contour call(s) reach a "
            f"figure:\n{listing}"
        )
