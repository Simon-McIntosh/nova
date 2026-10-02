"""Re-render the two scan-prototype per-trip residual SVGs from their receipts.

Each SVG is rendered from its *own* platform's receipts:
``parts/{platform}-production-execution.json`` and
``parts/{platform}-scanned-execution.json``, both of which carry the active-set
residual history under ``terminal.active_set_residuals``. The two former SVGs
were byte-identical apart from the render timestamp. This script re-renders each
from its own platform's numbers and prints the cross-platform residual
difference so the record can state whether the backends agree.

Reads committed receipts only; no solve runs. Run on the login node.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402  (deliberately aligned with the import block)

from nova.media.ink import DEFAULT_INK, trace_axes  # noqa: E402

PARTS = Path("docs/figures/millisecond-converged-solve/scan-prototype/parts")
OUTPUT_ROOT = Path("docs/figures/millisecond-converged-solve/scan-prototype")

PRODUCTION_COLOR = DEFAULT_INK.flux_color
SCANNED_COLOR: str = DEFAULT_INK.thomson_secondary_color


def _residuals(platform: str, arm: str) -> "np.ndarray":
    payload = json.loads((PARTS / f"{platform}-{arm}-execution.json").read_text())
    values = np.asarray(
        [
            np.nan if value is None else value
            for value in payload["terminal"]["active_set_residuals"]
        ],
        dtype=float,
    )
    return values


def render(platform: str) -> dict:
    production = _residuals(platform, "production")
    scanned = _residuals(platform, "scanned")
    plt.style.use("data-ink")
    figure, axis = plt.subplots(figsize=(14, 6))
    trips = np.arange(production.size)
    axis.semilogy(
        trips,
        production,
        marker="o",
        markersize=6,
        linewidth=2.6,
        color=PRODUCTION_COLOR,
    )
    axis.semilogy(
        trips,
        scanned,
        marker="x",
        markersize=7,
        linewidth=2.6,
        linestyle="--",
        color=SCANNED_COLOR,
    )
    axis.set_xlabel("trip index")
    axis.set_ylabel("live residual (relative)")
    axis.set_xticks(trips)
    trace_axes(axis)
    finite = np.where(np.isfinite(production))[0]
    last = int(finite[-1]) if finite.size else 0
    axis.annotate(
        "production fori_loop",
        (last, float(production[last])),
        textcoords="offset points",
        xytext=(6, 4),
        ha="left",
        fontsize=DEFAULT_INK.label_fontsize,
        color=PRODUCTION_COLOR,
    )
    axis.annotate(
        "scanned trips",
        (last, float(scanned[last])),
        textcoords="offset points",
        xytext=(6, -14),
        ha="left",
        fontsize=DEFAULT_INK.label_fontsize,
        color=SCANNED_COLOR,
    )
    figure.tight_layout()
    output = OUTPUT_ROOT / f"per-trip-residual-{platform}.svg"
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, format="svg")
    plt.close(figure)
    mask = np.isfinite(production) & np.isfinite(scanned)
    return {
        "platform": platform,
        "scanned_minus_production_sup": float(
            np.max(np.abs(production[mask] - scanned[mask]))
        )
        if mask.any()
        else None,
        "terminal_residual": float(production[mask][-1]) if mask.any() else None,
        "figure": str(output),
    }


def main() -> int:
    summaries = {platform: render(platform) for platform in ("cpu", "cuda")}
    cpu = _residuals("cpu", "production")
    cuda = _residuals("cuda", "production")
    mask = np.isfinite(cpu) & np.isfinite(cuda)
    agreement = {
        "cpu_vs_cuda_max_abs_difference": float(np.max(np.abs(cpu[mask] - cuda[mask]))),
        "cpu_vs_cuda_max_relative_difference": float(
            np.max(np.abs(cpu[mask] - cuda[mask]) / np.abs(cpu[mask]))
        )
        if mask.any()
        else None,
        "bit_identical": bool(np.array_equal(cpu[mask], cuda[mask])),
        "cpu_terminal_residual": float(cpu[mask][-1]) if mask.any() else None,
        "cuda_terminal_residual": float(cuda[mask][-1]) if mask.any() else None,
    }
    print(
        json.dumps(
            {"per_platform": summaries, "agreement": agreement},
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
