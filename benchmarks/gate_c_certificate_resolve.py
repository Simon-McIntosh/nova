"""Re-solve the Solovev certificate rows under the gate-C resolve figure root.

Thin wrapper around ``benchmarks.rung_a_certificate_resolve``: the committed
driver owns the per-row fresh-process measure, the in-allocation sweep and
the comparison table against the committed aggregate receipt.  This module
only redirects the driver's output roots to
``docs/figures/cut-cell-current-attribution/gate-c-resolve`` with the
matching report and run directories, then re-labels each landed row's figure
citation to that root, so the sibling ``gate-c-certificate`` re-solve's
artifacts are never touched.

All three driver modes (``--row``, ``--sweep``, ``--table``) are forwarded
unchanged.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import sys

_WORKTREE = Path(__file__).resolve().parents[1]
_FIGURE_ROOT = _WORKTREE / "docs/figures/cut-cell-current-attribution/gate-c-resolve"
_REPORT_ROOT = Path(
    "/home/ITER/mcintos/.config/reckon/crew/reports/nova/s19-review/gate-c-resolve"
)

# Redirect before the driver is imported: the driver resolves its roots from
# these variables at import time.  The run root is supplied by the launch
# script (sbatch --export) and only defaulted here, so a manual run still
# avoids writing sweep state into the sibling run's directory.
os.environ["GATE_C_CERTIFICATE_FIGURE_ROOT"] = str(_FIGURE_ROOT)
os.environ["GATE_C_CERTIFICATE_REPORT_ROOT"] = str(_REPORT_ROOT)
os.environ.setdefault(
    "GATE_C_CERTIFICATE_RUN_ROOT",
    "/home/ITER/mcintos/.config/reckon/crew/runs/"
    "r-20260911T044629802980-cca-gate-c-certificate-resolve-2",
)

import benchmarks.rung_a_certificate_resolve as _driver  # noqa: E402


def _relabel_figure_srcs() -> None:
    """Point every row figure citation at this driver's redirected figure root.

    The committed driver hard-codes the ``gate-c-certificate`` src label on
    each landed row's figure; with the output root redirected that label would
    cite the sibling re-solve's panels instead of those this driver wrote, so
    restate it before the comparison table is built from the row stems.
    """

    def relabel(payload: dict) -> bool:
        figure = payload.get("figure")
        if not isinstance(figure, dict):
            return False
        src = figure.get("project_absolute_src")
        if not isinstance(src, str) or "gate-c-certificate" not in src:
            return False
        figure["project_absolute_src"] = src.replace(
            "gate-c-certificate", "gate-c-resolve"
        )
        return True

    for root in (_driver.PART_ROOT, _driver.SCALAR_ROOT):
        for stem in root.glob("*.json"):
            payload = json.loads(stem.read_text(encoding="utf-8"))
            if relabel(payload):
                stem.write_text(
                    json.dumps(payload, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8",
                )


def main() -> None:
    invocation = list(sys.argv[1:])
    sys.argv[0] = __file__
    _driver.main()
    _relabel_figure_srcs()
    if "--sweep" in invocation:
        _driver._table()


if __name__ == "__main__":
    main()
