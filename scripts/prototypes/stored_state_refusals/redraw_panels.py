"""Redraw one run's published panels from the terminal fluxes it stored.

The panels an arm published under the harness painter drew the solved series on
the stored state's flux band, so a refused iterate crossed no level and the blue
series carried no drawn contour. This script rebuilds each panel from the stored
terminal flux with the shared level array in ``panels.py``, writes the panel's
record into the arm's own measurement, and cites the caption the fragment uses.

    redraw_panels.py <run directory>
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
for entry in (str(ROOT), str(HERE)):
    if entry not in sys.path:
        sys.path.insert(0, entry)

import numpy as np  # noqa: E402

import panels  # noqa: E402
import records  # noqa: E402


def _prepare(run: Path) -> None:
    """Point the case reader at this run's scratch and cached machine."""
    scratch = run / "redraw"
    scratch.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("NOVA_EFIT_COMPARISON_OUTPUT", str(scratch))
    os.environ.setdefault("NOVA_EFIT_COMPARISON_FIGURES", str(scratch))
    machine_cache = run / "machine-cache"
    if machine_cache.is_dir():
        os.environ.setdefault("XDG_DATA_HOME", str(machine_cache))


def redraw(run: Path) -> list[dict]:
    """Redraw every arm that stored a terminal flux, and record what was drawn."""
    _prepare(run)
    import pytest
    from nova.jax.config import configure_dtypes

    configure_dtypes()
    from tests.test_magnetics_fit_efit_comparison import _problem, _read_case

    monkeypatch = pytest.MonkeyPatch()
    merged = records.measurements(run)
    data_by_case: dict = {}
    drawn: list[dict] = []
    arms = [
        (label, arm)
        for label, arm in merged.items()
        if isinstance(arm, dict) and arm.get("terminal_flux")
    ]
    # An arm whose panel record already counted drawn nova contours is done, so
    # a re-submission after an expiry redraws only the arms that are missing
    # rather than paying every case's machine build again.
    pending = [
        (label, arm)
        for label, arm in arms
        if not (arm.get("panel_record") or {}).get("nova_levels_drawn")
    ]
    arms = pending or arms
    # The MAST-U machine is in the run's cache and the other two are rebuilt
    # here, so draw the cached case first: an expiry then loses the last case
    # rather than a case in the middle.
    order = {"mastu": 0, "tcv": 1, "iter": 2}
    arms.sort(key=lambda item: (order.get(item[1].get("case"), 9), item[0]))
    for label, arm in arms:
        case_name = arm.get("case")
        if case_name not in data_by_case:
            data_by_case[case_name] = _problem(_read_case(case_name), monkeypatch)
        data = data_by_case[case_name]
        destination = Path(
            str(arm.get("panel") or (run / f"{data['name']}-{label}.svg"))
        )
        flux = np.load(arm["terminal_flux"])["flux"]
        panel = panels.draw_panel(data, flux, destination, "")
        panel.update(
            label=label,
            residual=arm.get("residual"),
            reason=arm.get("termination_reason"),
            qualified=arm.get("qualified"),
        )
        panels.write_description(destination, panels.caption(panel))
        holder = records.owner(run, label)
        payload = json.loads(holder.read_text())
        payload[label]["panel"] = str(destination)
        payload[label]["panel_record"] = panel
        holder.write_text(json.dumps(payload, indent=1) + "\n")
        drawn.append(panel)
        print(
            "PANEL",
            str(destination),
            "imas_efit_levels_drawn=" + str(panel["imas_efit_levels_drawn"]),
            "nova_levels_drawn=" + str(panel["nova_levels_drawn"]),
            "nova_range=" + json.dumps(panel["nova_range"]),
            "nulls=" + json.dumps(panel["nulls"]["nova"]),
            flush=True,
        )
    return drawn


def main() -> None:
    redraw(Path(sys.argv[1]).resolve())


if __name__ == "__main__":
    main()
