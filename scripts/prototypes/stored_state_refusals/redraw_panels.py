"""Redraw published panels from the terminal fluxes the arms stored.

The panels an arm published under the harness painter drew the solved series on
the stored state's flux band, so a refused iterate crossed no level and the blue
series carried no drawn contour. This script rebuilds each panel from the stored
terminal flux with the shared level array in ``panels.py``, writes the panel's
record into the arm's own measurement when the arm has a measurement receipt,
and cites the caption the fragment uses.

An arm stores its terminal flux in one of two shapes. A solved arm writes the
flux array; a refused arm writes no receipt, so the pickled fixed-point history
it rejected is the only record of the state it stopped on, and either the
receipt's ``terminal_state`` or the history's ``state`` carries it. Both are
read here, because a single drawing path is what keeps every published panel on
one level array and one caption vocabulary.

    redraw_panels.py <run directory>
"""

from __future__ import annotations

import json
import os
import pickle
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


def terminal_flux(arm: dict) -> tuple[np.ndarray, str]:
    """Return one arm's stored terminal flux and the shape it was stored in.

    ``terminal_flux`` names an npz whose ``flux`` array a solved arm wrote.
    ``terminal_pickle`` names a pickled receipt (``terminal_source: receipt``)
    or a pickled fixed-point history (``terminal_source: history``); a receipt
    carries the flux of the state its solve terminated on and a history carries
    the state the solve refused at.
    """
    path = arm.get("terminal_flux")
    if path:
        return np.load(path)["flux"], "npz"
    path = arm.get("terminal_pickle")
    if not path:
        raise ValueError(f"{arm.get('label')}: no stored terminal flux")
    with open(path, "rb") as handle:
        stored = pickle.load(handle)
    if arm.get("terminal_source") == "receipt":
        flux = stored.terminal_state.flux
    else:
        flux = stored.state
    return np.asarray(flux, dtype=float).ravel(), str(
        arm.get("terminal_source") or "history"
    )


def _solved_axis(data, flux):
    """Return the axis the terminal state's own topology read admits."""
    import jax

    try:
        _, topology = data["profile"].operator.read(jax.numpy.asarray(flux))
    except ValueError as error:
        return None, f"{type(error).__name__}: {error}"
    return [float(item) for item in np.asarray(topology.axis).ravel()[:2]], None


def draw_arms(run: Path, arms, destination_root: Path | None = None) -> list[dict]:
    """Draw one panel per arm, from the terminal flux that arm stored.

    Each arm names its ``case``, its stored flux and optionally the panel path,
    the residual, reason and qualification the caption states, and the
    ``recorded_axis`` its own record kept. A recorded axis is re-read from the
    terminal flux and compared with it to 1e-9, so a panel never presents a
    different stationary point than the record beside it.
    """
    _prepare(run)
    import pytest
    from nova.jax.config import configure_dtypes

    configure_dtypes()
    from tests.test_magnetics_fit_efit_comparison import _problem, _read_case

    monkeypatch = pytest.MonkeyPatch()
    data_by_case: dict = {}
    drawn: list[dict] = []
    for label, arm in arms:
        case_name = arm.get("case")
        if case_name not in data_by_case:
            data_by_case[case_name] = _problem(_read_case(case_name), monkeypatch)
        data = data_by_case[case_name]
        node = np.asarray(data["machine"].node)
        flux, source = terminal_flux(dict(arm, label=label))
        # A matched solution's state carries the node, wall and sample
        # coordinates in that order, so it is longer than the solve mesh and
        # the painter slices it back; a state shorter than the mesh cannot be
        # the solve's.
        if flux.size < len(node):
            raise ValueError(
                f"{label}: {flux.size} terminal values carry {len(node)} nodes"
            )
        if destination_root is not None:
            destination = destination_root / f"{label}.svg"
        else:
            destination = Path(
                arm.get("panel") or (run / f"{data['name']}-{label}.svg")
            )
        recorded = [float(item) for item in (arm.get("recorded_axis") or [])] or None
        drawn_axis, refusal = _solved_axis(data, flux)
        if recorded is not None and drawn_axis is None:
            raise ValueError(f"{label}: the recorded axis did not re-read: {refusal}")
        axis_matches = None
        if recorded is not None and drawn_axis is not None:
            axis_matches = bool(
                np.allclose(np.asarray(drawn_axis), np.asarray(recorded), atol=1.0e-9)
            )
            if not axis_matches:
                raise ValueError(
                    f"{label}: the panel's axis {drawn_axis} is not the record's "
                    f"{recorded}"
                )
        panel = panels.draw_panel(data, flux, destination, "")
        panel.update(
            label=label,
            residual=arm.get("residual"),
            reason=arm.get("termination_reason"),
            qualified=arm.get("qualified"),
            terminal_source=source,
            recorded_axis=recorded,
            drawn_axis=drawn_axis,
            axis_matches_record=axis_matches,
        )
        panels.write_description(destination, panels.caption(panel))
        holder = records.owner(run, label)
        if holder is not None:
            payload = json.loads(holder.read_text())
            payload[label]["panel"] = str(destination)
            payload[label]["panel_record"] = panel
            holder.write_text(json.dumps(payload, indent=1) + "\n")
        drawn.append(panel)
        print(
            "PANEL",
            str(destination),
            "levels_drawn=%s/%s"
            % (panel["imas_efit_levels_drawn"], panel["nova_levels_drawn"]),
            "nova_range=" + json.dumps(panel["nova_range"]),
            "nulls=" + json.dumps(panel["nulls"].get("nova")),
            "axis_matches=" + str(axis_matches),
            flush=True,
        )
    return drawn


def redraw(run: Path) -> list[dict]:
    """Redraw every arm that stored a terminal flux, and record what was drawn."""
    merged = records.measurements(run)
    arms = [
        (label, arm)
        for label, arm in merged.items()
        if isinstance(arm, dict)
        and (arm.get("terminal_flux") or arm.get("terminal_pickle"))
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
    return draw_arms(run, arms)


def main() -> None:
    redraw(Path(sys.argv[1]).resolve())


if __name__ == "__main__":
    main()
