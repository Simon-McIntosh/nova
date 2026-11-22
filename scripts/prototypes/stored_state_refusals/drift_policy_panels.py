"""Draw one poloidal flux panel per terminal state this record's arms reached.

Each panel contours the stored imas-efit state and the arm's terminal iterate on
one shared level array, draws the wall, and draws both null sets. A refused arm
writes no receipt, so its terminal state is read from the trajectory pickle the
solver dumped when it raised: a solve that stops at its first topology read
terminates on the state it read, and that is what the panel shows beside the
stored state.

    drift_policy_panels.py <run directory> <figure directory> \
        <machine cache> <scratch directory>

Run it through ``submit.sh`` here with ``--runner drift_policy_panels.py``; it
needs no GPU and writes each SVG beside a PNG preview for inspection.
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

ARMS = (
    (
        "tcv-input-stored",
        "tcv",
        "attribution-fixed/tcv-input-stored.json",
        "receipt",
        "attribution-fixed/tcv-input-stored-receipt.pkl",
    ),
    (
        "tcv-input-comparison",
        "tcv",
        "attribution-fixed/tcv-input-comparison.json",
        "receipt",
        "attribution-fixed/tcv-input-comparison-receipt.pkl",
    ),
    (
        "tcv-record-seed",
        "tcv",
        "record-seed/tcv-record-seed.json",
        "receipt",
        "record-seed/tcv-record-seed-receipt.pkl",
    ),
    (
        "iter-input-stored",
        "iter",
        "attribution-fixed/iter-input-stored.json",
        "receipt",
        "attribution-fixed/iter-input-stored-receipt.pkl",
    ),
    (
        "iter-input-comparison",
        "iter",
        "attribution-fixed/iter-input-comparison.json",
        "receipt",
        "attribution-fixed/iter-input-comparison-receipt.pkl",
    ),
    (
        "iter-record-seed",
        "iter",
        "iter-record-seed-json/iter-record-seed.json",
        "history",
        "iter-record-seed-json/iter-record-seed-rejected-history.pkl",
    ),
)


def _terminal_flux(source: str, path: Path) -> np.ndarray:
    with open(path, "rb") as handle:
        stored = pickle.load(handle)
    flux = stored.terminal_state.flux if source == "receipt" else stored.state
    return np.asarray(flux, dtype=float).ravel()


def _recorded_axis(arm: dict) -> list[float] | None:
    value = (arm.get("axis") or {}).get("magnetic_axis")
    return None if value is None else [float(item) for item in value]


def _solved_axis(data, flux):
    """Return the axis the terminal state's own topology read admits."""
    import jax

    try:
        _, topology = data["profile"].operator.read(jax.numpy.asarray(flux))
    except ValueError as error:
        return None, f"{type(error).__name__}: {error}"
    return [float(item) for item in np.asarray(topology.axis).ravel()[:2]], None


def draw(run: Path, destination_root: Path, machine_cache: Path, scratch: Path):
    scratch.mkdir(parents=True, exist_ok=True)
    destination_root.mkdir(parents=True, exist_ok=True)
    os.environ["NOVA_EFIT_COMPARISON_OUTPUT"] = str(scratch)
    os.environ["NOVA_EFIT_COMPARISON_FIGURES"] = str(scratch)
    os.environ["XDG_DATA_HOME"] = str(machine_cache)

    import pytest
    from nova.jax.config import configure_dtypes

    configure_dtypes()
    from tests.test_magnetics_fit_efit_comparison import _problem, _read_case

    monkeypatch = pytest.MonkeyPatch()
    data_by_case: dict = {}
    drawn: list[dict] = []
    for label, case_name, record_name, source, flux_name in ARMS:
        arm = json.loads((run / record_name).read_text())
        if case_name not in data_by_case:
            data_by_case[case_name] = _problem(_read_case(case_name), monkeypatch)
        data = data_by_case[case_name]
        flux = _terminal_flux(source, run / flux_name)
        node = np.asarray(data["machine"].node)
        # The state carries the node, wall and sample coordinates in that order,
        # so it is longer than the solve mesh and the panel slices it back.
        if flux.size < len(node):
            raise ValueError(
                f"{label}: {flux.size} terminal values carry {len(node)} nodes"
            )
        recorded = _recorded_axis(arm)
        drawn_axis, refusal = _solved_axis(data, flux)
        axis_matches = None
        if recorded is not None and drawn_axis is not None:
            axis_matches = bool(
                np.allclose(np.asarray(drawn_axis), np.asarray(recorded), atol=1.0e-9)
            )
        if recorded is not None and drawn_axis is None:
            raise ValueError(f"{label}: the recorded axis did not re-read: {refusal}")
        destination = destination_root / f"{label}.svg"
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


def main() -> None:
    run, destination_root, machine_cache, scratch = (
        Path(argument).resolve() for argument in sys.argv[1:5]
    )
    drawn = draw(run, destination_root, machine_cache, scratch)
    summary = dict(panels=drawn, source_revision=os.environ.get("SOURCE_REVISION"))
    (run / "panel-redraw.json").write_text(json.dumps(summary, indent=1) + "\n")
    print("DRIFT_POLICY_PANELS_FINISHED", flush=True)


if __name__ == "__main__":
    main()
