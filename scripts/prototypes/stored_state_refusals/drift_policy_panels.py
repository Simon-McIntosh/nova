"""Draw one published panel per terminal state this record's arms reached.

The drawing, the caption and the description are ``redraw_panels.draw_arms``'s,
which is the one path every published panel goes through; this file only names
the arms and where each one stored its terminal state. Five arms wrote a receipt,
whose ``terminal_state`` carries the flux its solve stopped on; the ITER refusal
wrote no receipt, so its state is the pickled fixed-point history the solver
rejected.

    drift_policy_panels.py <run directory> <figure directory> <scratch directory>

Writes ``<run directory>/panel-redraw.json`` with each drawing's record.
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

import redraw_panels  # noqa: E402

# (label, case, record, terminal source, stored flux)
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


def arm_specs(run: Path) -> list[tuple[str, dict]]:
    """Return one drawing specification per arm, from the arm's own record."""
    specs = []
    for label, case_name, record_name, source, flux_name in ARMS:
        record = json.loads((run / record_name).read_text())
        axis = (record.get("axis") or {}).get("magnetic_axis")
        specs.append(
            dict(
                case=case_name,
                terminal_pickle=str(run / flux_name),
                terminal_source=source,
                residual=record.get("residual"),
                termination_reason=record.get("termination_reason"),
                qualified=record.get("qualified"),
                recorded_axis=axis,
            )
        )
    return list(zip([arm[0] for arm in ARMS], specs, strict=True))


def main() -> None:
    run, destination_root, scratch = (
        Path(argument).resolve() for argument in sys.argv[1:4]
    )
    os.environ["NOVA_EFIT_COMPARISON_OUTPUT"] = str(scratch)
    os.environ["NOVA_EFIT_COMPARISON_FIGURES"] = str(scratch)
    drawn = redraw_panels.draw_arms(run, arm_specs(run), destination_root)
    (run / "panel-redraw.json").write_text(
        json.dumps(
            dict(panels=drawn, source_revision=os.environ.get("SOURCE_REVISION")),
            indent=1,
        )
        + "\n"
    )
    print("DRIFT_POLICY_PANELS_FINISHED", flush=True)


if __name__ == "__main__":
    main()
