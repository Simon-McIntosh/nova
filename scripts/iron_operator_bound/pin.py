"""Measure what the volume-rule repair does to the pinned shell operator.

The pin compares the default-argument operator of
:func:`nova.biot.tiledassembly.linear_galerkin_cell_interaction` with a stored
control at 1e-12.  This script reports the size of the change the repair makes
to that operator, and the quotient of the repaired operator's shell cells, so
the delta a control regeneration would absorb is a measured number.

    python scripts/iron_operator_bound/pin.py --output <dir>
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
from scipy.linalg import block_diag, eigh

from nova.jax.config import configure_dtypes

configure_dtypes()

from nova.biot import magnetisation as iron  # noqa: E402
from nova.biot import tiledassembly  # noqa: E402
from scripts.iron_linear_magnetisation.controls import shell  # noqa: E402


def repaired_rule(order):
    """Return the section rule the repair proposes."""
    return max(12, order // 2)


def quotient(interaction, gram):
    """Return the extreme metric-weighted quotients of one operator."""
    metric = block_diag(*gram, *gram)
    energy = metric @ np.asarray(interaction) / iron._MU0
    values = eigh((energy + energy.T) / 2, metric)[0]
    return float(values[0]), float(values[-1])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    root = Path(__file__).parents[2]
    control = json.loads(
        (
            root / "scripts" / "iron_linear_magnetisation" / "operator-control.json"
        ).read_text()
    )
    cells = iron._mesh(shell(), 0.04)
    assert len(cells.polygons) == control["shell_cells"] == 228
    polygons = tuple(cells.polygons[index] for index in control["selected_cells"])
    stored = np.asarray(control["operator"])
    delivered, blocks = iron.linear_galerkin_cell_interaction(polygons, plan=iron._PLAN)
    with patch.object(tiledassembly, "_volume_section_order", repaired_rule):
        repaired, repaired_blocks = iron.linear_galerkin_cell_interaction(
            polygons, plan=iron._PLAN
        )
    assert np.array_equal(blocks[0], repaired_blocks[0])
    scale = np.linalg.norm(stored)
    record = dict(
        rule_delivered=tiledassembly._volume_section_order(24),
        rule_repaired=repaired_rule(24),
        pin_relative_defect=float(
            np.linalg.norm(np.asarray(repaired) - stored) / scale
        ),
        repair_relative_change=float(
            np.linalg.norm(np.asarray(repaired) - np.asarray(delivered)) / scale
        ),
        delivered_relative_to_control=float(
            np.linalg.norm(np.asarray(delivered) - stored) / scale
        ),
        delivered_quotient=quotient(delivered, blocks),
        repaired_quotient=quotient(repaired, blocks),
    )
    (args.output / "pin.json").write_text(json.dumps(record, indent=1) + "\n")
    print(json.dumps(record), flush=True)


main()
