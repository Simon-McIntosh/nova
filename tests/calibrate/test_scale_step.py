"""Manufactured scale histories recover the exact acquisition ladder."""

import math

import pytest

from nova.calibrate.scale_step import (
    ChannelScaleHistory,
    scale_blocks,
    scale_steps,
)
from nova.imas import mast_acquisition_scale
from nova.imas.mast_block_scale import channel_blocks


@pytest.mark.parametrize("rung", [0.5, 1 / math.sqrt(2), math.sqrt(2), 2.0])
def test_each_declared_scale_rung_is_recovered_exactly(rung):
    series = {index: [1.0 if index < 4 else rung] for index in range(8)}
    blocks = scale_blocks("probe", series)
    steps = scale_steps(blocks)

    assert len(blocks) == 2
    assert len(steps) == 1
    assert steps[0].rung == pytest.approx(rung, rel=1.0e-12)
    assert steps[0].ladder_distance == pytest.approx(0.0, abs=1.0e-12)
    assert steps[0].on_ladder


def test_mast_block_tables_consume_the_package_history_kernel():
    series = {index: [1.0 if index < 4 else 2.0] for index in range(8)}
    blocks = scale_blocks("probe", series)
    history = ChannelScaleHistory("probe", blocks, shot_count=8)

    table = channel_blocks(history, series)

    assert [row.rung for row in table] == [1.0, 2.0]
    assert mast_acquisition_scale.scale_blocks is scale_blocks
