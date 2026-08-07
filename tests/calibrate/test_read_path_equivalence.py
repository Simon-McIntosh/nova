"""Whether serving corrections from the document reads the same as the table did.

Moving a correction from a table beside the read path into a versioned document is
worth nothing if it moves a number, and the ways it could are quiet ones: a rung
transcribed to fewer digits, an interval that lost an endpoint, a refused block that
started dividing by the ratio it refused.  None of those would fail a unit test of
either side on its own, because each side is self-consistent.

So this compares the two readers against each other, on the same reads, and asserts
equality rather than closeness.  ``data/banked_block_scales.json`` is the table the
read path served before the document existed, frozen here where no production code
can reach it; the document is what serves those reads now.  A rung is a power of two
or its square root and an interval is a shot number, so every quantity compared here
is exactly representable and there is no tolerance to choose.  A test that admitted
one would be admitting the drift it exists to catch.

The reads cover every channel the table carries at both endpoints of every block, at
a shot inside each block, on both sides of every boundary between blocks, in the
middle of each gap, and outside the archive at either end -- because the disagreements
worth fearing live at edges, and a comparison sampling block interiors alone would
pass while the boundaries moved.

What the document does NOT change is asserted too.  It carries five promoted sensor
gains that this read path has never divided out, along with pickup states and
exclusions, and a cutover that started applying any of them would rescale every fit
downstream while looking like a change of storage.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from nova.calibrate.correction_model import CorrectionKind
from nova.calibrate.correction_set import read_correction_set
from nova.calibrate.corrections import build_chain
from nova.imas.mast_block_scale import (
    BRACKETED,
    MEASURED,
    REFUSED,
    UNMEASURED,
    BlockScaleTable,
    CorrectionSetScales,
    promoted_block_scales,
)
from nova.imas.mast_corrections import (
    acquisition_off_ladder_channels,
    acquisition_stepping_channels,
    promoted_channel_scales,
    withheld_channel_scales,
)

BANK = Path(__file__).parent / "data" / "banked_block_scales.json"
"""The block table the read path served before the correction document existed."""

BLOCK_COUNT = 113
"""Blocks the table carries, over 76 channels."""

REFUSED_BLOCKS = {
    "obr18": 0.6836652336349689,
    "obv03": 0.5182984917648246,
    "obv12": 0.4383580284135138,
    "obv18": 1.3733948295617209,
    "obv19": 1.6003469492828493,
}
"""The five blocks whose step is not a ladder rung, and the ratio each measured."""

RETIRED_GAINS = {
    "obr17": 0.5011,
    "obv04": 0.8571,
    "obr05": 1.1043,
    "obv05": 0.9449,
    "ccbv35": 0.9474,
}
"""The promoted scales as the record module carried them before the document did."""

RETIRED_WITHHELD = ("obv11",)
"""The refused scale as that module carried it."""

RETIRED_STEPPING = 19
"""Channels it recorded as stepping, and the count the document must still give."""

RETIRED_OFF_LADDER = 5
"""Blocks it recorded as off the ladder, likewise."""

PROMOTED_GAINS = tuple(RETIRED_GAINS)
"""Channels the document carries a promoted sensor gain for."""

PAIR_STATE_CHANNELS = ("obv03", "obr05")
"""Channels the document records a pickup state for."""

EXCLUDED_CHANNELS = ("obv10", "obv11")
"""Channels the document records an exclusion for."""


@pytest.fixture(scope="module")
def banked():
    """Return the reader the archive was served by before the cutover."""

    return BlockScaleTable.from_dict(json.loads(BANK.read_text()))


@pytest.fixture(scope="module")
def served():
    """Return the reader that serves those reads from the correction document."""

    return CorrectionSetScales.create(read_correction_set("mast", "magnetics"))


def probe_pulses(blocks) -> tuple[int, ...]:
    """Return the shots to compare a channel on, edges before interiors."""

    pulses: set[int] = set()
    for block in blocks:
        pulses.update(block.shots)
        pulses.update({block.first_shot, block.last_shot})
        pulses.add((block.first_shot + block.last_shot) // 2)
    for first, second in zip(blocks, blocks[1:], strict=False):
        pulses.update({first.last_shot + 1, second.first_shot - 1})
        pulses.add((first.last_shot + second.first_shot) // 2)
    pulses.update({blocks[0].first_shot - 1, blocks[-1].last_shot + 1})
    return tuple(sorted(pulses))


def inside_pulses(block) -> tuple[int, ...]:
    """Return the shots one block holds over, endpoints and interior alike."""

    pulses = set(block.shots)
    pulses.update({block.first_shot, block.last_shot})
    pulses.add((block.first_shot + block.last_shot) // 2)
    return tuple(sorted(pulses))


def identical(left: float, right: float) -> bool:
    """Return whether two readings are the same number, not merely close ones.

    A rung the document refused is carried as a value that is not a number on both
    sides, and two such readings agree: what is being compared is whether each reader
    declines in the same place, which is a fact about the block and not about
    arithmetic.  Every other comparison here is a plain float equality.
    """

    return left == right or (math.isnan(left) and math.isnan(right))


def agree(banked_read, served_read) -> bool:
    """Return whether two readers answered one read the same way."""

    return (
        banked_read.channel == served_read.channel
        and banked_read.shot == served_read.shot
        and identical(float(banked_read.scale), float(served_read.scale))
        and banked_read.disposition == served_read.disposition
        and len(banked_read.candidates) == len(served_read.candidates)
        and all(
            identical(float(left), float(right))
            for left, right in zip(
                banked_read.candidates, served_read.candidates, strict=False
            )
        )
    )


def test_the_bank_is_the_table_the_read_path_served(banked):
    """The comparison is worthless if the bank drifted from what it records."""

    blocks = [block for channel in banked.channels for block in banked.blocks[channel]]
    assert len(blocks) == BLOCK_COUNT
    assert len(banked.channels) == 76
    assert len(banked.stepping) == 19
    off_ladder = {block.channel: block.scale for block in blocks if not block.on_ladder}
    assert off_ladder == REFUSED_BLOCKS


def test_the_read_path_serves_the_reader_this_bench_compares(banked):
    """Without this the bench could pass while the read path served something else."""

    promoted = promoted_block_scales()
    assert isinstance(promoted, CorrectionSetScales)
    assert promoted.channels == banked.channels
    for channel in banked.channels:
        for pulse in probe_pulses(banked.blocks[channel]):
            assert agree(
                banked.correction(channel, pulse), promoted.correction(channel, pulse)
            )


def test_both_readers_carry_the_same_channels(banked, served):
    assert served.channels == banked.channels
    assert served.stepping == banked.stepping
    assert served.corrected == banked.corrected


def test_every_channel_reads_the_same_on_every_block_edge(banked, served):
    """The gate: same factor, same warrant, same candidates, to float equality."""

    disagreements = []
    reads = 0
    for channel in banked.channels:
        for pulse in probe_pulses(banked.blocks[channel]):
            reads += 1
            first = banked.correction(channel, pulse)
            second = served.correction(channel, pulse)
            if not agree(first, second):
                disagreements.append((channel, pulse, first, second))
    assert not disagreements
    assert reads > 3000


def test_the_reads_reach_every_block_and_every_boundary(banked):
    """Guard the gate above against covering less than it claims to."""

    covered = 0
    boundaries = 0
    for channel in banked.channels:
        blocks = banked.blocks[channel]
        pulses = set(probe_pulses(blocks))
        for block in blocks:
            assert {block.first_shot, block.last_shot} <= pulses
            covered += 1
        for first, second in zip(blocks, blocks[1:], strict=False):
            assert {first.last_shot + 1, second.first_shot - 1} <= pulses
            boundaries += 1
    assert covered == BLOCK_COUNT
    assert boundaries == BLOCK_COUNT - len(banked.channels)


def test_a_read_before_and_after_the_archive_is_unmeasured(banked, served):
    for channel in banked.channels:
        blocks = banked.blocks[channel]
        for pulse in (blocks[0].first_shot - 1, blocks[-1].last_shot + 1):
            assert agree(
                banked.correction(channel, pulse), served.correction(channel, pulse)
            )
            assert served.correction(channel, pulse).disposition == UNMEASURED


def test_a_channel_neither_reader_measured_reads_unmeasured(banked, served):
    assert "ccbv99" not in served.channels
    assert agree(banked.correction("ccbv99", 14061), served.correction("ccbv99", 14061))
    assert served.correction("ccbv99", 14061).disposition == UNMEASURED


# --- the five blocks whose step is not a rung ----------------------------


@pytest.mark.parametrize("channel", sorted(REFUSED_BLOCKS))
def test_a_refused_block_still_reads_as_published(banked, served, channel):
    """A refusal must survive the cutover as a refusal, not become a division."""

    blocks = [block for block in banked.blocks[channel] if not block.on_ladder]
    assert len(blocks) == 1
    for pulse in inside_pulses(blocks[0]):
        first = banked.correction(channel, pulse)
        second = served.correction(channel, pulse)
        assert agree(first, second)
        assert second.disposition == REFUSED
        assert second.scale == 1.0
        assert not second.applied
        assert second.flagged
        assert second.candidates == (REFUSED_BLOCKS[channel],)


def test_a_refused_block_divides_no_samples(served):
    """The refusal is only worth having if the array comes back untouched."""

    samples = np.asarray([1.0, 2.0, 4.0])
    values, rows = served.normalise(19219, {"obv12": samples})
    assert values["obv12"].tolist() == samples.tolist()
    assert rows[0].disposition == REFUSED


# --- what the document must not start doing ------------------------------


def test_the_read_path_divides_by_no_promoted_gain(served):
    """The gains are in the document and have never been on this read path.

    Drawing the whole chain instead of the acquisition stage would divide five
    channels by a further factor here, which is a change to every downstream
    amplitude and not a change of storage.  Whether to make it is a decision with
    its own evidence; making it as a side effect of moving a file is not.
    """

    gains = {
        row.channel: row.value
        for row in served.document.corrections
        if CorrectionKind(row.kind) is CorrectionKind.gain and row.value is not None
    }
    assert set(gains) == set(PROMOTED_GAINS)
    for channel in PROMOTED_GAINS:
        whole = build_chain(served.document, channel, pulse=14100)
        drawn = build_chain(
            served.document,
            channel,
            pulse=14100,
            kinds=(CorrectionKind.acquisition_scale,),
        )
        assert whole.multiplier == pytest.approx(drawn.multiplier * gains[channel])
        assert served.correction(channel, 14100).scale == drawn.multiplier


def test_the_document_serves_the_numbers_the_record_module_carried(served):
    """The four quantities that were literals in nova.imas, now read from here.

    Float equality on the gains: each is the four-decimal constant the adjudication
    promoted, and the document carries the unrounded ratio separately, so a reader
    that started serving the ratio would move five channels in the fourth decimal
    without failing anything that checks the document alone.
    """

    assert promoted_channel_scales() == RETIRED_GAINS
    assert withheld_channel_scales() == RETIRED_WITHHELD
    assert acquisition_stepping_channels() == RETIRED_STEPPING
    assert acquisition_off_ladder_channels() == RETIRED_OFF_LADDER
    assert acquisition_stepping_channels() == len(served.stepping)


def test_obr17_reads_its_rung_and_not_its_half(banked, served):
    """The lifetime factor of two is recorded, and this path leaves it in place."""

    for pulse in probe_pulses(banked.blocks["obr17"]):
        assert agree(
            banked.correction("obr17", pulse), served.correction("obr17", pulse)
        )
        assert served.correction("obr17", pulse).scale != 0.5011


@pytest.mark.parametrize("channel", PAIR_STATE_CHANNELS)
def test_a_recorded_pair_state_changes_no_read(banked, served, channel):
    """obr05 resolves to no single value at all, so applying it would refuse."""

    for pulse in probe_pulses(banked.blocks[channel]):
        assert agree(
            banked.correction(channel, pulse), served.correction(channel, pulse)
        )


@pytest.mark.parametrize("channel", EXCLUDED_CHANNELS)
def test_an_excluded_channel_reads_as_it_did(banked, served, channel):
    """The exclusion is recorded rather than promoted, so no read refuses on it."""

    blocks = banked.blocks.get(channel, ())
    pulses = probe_pulses(blocks) if blocks else (14100,)
    for pulse in pulses:
        assert agree(
            banked.correction(channel, pulse), served.correction(channel, pulse)
        )


# --- the division the readers exist to perform ---------------------------


def test_a_doubled_block_halves_the_same_samples_on_both_paths(banked, served):
    """Equality on the arrays, not only on the factor the readers report."""

    doubled = [
        (channel, block)
        for channel in banked.channels
        for block in banked.blocks[channel]
        if block.on_ladder and block.rung == 2.0
    ]
    assert doubled
    samples = np.asarray([1.0, -2.5, 4.0])
    for channel, block in doubled:
        pulse = block.first_shot
        first = banked.normalise(pulse, {channel: samples})[0][channel]
        second = served.normalise(pulse, {channel: samples})[0][channel]
        assert second.tolist() == first.tolist()
        assert second.tolist() == (samples / 2.0).tolist()


def test_one_shot_corrects_its_channels_independently(banked, served):
    """The whole array of a real shot, through both readers at once."""

    pulse = 14099
    probes = {channel: np.asarray([1.0, 2.0, 3.0]) for channel in banked.channels}
    first, first_rows = banked.normalise(pulse, probes)
    second, second_rows = served.normalise(pulse, probes)
    assert [row.disposition for row in second_rows] == [
        row.disposition for row in first_rows
    ]
    assert {row.disposition for row in second_rows} >= {MEASURED, BRACKETED}
    for channel in probes:
        assert second[channel].tolist() == first[channel].tolist()
