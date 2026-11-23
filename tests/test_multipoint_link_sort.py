"""Link ordering for multi-point constraints across the frame index.

The frame is re-sorted after every inserted element, so the sort must cost
time linear in the row count. These tests pin the ordering semantics against
a reference that walks the rows one at a time, bound the sort's cost at a
stated row count, and check the stored MAST-U case machine's frame columns.
"""

import numpy as np
import pytest

from nova.frame.columnar import Index
from nova.frame.framelink import FrameLink


def _reference_sort_link(frame):
    """Return a copy of the per-row link walk the linear sort replaces.

    Kept verbatim so the ordering the frame has always produced is the oracle
    the linear implementation is measured against.
    """
    for position, link in enumerate(frame.link):
        if link and link in frame.index:
            name = frame.index[position]
            link_index = frame.index.get_loc(link)
            if link_index > position:  # reverse
                frame.at[link, "link"] = name
                frame.loc[np.asarray(frame.link) == link, "link"] = name
                frame.at[name, "link"] = ""


def _frame(index, link):
    """Return a frame holding the given index and link column as delivered."""
    count = len(index)
    frame = FrameLink(
        {"x": range(count), "link": [""] * count},
        index=list(index),
        metadata={"Required": []},
    )
    frame.link = np.array(link, dtype=object)
    return frame


def _columns(index, link, reference):
    """Return index, link, ref and subref after sorting and reduction."""
    frame = _frame(index, link)
    if reference:
        _reference_sort_link(frame)
    else:
        frame.multipoint.sort_link()
    frame.multipoint.build()
    return (
        list(frame.index),
        list(frame.link),
        list(frame.ref),
        list(frame.subref),
    )


def _layout(shape, count, seed):
    """Return a link column with a named structure over count rows.

    The shapes cover the reversals the sort exists for: a forward chain, a
    star onto the last row, reversed chains, clustered jumps and mixed noise.
    """
    rng = np.random.default_rng(seed)
    index = [f"r{row}" for row in range(count)]
    link = [""] * count
    for row in range(count):
        if shape == "chain" and row + 1 < count:
            link[row] = index[row + 1]
        elif shape == "star":
            link[row] = index[-1] if rng.random() < 0.7 else index[rng.integers(count)]
        elif shape == "revchain" and row > 0:
            link[row] = index[row - 1]
        elif shape == "clusters" and rng.random() < 0.6:
            link[row] = index[min(count - 1, row + 1 + int(rng.integers(0, 4)))]
        elif shape == "mixed":
            draw = rng.random()
            if draw < 0.35:
                link[row] = index[rng.integers(count)]
            elif draw < 0.45:
                link[row] = "absent"
    return index, link


@pytest.mark.parametrize("shape", ["chain", "star", "revchain", "clusters", "mixed"])
@pytest.mark.parametrize("count", [1, 2, 3, 5, 9, 17, 40])
def test_sort_matches_reference_walk(shape, count):
    index, link = _layout(shape, count, seed=count * 31 + len(shape))
    assert _columns(index, link, reference=False) == _columns(
        index, link, reference=True
    )


def test_sort_random_layouts_match_reference_walk():
    rng = np.random.default_rng(20261122)
    for trial in range(120):
        count = int(rng.integers(0, 24))
        index = [f"row{row}" for row in range(count)]
        link = [
            str(rng.choice([*index, "absent", ""])) if count else ""
            for _ in range(count)
        ]
        assert _columns(index, link, reference=False) == _columns(
            index, link, reference=True
        ), f"trial {trial}"


def test_sort_points_every_link_at_an_earlier_row():
    index, link = _layout("chain", 64, seed=5)
    frame = _frame(index, link)
    frame.multipoint.sort_link()
    position = {label: row for row, label in enumerate(frame.index)}
    for row, label in enumerate(np.asarray(frame.link, dtype=object)):
        if label:
            assert position[label] < row


def test_sort_of_a_backward_linked_frame_is_a_no_op():
    index = [f"r{row}" for row in range(6)]
    link = ["", "r0", "r0", "r1", "r1", "r2"]
    frame = _frame(index, link)
    before = np.array(frame.link, dtype=object).copy()
    frame.multipoint.sort_link()
    assert np.array_equal(np.array(frame.link, dtype=object), before)


def test_link_sort_cost_is_linear_at_a_stated_row_count(monkeypatch):
    """A per-row label walk makes one get_loc call per row at this size."""
    count = 12000
    index, link = _layout("star", count, seed=11)
    frame = _frame(index, link)
    calls = {"count": 0}
    get_loc = Index.get_loc

    def counting_get_loc(self, label):
        calls["count"] += 0 if label is None else 1
        return get_loc(self, label)

    monkeypatch.setattr(Index, "get_loc", counting_get_loc)
    frame.multipoint.sort_link()
    assert calls["count"] <= count // 100, (
        f"{calls['count']} label scans for {count} rows: the sort walks every "
        "row instead of resolving the link column in one indexer pass"
    )


@pytest.mark.slow
def test_stored_case_machine_link_columns_match_reference_walk(monkeypatch, tmp_path):
    """The stored MAST-U case machine's frame columns are unchanged by the sort.

    The machine is built cold, element by element, so every insert re-sorts a
    frame that grows to the case's full row count. States from that build are
    replayed through the per-row reference walk and the columns compared, which
    is the before/after identity of the ordering at the size the build reaches.
    """
    from tests import test_equilibrium_forward_reference as owner
    from tests.test_magnetics_fit_efit_comparison import _read_case
    from nova.frame.multipoint import MultiPoint

    monkeypatch.setenv("NOVA_EFIT_COMPARISON_OUTPUT", str(tmp_path))
    case = _read_case("mastu")["case"]
    captured = []
    recorded = {"size": 0}
    sort = MultiPoint.sort_link

    def recording_sort(self):
        size = len(self.frame)
        if size >= 4000 and size - recorded["size"] >= 2000:
            recorded["size"] = size
            captured.append(
                (
                    list(np.asarray(self.frame.index, dtype=object)),
                    list(np.asarray(self.frame.link, dtype=object)),
                    size,
                )
            )
        return sort(self)

    monkeypatch.setattr(MultiPoint, "sort_link", recording_sort)
    owner.build_machine(case, -300, passive=bool(case.passive))
    monkeypatch.setattr(MultiPoint, "sort_link", sort)
    assert len(captured) >= 3, f"only {len(captured)} frame states captured"
    assert max(size for _, _, size in captured) > 25000
    for index, link, size in captured:
        assert _columns(index, link, reference=False) == _columns(
            index, link, reference=True
        ), f"frame columns differ from the reference walk at {size} rows"
