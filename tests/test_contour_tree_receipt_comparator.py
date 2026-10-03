"""Field-by-field checks for the contour-tree receipt comparator.

Two receipt files written by ``benchmarks.contour_tree_brute_force`` compare
equal only when every receipt field matches. A receipt differing in exactly one
field must fail naming that field, once for each compared field, so a
comparator that silently drops a field is caught by the field it dropped. The
comparison takes its field set as an explicit argument defaulting to every
field; a test may narrow that set only to show the negative control, and the
CLI never does.

The declared negative control mutates the default field set to omit the
edge-mask field; the ``edge_valid`` case below then fails on receipts that
differ only there.
"""

from __future__ import annotations

import copy
from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np
import pytest

from benchmarks.contour_tree_brute_force import (
    RECEIPT_FIELDS,
    compare_receipts,
    main,
    receipt_from_tree,
    write_receipt_file,
)
from nova.equilibrium.contour_tree import build_contour_tree
from nova.jax.config import configure_dtypes


configure_dtypes()


def _receipt() -> dict[str, object]:
    """One small synthetic receipt with a populated slot in every field."""

    tree = SimpleNamespace(
        node_vertex=np.array([0, 1, 2, -1], dtype=np.int64),
        node_psi=np.array([1.5, 0.5, -1.0, 0.0], dtype=np.float64),
        node_valid=np.array([True, True, True, False]),
        critical_type=np.array([2, 1, 0, 3], dtype=np.int64),
        edges=np.array([[0, 1], [1, 2], [-1, -1]], dtype=np.int64),
        edge_valid=np.array([True, True, False]),
        overflow=np.asarray(False),
    )
    return {"fixture": receipt_from_tree(tree)}


def _altered(receipt: dict[str, object], field: str) -> dict[str, object]:
    """Copy a receipt and change exactly its named field."""

    changed = copy.deepcopy(receipt)
    value = changed["fixture"][field]
    if isinstance(value, list) and value and isinstance(value[0], list):
        rows = [list(row) for row in value]
        rows[0][0] = rows[0][0] + 1
        changed["fixture"][field] = rows
    elif isinstance(value, list):
        entries = list(value)
        entries[0] = not entries[0] if isinstance(entries[0], bool) else entries[0] + 1
        changed["fixture"][field] = entries
    else:
        changed["fixture"][field] = not value
    return changed


def test_receipt_from_tree_carries_every_field():
    """The serialised receipt exposes exactly the compared fields."""

    assert tuple(sorted(_receipt()["fixture"])) == tuple(sorted(RECEIPT_FIELDS))


@pytest.mark.parametrize("field", RECEIPT_FIELDS)
def test_identical_receipts_pass(tmp_path, field):
    """Receipts differing in no field compare equal with a zero exit."""

    base = _receipt()
    base_path = write_receipt_file(base, tmp_path / "base.json")
    head_path = write_receipt_file(copy.deepcopy(base), tmp_path / "head.json")
    assert main(["--compare", str(base_path), str(head_path)]) == 0
    assert all(item.mismatches == 0 for item in compare_receipts(base, base))


@pytest.mark.parametrize("field", RECEIPT_FIELDS)
def test_single_field_difference_names_that_field(tmp_path, field, capsys):
    """A one-field difference fails, and names only that field."""

    base = _receipt()
    head = _altered(base, field)
    base_path = write_receipt_file(base, tmp_path / "base.json")
    head_path = write_receipt_file(head, tmp_path / "head.json")
    code = main(["--compare", str(base_path), str(head_path)])
    printed = capsys.readouterr().out
    mismatched = {
        item.field for item in compare_receipts(base, head) if item.mismatches
    }
    assert mismatched == {field}
    assert code != 0
    assert f"{field}:" in printed


def test_narrowed_field_set_hides_its_omitted_field():
    """A narrowed field set cannot see a difference outside it.

    This is the mechanism the negative control relies on: only a comparison
    told to drop the edge-mask field misses an edge-mask-only difference.
    """

    base = _receipt()
    head = _altered(base, "edge_valid")
    reduced = tuple(name for name in RECEIPT_FIELDS if name != "edge_valid")
    assert all(
        item.mismatches == 0 for item in compare_receipts(base, head, fields=reduced)
    )
    assert any(item.mismatches for item in compare_receipts(base, head))


def test_real_tree_receipt_round_trips_through_writer_and_loader(tmp_path, capsys):
    """A tree built by the solver round-trips through the receipt files.

    Identical receipts compare equal with a zero exit; a receipt differing in
    one field exits nonzero and names that field.
    """

    tree = build_contour_tree(
        jnp.asarray([3.0, 1.0, 2.0, 0.5, 1.5, -1.0], dtype=jnp.float64),
        jnp.ones(6, dtype=bool),
        jnp.zeros(6, dtype=bool),
        jnp.asarray([[0, 1], [1, 2], [2, 3], [3, 4], [4, 5]], dtype=jnp.int32),
        jnp.ones(5, dtype=bool),
        jnp.asarray(1, dtype=jnp.int32),
    )
    receipt = receipt_from_tree(tree)
    base = write_receipt_file({"fixture": receipt}, tmp_path / "base.json")
    head = write_receipt_file(
        {"fixture": copy.deepcopy(receipt)}, tmp_path / "head.json"
    )
    assert main(["--compare", str(base), str(head)]) == 0

    field = "critical_type"
    bad = write_receipt_file(
        _altered({"fixture": receipt}, field), tmp_path / "bad.json"
    )
    code = main(["--compare", str(base), str(bad)])
    printed = capsys.readouterr().out
    assert code == 1
    assert f"{field}:" in printed
