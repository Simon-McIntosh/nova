"""Guard the certificate identity comparator against a vacuous comparison.

The comparator reads a row as within drift from three reductions over the two
terminal states: bit equality, a maximum absolute difference and the terminal
residuals.  All three are vacuous when the identity set is empty -- two empty
arrays compare equal, a maximum over an empty difference defaults to zero, and
the residual of nothing is exactly zero -- so an empty set is indistinguishable
from a machine-precision match.
"""

import numpy as np
import pytest

from benchmarks.solve_program_size_gate import (
    CERTIFICATE_ROWS,
    EmptyIdentitySetError,
    _require_identity_rows,
    run_certificate_identity,
)


def test_empty_identity_set_would_read_as_within_drift():
    """The defect the guard closes: an empty comparison is a passing comparison."""
    empty = np.zeros(0, dtype=np.float64)
    assert np.array_equal(empty, empty)
    assert float(np.max(np.abs(empty - empty), initial=0.0)) == 0.0


def test_identity_row_count_states_the_denominator():
    with pytest.raises(EmptyIdentitySetError):
        _require_identity_rows(0, "solovev:diverted-single-null:-110")
    assert _require_identity_rows(3441, "solovev:diverted-single-null:-110") == 3441


def test_empty_set_refused_then_nonempty_set_accepted():
    """One comparator call refuses the empty set and accepts the stated one."""
    empty_set: tuple[tuple[str, int], ...] = ()
    with pytest.raises(EmptyIdentitySetError) as refusal:
        _require_identity_rows(len(empty_set), "certificate identity rows")
    assert "certificate identity rows" in str(refusal.value)

    nonempty_set = (("diverted-single-null", -500),)
    assert _require_identity_rows(len(nonempty_set), "certificate identity rows") == 1


def test_driver_refuses_an_empty_certificate_before_the_allocation_check(
    monkeypatch, tmp_path
):
    """An empty configured identity set is refused by name, not by the allocator."""
    monkeypatch.setattr(
        "benchmarks.solve_program_size_gate.CERTIFICATE_ROWS", (), raising=True
    )
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    with pytest.raises(EmptyIdentitySetError):
        run_certificate_identity(tmp_path / "receipt.json", None)


def test_committed_certificate_states_a_nonempty_identity_set():
    assert len(CERTIFICATE_ROWS) == 4
    assert all(
        isinstance(case, str) and isinstance(cells, int) and cells != 0
        for case, cells in CERTIFICATE_ROWS
    )
