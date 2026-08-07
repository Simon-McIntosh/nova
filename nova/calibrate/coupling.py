"""Regress what a channel reads that the description does not predict.

Subtract a described forward model from a measurement and what is left is a
residual field.  Regressing that residual on the currents that were flowing turns it
into a coupling: signed field per ampere of drive, on the channel's own axis, which
is a statement about the machine rather than about the pulse it was measured on.

Three things stand between the residual and a coupling that means anything, and each
is a guard here rather than a judgement left to the caller.

A drive that barely moved carries no information about its own coefficient, and
including its column adds a direction the fit must span with nothing to span it.
Columns are admitted on the span they actually swept, against both an absolute floor
and a share of the largest span in the same fit, so a pulse that drove one circuit
hard and another by a rounding error does not report a coefficient for the second.

Two circuits driven together cannot be told apart within one fit.  Where their
waveforms track each other the two columns are one column, and a solve handed both
returns a pair of numbers whose difference nothing constrained.  Such a pair is
merged into a single sum column and reported under a name that says so, which is an
honest coupling to the pair rather than a fabricated split between its members.

What survives merging can still be ill-conditioned, and a coefficient read off an
ill-conditioned design is a ratio of two nearly cancelling numbers.  The condition
number of the scaled design is measured before any coefficient is read, and a fit
past the limit is refused rather than reported with a caveat.

Pooling across fits is by median rather than mean.  Each fit is one pulse, the pulses
are few, and one pulse whose drive map is wrong should move the pooled number by one
vote and not by its magnitude.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from nova.calibrate.gain import baseline_offset

ACTIVE_FLOOR = 2000.0
"""Span a drive column must sweep before it is admitted, in its own units.

Its units are the drive's: ampere turns where the columns are ampere turns.  A floor
in absolute terms is what stops a fit reading a coefficient off a circuit that only
ever moved by its own noise, which no relative test can do on a pulse where nothing
was driven hard.
"""

ACTIVE_SHARE = 0.05
"""Share of the largest span in the same fit a column must also clear.

The absolute floor alone admits a column that is real but negligible beside the
drive under test, and such a column costs a degree of freedom to estimate something
the pulse cannot resolve.
"""

MERGE_CORRELATION = 0.98
"""Correlation above which two columns are one column.

Set where a pair driven in series stops being separable in practice rather than in
principle: below it the two waveforms differ by more than their own acquisition
scatter, above it the difference is scatter.
"""

CONDITION_LIMIT = 30.0
"""Condition number of the scaled design past which no coefficient is read.

Scaled means each column divided by its own standard deviation, so the number
measures how nearly the drive patterns are collinear and not how differently the
circuits are sized.
"""

MINIMUM_SAMPLES = 200
"""Samples a channel must contribute before its coefficient is estimated."""

MINIMUM_POOLED_FITS = 2
"""Fits a channel needs before a pooled coupling is reported for it.

One fit is one pulse, and a median over one value is that value with a claim of
robustness attached to it.
"""


class CouplingError(ValueError):
    """Raised when a coupling cannot be estimated from what was supplied."""


@dataclass(frozen=True)
class DriveBlock:
    """The regressor columns one fit reads, and how they came to be those columns.

    ``merged`` names the groups that were collapsed, so a coefficient reported
    against ``p4_pair`` can be traced to the members it stands for without consulting
    the caller's pair table.
    """

    names: tuple[str, ...]
    columns: np.ndarray
    condition: float
    merged: tuple[tuple[str, ...], ...] = ()
    dropped: tuple[str, ...] = ()

    @property
    def spans(self) -> np.ndarray:
        """Return the range each admitted column sweeps."""

        return np.ptp(self.columns, axis=0)

    @property
    def conditioned(self) -> bool:
        """Return whether the design separates its columns well enough to read."""

        return bool(np.isfinite(self.condition) and self.condition <= CONDITION_LIMIT)


@dataclass(frozen=True)
class CouplingFit:
    """One channel's signed coupling to every column of one drive block."""

    channel: str
    names: tuple[str, ...]
    coefficients: np.ndarray
    intercept: float
    variance_explained: float
    sample_count: int

    def coupling(self, name: str) -> float:
        """Return the signed coupling to one named drive."""

        if name not in self.names:
            raise CouplingError(
                f"{self.channel} was not fitted against {name!r}; the fit carries "
                f"{list(self.names)}"
            )
        return float(self.coefficients[self.names.index(name)])


def baseline_free_residual(
    signal: np.ndarray | Sequence[float],
    drive: np.ndarray,
    response: np.ndarray | Sequence[float],
    *,
    baseline_mask: np.ndarray | Sequence[bool] | None = None,
) -> np.ndarray:
    """Return what a channel read that its described response does not predict.

    ``response`` is the channel's row of the forward model: field per unit of each
    drive column, in the same order.  The standing offset is measured from the
    channel's own quiet window when one is given, because an integrator's zero is a
    property of the pulse and not of the machine.
    """

    values = np.asarray(signal, dtype=float)
    row = np.asarray(response, dtype=float)
    columns = np.asarray(drive, dtype=float)
    if columns.ndim != 2 or columns.shape[1] != row.size:
        raise CouplingError(
            f"a drive of shape {columns.shape} cannot be contracted with a response "
            f"of {row.size} entries"
        )
    if columns.shape[0] != values.size:
        raise CouplingError(
            f"{values.size} samples were given against {columns.shape[0]} drive rows"
        )
    offset = 0.0 if baseline_mask is None else baseline_offset(values, baseline_mask)
    return (values - offset) - columns @ row


def active_columns(
    drive: np.ndarray,
    *,
    mask: np.ndarray | Sequence[bool] | None = None,
    floor: float = ACTIVE_FLOOR,
    share: float = ACTIVE_SHARE,
) -> tuple[int, ...]:
    """Return the indices of the drive columns that swept enough to be fitted."""

    columns = np.asarray(drive, dtype=float)
    rows = columns if mask is None else columns[np.asarray(mask, dtype=bool)]
    if rows.size == 0:
        return ()
    spans = np.ptp(rows, axis=0)
    cutoff = max(float(floor), float(share) * float(spans.max()))
    return tuple(int(index) for index in np.flatnonzero(spans > cutoff))


def _correlation(first: np.ndarray, second: np.ndarray) -> float:
    """Return the correlation of two columns, zero where either does not vary."""

    if first.std() <= 0.0 or second.std() <= 0.0:
        return 0.0
    return float(np.corrcoef(first, second)[0, 1])


def build_drive_block(
    drive: np.ndarray,
    names: Sequence[str],
    *,
    mask: np.ndarray | Sequence[bool] | None = None,
    merge_groups: Mapping[tuple[str, ...], str] | None = None,
    floor: float = ACTIVE_FLOOR,
    share: float = ACTIVE_SHARE,
    merge_correlation: float = MERGE_CORRELATION,
) -> DriveBlock:
    """Admit the columns a fit can read, merging the ones it cannot separate.

    ``merge_groups`` maps a tuple of column names to the name their sum is reported
    under.  A group merges only when every member is active and every member tracks
    the first above ``merge_correlation``, so a pair driven independently on this
    pulse stays two columns and is fitted as two.
    """

    columns = np.asarray(drive, dtype=float)
    if columns.ndim != 2 or columns.shape[1] != len(names):
        raise CouplingError(
            f"a drive of shape {columns.shape} does not match {len(names)} names"
        )
    rows = (
        np.ones(columns.shape[0], dtype=bool)
        if mask is None
        else np.asarray(mask, dtype=bool)
    )
    active = active_columns(columns, mask=rows, floor=floor, share=share)
    active_names = [names[index] for index in active]
    lookup = {name: index for index, name in enumerate(names)}

    selected: list[tuple[str, np.ndarray]] = []
    merged: list[tuple[str, ...]] = []
    absorbed: set[str] = set()
    for members, label in (merge_groups or {}).items():
        if not all(member in active_names for member in members):
            continue
        block = [columns[:, lookup[member]] for member in members]
        first = block[0][rows]
        if not all(
            _correlation(first, other[rows]) > merge_correlation for other in block[1:]
        ):
            continue
        selected.append((label, np.sum(block, axis=0)))
        merged.append(tuple(members))
        absorbed.update(members)
    for name in active_names:
        if name not in absorbed:
            selected.append((name, columns[:, lookup[name]]))

    if not selected:
        return DriveBlock(
            names=(),
            columns=np.zeros((columns.shape[0], 0)),
            condition=np.inf,
            dropped=tuple(name for name in names if name not in active_names),
        )
    block = np.column_stack([column for _, column in selected])
    return DriveBlock(
        names=tuple(name for name, _ in selected),
        columns=block,
        condition=scaled_condition(block[rows]),
        merged=tuple(merged),
        dropped=tuple(name for name in names if name not in active_names),
    )


def scaled_condition(block: np.ndarray) -> float:
    """Return the condition number of a design with each column scaled to unit spread.

    Scaling first is what makes the number a statement about collinearity: an
    unscaled design of a kiloampere circuit beside a ten-ampere one is ill-conditioned
    by its units alone, and reading that as inseparability would refuse every fit that
    mixes circuit sizes.
    """

    columns = np.asarray(block, dtype=float)
    if columns.size == 0 or columns.shape[0] < 2:
        return np.inf
    scale = columns.std(axis=0)
    if np.any(scale <= 0.0):
        return np.inf
    return float(np.linalg.cond(columns / scale))


def joint_drive_fit(
    residual: np.ndarray | Sequence[float],
    block: DriveBlock,
    *,
    channel: str = "",
    mask: np.ndarray | Sequence[bool] | None = None,
    minimum_samples: int = MINIMUM_SAMPLES,
) -> CouplingFit | None:
    """Regress one channel's residual on every admitted drive at once.

    Jointly, never one drive at a time: a coefficient fitted against one column of a
    correlated set absorbs whatever the columns it omits would have taken, and the
    number that comes back is the residual's projection onto one drive rather than
    its coupling to that drive.

    Returns None where the channel does not clear ``minimum_samples`` after its own
    non-finite samples are removed, which is a channel this pulse cannot speak for
    rather than an error.
    """

    if not block.conditioned:
        raise CouplingError(
            f"the drive block conditions at {block.condition:.4g} against a limit of "
            f"{CONDITION_LIMIT}, so its columns are too nearly collinear for any "
            "coefficient read off them to be separated from its neighbours"
        )
    values = np.asarray(residual, dtype=float)
    rows = np.ones(values.size, dtype=bool) if mask is None else np.asarray(mask, bool)
    keep = rows & np.isfinite(values) & np.isfinite(block.columns).all(axis=1)
    if int(keep.sum()) < minimum_samples:
        return None
    y = values[keep]
    design = np.column_stack([np.ones(y.size), block.columns[keep]])
    coefficients, *_ = np.linalg.lstsq(design, y, rcond=None)
    fitted = design @ coefficients
    total = float(np.sum((y - y.mean()) ** 2))
    explained = (
        np.nan if total <= 0.0 else 1.0 - float(np.sum((y - fitted) ** 2)) / total
    )
    return CouplingFit(
        channel=channel,
        names=block.names,
        coefficients=np.asarray(coefficients[1:], dtype=float),
        intercept=float(coefficients[0]),
        variance_explained=float(explained),
        sample_count=int(keep.sum()),
    )


def pool_couplings(
    rows: Iterable[np.ndarray | Sequence[float]],
    *,
    minimum_fits: int = MINIMUM_POOLED_FITS,
) -> tuple[np.ndarray, np.ndarray]:
    """Pool per-fit coupling vectors into one median vector and its support count.

    Channels backed by fewer than ``minimum_fits`` finite entries come back as
    not-a-number rather than as the one value that happened to exist, so a consumer
    cannot read a single pulse as a pooled result.
    """

    block = np.asarray([np.asarray(row, dtype=float) for row in rows], dtype=float)
    if block.ndim != 2 or block.size == 0:
        raise CouplingError("pooling needs at least one coupling vector")
    counts = np.isfinite(block).sum(axis=0)
    pooled = np.full(block.shape[1], np.nan)
    supported = np.flatnonzero(counts >= max(1, minimum_fits))
    for column in supported:
        finite = block[np.isfinite(block[:, column]), column]
        pooled[column] = float(np.median(finite))
    return pooled, counts
