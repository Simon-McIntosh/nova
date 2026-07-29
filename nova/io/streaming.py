"""Streaming seam for real-time-adjacent signal feeds (interface only).

This module declares the contract a streaming signal source must satisfy so
the spine can consume live-adjacent data with the same standard-name keying as
the batch store. It is a typed :class:`typing.Protocol` and carries no
implementation: the batch path (ingest/egress over zarr) is what ships now,
and a memory-backend ring buffer fed by an Access Layer ``put_slice`` loop is
the intended first concrete backend.

The design intent is that a ring-buffer backend exposes the same
standard-name-keyed, time-indexed view the batch :class:`xarray.Dataset` does,
so a consumer written against the store also drives off a live feed unchanged.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class SignalStream(Protocol):
    """A standard-name-keyed, append-only view over a streaming signal source.

    A backend keeps a bounded (ring-buffer) history per standard name. Slices
    arrive newest-last via :meth:`put_slice`; consumers read the retained
    window by standard name via :meth:`window` and discover availability via
    :meth:`names`. The window semantics mirror the batch store so a consumer
    written against :class:`xarray.Dataset` variables ports to the live feed
    without change.
    """

    @property
    def capacity(self) -> int:
        """Return the maximum number of time slices retained per signal."""
        ...

    def names(self) -> tuple[str, ...]:
        """Return the standard names currently available in the buffer."""
        ...

    def put_slice(self, time: float, values: dict[str, np.ndarray]) -> None:
        """Append one time slice keyed by standard name (evicting the oldest).

        Parameters
        ----------
        time:
            Time stamp of the slice, appended to the shared time window.
        values:
            Mapping of standard name to the slice's value for that signal.
        """
        ...

    def window(self, name: str) -> np.ndarray:
        """Return the retained history for ``name``, oldest-first."""
        ...

    def time(self) -> np.ndarray:
        """Return the retained time stamps, oldest-first."""
        ...
