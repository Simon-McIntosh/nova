"""Frame with fast array accessors.

``DataArray`` adds the label / positional accessors (``loc``, ``iloc``,
``at``, ``iat``) and the array accessor (``aloc``) to the columnar
:class:`~nova.frame.dataframe.DataFrame`. Because the whole frame is now a
struct-of-arrays, the array group no longer needs a shadow cache: every
accessor reads and writes the single store, so an attribute or item access
returns a live ndarray view.
"""

from __future__ import annotations

from functools import cached_property

from nova.frame.arraylocindexer import ArrayLocIndexer
from nova.frame.dataframe import DataFrame
from nova.frame.indexer import (
    LabelAccessor,
    PositionAccessor,
    ScalarLabelAccessor,
    ScalarPositionAccessor,
)

# pylint: disable=too-many-ancestors


class DataArray(DataFrame):
    """Columnar frame exposing label, positional and array accessors."""

    @cached_property
    def loc(self) -> LabelAccessor:
        """Return the label accessor."""
        return LabelAccessor(self)

    @cached_property
    def iloc(self) -> PositionAccessor:
        """Return the integer-position accessor."""
        return PositionAccessor(self)

    @cached_property
    def at(self) -> ScalarLabelAccessor:
        """Return the single-element label accessor."""
        return ScalarLabelAccessor(self)

    @cached_property
    def iat(self) -> ScalarPositionAccessor:
        """Return the single-element integer accessor."""
        return ScalarPositionAccessor(self)

    @cached_property
    def aloc(self) -> ArrayLocIndexer:
        """Return the live array-column accessor."""
        return ArrayLocIndexer("array", self)

    def _clear_accessors(self):
        """Drop cached accessors so they rebind after a store rebuild."""
        for name in ("loc", "iloc", "at", "iat", "aloc"):
            self.__dict__.pop(name, None)

    def update_frame(self):
        """No-op: the array group shares the single columnar store."""

    def unlink_array(self):
        """No-op retained for API parity with the former array cache."""
        return self

    def overwrite_array(self, data=None):
        """No-op retained for API parity with the former array cache."""
