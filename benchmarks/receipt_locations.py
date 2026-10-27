"""Receipt locations come from the caller, never from a path baked into source.

A receipt written under a crew run or a worktree is reclaimed with it, so a
benchmark may not carry such a location as its default.  The location is named
by :data:`CREW_REPORTS_VARIABLE` or :data:`CREW_RUNS_VARIABLE`, and a missing
variable raises under the variable's own name rather than resolving a
directory that is not there.

The location resolves the first time it is used rather than when the module is
imported, so importing a benchmark to reach one of its pure functions does not
require the environment to be set.
"""

from __future__ import annotations

import os
from pathlib import Path

CREW_REPORTS_VARIABLE = "NOVA_CREW_REPORTS"
CREW_RUNS_VARIABLE = "NOVA_CREW_RUNS"


class CallerSuppliedPath(os.PathLike):
    """A path resolved from one environment variable when it is first used.

    It answers the path, string and division operations a receipt location is
    used for, and delegates anything else to the resolved :class:`Path`.  A
    missing variable raises ``SystemExit`` on first use, naming the variable.
    """

    __slots__ = ("_variable", "_description", "_parts", "_resolved")

    def __init__(self, variable: str, description: str, parts: tuple[str, ...]):
        self._variable = variable
        self._description = description
        self._parts = parts
        self._resolved: Path | None = None

    def resolve_path(self) -> Path:
        """Return the resolved path, raising under the variable's name if unset."""
        if self._resolved is None:
            root = os.environ.get(self._variable)
            if not root:
                raise SystemExit(
                    f"{self._variable} is not set; it must name the {self._description}"
                )
            self._resolved = Path(root).joinpath(*self._parts)
        return self._resolved

    def __fspath__(self) -> str:
        return os.fspath(self.resolve_path())

    def __str__(self) -> str:
        return str(self.resolve_path())

    def __repr__(self) -> str:
        return repr(self.resolve_path())

    def __truediv__(self, other: Any) -> Path:
        return self.resolve_path() / other

    def __rtruediv__(self, other: Any) -> Path:
        return Path(other) / self.resolve_path()

    def __getattr__(self, name: str) -> Any:
        return getattr(self.resolve_path(), name)


def crew_reports(*parts: str) -> CallerSuppliedPath:
    """Return a path under the crew report root named by ``NOVA_CREW_REPORTS``."""
    return CallerSuppliedPath(
        CREW_REPORTS_VARIABLE, "crew report root", tuple(parts)
    )


def crew_runs(*parts: str) -> CallerSuppliedPath:
    """Return a path under the crew run root named by ``NOVA_CREW_RUNS``."""
    return CallerSuppliedPath(CREW_RUNS_VARIABLE, "crew run root", tuple(parts))