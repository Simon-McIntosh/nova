"""Registry of characterized entry points.

Each :class:`EntryPoint` names a fitting/metrology operation, the recorded
inputs it consumes, a zero-argument ``run`` callable that produces a result
object, and the tolerance class for each canonical output array (by exact key
or by key prefix). The generator walks this registry to build goldens; the
component lane walks it to compare live runs against them.

The ``run`` callables live in :mod:`_entrypoints`, which imports the assembly
code lazily so this registry can be inspected without a heavy import.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field


@dataclass(frozen=True)
class EntryPoint:
    """One characterized operation."""

    id: str
    callable: str
    run: Callable[[], object]
    inputs: tuple[str, ...] = ()
    # Map an array key (exact) or key prefix (ending with '.') to a tolerance
    # class name. Longest matching prefix wins; unmatched keys use
    # ``tolerances_default``.
    tolerances: dict[str, str] = field(default_factory=dict)
    # Tolerance class for keys not matched by ``tolerances`` (e.g. an entry
    # whose every output is a millimetre delta sets this to ``length_mm``).
    tolerances_default: str = "default"
    # Environmental requirement gate; the entry point is skipped (not failed)
    # when this returns a reason string, runnable when it returns None.
    skip_reason: Callable[[], str | None] = lambda: None

    def tolerance_for(self, array_key: str) -> str:
        """Return the tolerance class name for a canonical array key."""
        if array_key in self.tolerances:
            return self.tolerances[array_key]
        best, best_len = self.tolerances_default, -1
        for pattern, klass in self.tolerances.items():
            if pattern.endswith(".") and array_key.startswith(pattern):
                if len(pattern) > best_len:
                    best, best_len = klass, len(pattern)
        return best


def registry() -> list[EntryPoint]:
    """Return the list of characterized entry points."""
    from . import _entrypoints

    return _entrypoints.build_registry()
