"""Manage access to IMAS database via DBEntry class."""
from dataclasses import dataclass, field


@dataclass
class Callstate:
    """Manage db_entry callstate."""

    args: tuple = ()
    kwargs: dict = field(default_factory=dict)

    def __call__(self, args, kwargs):
        """Update callstate."""
        self["args"] = args
        self["kwargs"] = kwargs

    def __setitem__(self, key, value):
        """Update callstate attributes."""
        match key:
            case "args":
                self.args += value
            case "kwargs":
                self.kwargs |= value
            case _:
                raise NotImplementedError(f"Mapping {key} not implemented")

    def clear(self):
        """Clear state."""
        self.args = ()
        self.kwargs.clear()
