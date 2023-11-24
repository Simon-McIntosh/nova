"""Manage diagnostic name mappings between IMAS and CODAC UDA."""
from dataclasses import dataclass, field
from functools import cached_property
import re


@dataclass
class Diagnostic:
    """Map diagnostic name to UDA variable."""

    name: str
    intergral: bool = True
    group: str = field(init=False)
    category: str = field(init=False)
    index: int = field(init=False)

    def __post_init__(self):
        """Translate diagnostic name to uda variable."""
        self.group, self.category, self.index = self.split(self.name)

    @staticmethod
    def split(name: str) -> tuple[str, str, int]:
        """Return uda variable name from given diagnostic name."""
        match re.split("[:.-]", name):  # IDS
            case "55", str(group), "00", str(category), str(index):
                return group, category, int(index)
            case "D1", "H1", str(group), str(category), str(index):
                return group[:-2], category, int(index)
            case _:
                raise NotImplementedError(
                    f"Mapping for diagnostic name {name} " "not implemented."
                )

    @cached_property
    def uda_name(self):
        """Return uda attribute name."""
        return f"D1-H1-{self.group}00:{self.category}-{self.index}"

    @cached_property
    def ids_name(self):
        """Return ids attribute name."""
        return f"55.{self.group}.00-{self.category}-{self.index}"

    @property
    def field_name(self):
        """Return uda field name."""
        if self.intergral:
            return "adcI"
        return "adcP"

    @cached_property
    def variable(self):
        """Return uda variable including field name."""
        return f"{self.uda_name}/{self.field_name}"
