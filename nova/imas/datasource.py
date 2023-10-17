"""Manage the creation of machine description IDSs."""
from dataclasses import dataclass, field, fields
from datetime import datetime

import yaml

from nova.database.filepath import FilePath
from nova.geometry.polygon import Polygon
from nova.imas.database import Database, IDS
from nova.imas.metadata import Contact, Contacts


@dataclass
class CAD(Contacts):
    """Manage CAD source metadata."""

    cross_section: dict | Polygon
    reference: str | None = None
    objects: str | None = None
    filename: str | None = None
    date: datetime | str = field(default_factory=datetime.now)
    provider: Contact | str = "Vincent Bontemps, vincent.bontemps@iter.org"
    contact: Contact | str = "Guillaume Davin, Guillaume.Davin@iter.org"

    def __post_init__(self):
        """Convert data field to string."""
        if isinstance(self.date, datetime):
            self.date = self.date.strftime("%d/%m/%Y")
        super().__post_init__()

    @property
    def source(self):
        """Manage CAD source metadata."""
        return [
            f"{cad_field.name.replace('_', '-')}: {getattr(self, cad_field.name)}"
            for cad_field in fields(CAD)
        ]

    @property
    def data(self) -> dict:
        """Return CAD attribute."""
        return {
            cad_field.name: getattr(self, cad_field.name) for cad_field in fields(CAD)
        }


@dataclass
class YAML(Contacts, IDS):
    """Manage machine description YAML data."""

    pbs: int = 0
    provider: Contact | str = "default_name, default_email"
    officer: Contact | str = "default_name, default_email"
    description: str | None = None
    provenance: str | None = None
    status: str = "active"
    replaces: str = ""
    reason_for_replacement: str = ""

    @property
    def data(self):
        """Return yaml dict."""
        return {
            "ids": self.name,
            "pbs": f"PBS-{self.pbs}",
            "data_provider": self["provider", "name"],
            "data_provider_email": self["provider", "email"],
            "ro": self["officer", "name"],
            "ro_email": self["officer", "email"],
            "description": self.description,
            "provenance": self.provenance,
            "status": self.status,
            "replaces": self.replaces,
            "reason_for_replacement": self.reason_for_replacement,
        }

    def write(self):
        """Write machine description yaml file."""
        ids_path = Database(**self.ids_attrs).ids_path
        filepath = FilePath("md_summary.yaml", ids_path).filepath
        with open(filepath, "w") as file:
            yaml.dump(
                {f"{self.pulse}/{self.run}": self.data},
                file,
                default_flow_style=False,
                sort_keys=False,
            )


@dataclass
class DataSource(YAML, IDS):
    """Manage machine description data source."""

    machine: str = "iter_md"
    cad: CAD | dict = field(default_factory=CAD)

    def __post_init__(self):
        """Initialize data source attributes."""
        for _field in fields(DataSource):
            if isinstance(value := getattr(self, _field.name), dict):
                setattr(self, _field.name, _field.default_factory(**value))
        super().__post_init__()
