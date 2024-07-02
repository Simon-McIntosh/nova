"""Manage the creation of machine description IDSs."""

from dataclasses import dataclass, field, fields
from datetime import datetime
from functools import cached_property

import yaml

from nova.database.filepath import FilePath
from nova.imas.database import Database
from nova.imas.dataset import IdsBase
from nova.imas.metadata import Code, Contact, Contacts, Properties


@dataclass
class CAD(Contacts):
    """Manage CAD source metadata."""

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
    def provenance(self):
        """Manage CAD provenance metadata."""
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
class YAML(Contacts, IdsBase):
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

    def write_yaml(self):
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
class DataSource(YAML, IdsBase):
    """Manage machine description data source."""

    machine: str = "iter_md"
    cad: CAD | dict = field(default_factory=dict)
    attributes: dict = field(default_factory=dict)

    def __post_init__(self):
        """Initialize data source attributes."""
        if isinstance(self.cad, dict):
            self.cad = CAD(**self.cad)
        self.provenance = self.cad.reference
        super().__post_init__()

    @cached_property
    def yaml_attrs(self) -> dict:
        """Return yaml attributes with the duplicated ids key dropped."""
        return {attr: value for attr, value in self.data.items() if attr != "ids"}

    @cached_property
    def properties(self):
        """Return properties instance."""
        return Properties(
            homogeneous_time=2,
            comment=self.description,
            provider=self.provider,
            provenance=self.cad.provenance,
        )

    @cached_property
    def code(self):
        """Return code instance."""
        return Code()

    def update(self, ids):
        """Update properties and code ids nodes."""
        self.properties.update(ids.ids_properties)
        self.code.update(ids.code)
