"""Manage the creation of machine description IDSs."""
from dataclasses import dataclass, field, fields
from datetime import datetime
from functools import cached_property
from typing import ClassVar, TypedDict

import yaml

from nova.database.filepath import FilePath
from nova.imas.database import Database, IDS, IdsEntry
from nova.imas.metadata import Metadata


class Contact(TypedDict):
    """Manage yaml contact."""

    name: str | None
    email: str | None


class Contacts:
    """Manage contact attributes."""

    def update_contacts(self, *contacts: str):
        """Update contact types."""
        for attr in contacts:
            if isinstance(value := getattr(self, attr), str):
                contact = dict(zip(["name", "email"], value.split(",")))
                setattr(self, attr, contact)

    def __getitem__(self, key):
        """Return contact attribtes."""
        match key:
            case str(contact), str(attr):
                return getattr(self, contact)[attr]
            case str(attr):
                return self.data[attr]
            case _:
                raise NotImplementedError(f"{key} not implemented")

    @property
    def data(self) -> dict:
        """Return instance data."""
        raise NotImplementedError


@dataclass
class CADSource(Contacts):
    """Manage CAD source metadata."""

    reference: str | None = None
    objects: str | None = None
    filename: str | None = None
    date: datetime | str = field(default_factory=datetime.now)
    provider: str = "Vincent Bontemps, vincent.bontemps@iter.org"
    contact: str = "Guillaume Davin, Guillaume.Davin@iter.org"

    def __post_init__(self):
        """Convert data field to string."""
        if isinstance(self.date, datetime):
            self.date = self.date.strftime("%d/%m/%Y")
        self.update_contacts("provider", "contact")

    @property
    def source(self):
        """Manage CAD source metadata."""
        return [
            f"{cad_field.name.capitalize()}: {getattr(self, cad_field.name)}"
            for cad_field in fields(CADSource)
        ]

    @property
    def data(self) -> dict:
        """Return CAD attribute."""
        return {
            cad_field.name: getattr(self, cad_field.name)
            for cad_field in fields(CADSource)
        }


@dataclass
class YAML(Contacts):
    """Manage machine description YAML data."""

    name: str | None = None
    pbs: int = 0
    provider: Contact | str = "default_name, default_email"
    officer: Contact | str = "default_name, default_email"
    description: str | None = None
    provenance: str | None = None

    def __post_init__(self):
        """Initialize contacts attribute."""
        self.update_contacts("provider", "officer")

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
class Properties:
    """Manage machine description metadata."""

    status: str = "active"
    system: str | None = None
    comment: str | None = None
    replaces: str = ""
    reason_for_replacement: str = ""

    def update_metadata(self, ids_entry: IdsEntry):
        """Update ids with instance metadata."""
        metadata = Metadata(ids_entry.ids_data)
        comment = self.ids_metadata["comment"]
        provenance = self.ids_metadata["source"]
        metadata.put_properties(comment, homogeneous_time=2, provenance=provenance)


@dataclass
class MachineDescription(IDS):
    """Manage machine description data."""

    machine: str = "iter_md"
    cad: CADSource = field(default_factory=CADSource)
    yaml: YAML = field(default_factory=YAML)

    machine_description_attrs: ClassVar[list[str]] = [
        "status",
        "pulse",
        "run",
        "occurrence",
        "name",
        "system",
        "comment",
        "source",
        "replaces",
        "reason_for_replacement",
    ]

    @property
    def data(self) -> dict:
        """Return system metadata."""
        return {attr: getattr(self, attr) for attr in self.machine_description_attrs}

    def update_metadata(self, ids_entry: IdsEntry):
        """Update ids with instance metadata."""
        metadata = Metadata(ids_entry.ids_data)
        # code_parameters = self.polyline_attrs
        code_parameters = {}
        description = self.ids_metadata["description"]
        metadata.put_code(code_parameters, description=description)

    def _check_status(self):
        """Assert that machine description status status is set."""
        assert len(self.ids_metadata["status"]) > 0

    @cached_property
    def _base_metadata(self):
        """Return base ids metadata."""
        return {
            "description": "An algoritum for the aproximation of CAD generated "
            "multi-point conductor centerlines by a sequence of "
            "straight-line and arc segments."
            "See Also: nova.geometry.centerline.Centerline"
        }

    @cached_property
    def ids_metadata(self):
        """Return ids metadata."""
        metadata = {
            "CC_EXTRATED_CENTERLINES": {
                "status": "active",
                "pulse": 111003,
                "run": 2,
                "occurrence": 0,
                "name": "coils_non_axisymmetric",
                "system": "correction_coils",
                "comment": "Ex-Vessel Coils (EVC) Systems (CC) - conductor centerlines",
                "source": [
                    "Reference: DET-07879",
                    "Objects: Correction Coils + Feeders Centerlines Extraction for "
                    "IMAS database",
                    "Filename: CC_EXTRATED_CENTERLINES.xls",
                    "Date: 05/10/2023",
                    "Provider: Vincent Bontemps, vincent.bontemps@iter.org",
                    "Contact: Guillaume Davin, Guillaume.Davin@iter.org",
                ],
                "replaces": "111003/1",
                "reason_for_replacement": "update includes coil feeders and "
                "resolves conductor centerlines with line and arc segments",
            },
            "CS1L": {
                "status": "active",
                "pulse": 0,
                "run": 0,
                "occurrence": 0,
                "name": "coils_non_axisymmetric",
                "system": "central_solenoid",
                "comment": "* - conductor centerlines",
                "source": [
                    "Reference: DET-*",
                    "Objects: *",
                    "Filename: CS1L.xls",
                    "Date: 12/10/2023",
                    "Provider: Vincent Bontemps, vincent.bontemps@iter.org",
                    "Contact: Guillaume Davin, Guillaume.Davin@iter.org",
                ],
                "replaces": "*",
                "reason_for_replacement": "*",
            },
        }
        try:
            return metadata[self.filename]
        except KeyError as error:
            raise KeyError(
                f"Entry for {self.filename} not present in self.metadata"
            ) from error

        # "cross_section": "square, 0.0148": [0, 0, 0.0148]}


@dataclass
class DataSource:
    """Manage machine description data source."""

    ids: MachineDescription | dict = field(default_factory=MachineDescription)
    cad: CADSource | dict = field(default_factory=CADSource)
    yaml: YAML | dict = field(default_factory=YAML)

    def __post_init__(self):
        """Initialize data source attributes."""
        for _field in fields(DataSource):
            if isinstance(value := getattr(self, _field.name), dict):
                setattr(self, _field.name, _field.default_factory(**value))

    @property
    def ids_attrs(self):
        """Return machine description ids attributes."""
        return self.ids.ids_attrs
