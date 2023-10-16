from datetime import datetime
import pytest


from nova.imas.machinedescription import CADSource, YAML


@pytest.fixture
def cad_source():
    return CADSource()


def test_default_cad_source(cad_source):
    assert len(cad_source.source) == 6


def test_default_cad_source_fields(cad_source):
    assert [attr.split(":")[0] for attr in cad_source.source] == [
        "Reference",
        "Objects",
        "Filename",
        "Date",
        "Provider",
        "Contact",
    ]


def test_default_date(cad_source):
    assert cad_source.date == datetime.now().strftime("%d/%m/%Y")


def test_source_date():
    assert CADSource(date="27/7/2017").date == "27/7/2017"


def test_yaml_data():
    yaml = YAML(pbs=11, description="test str")
    assert yaml.data["pbs"] == "PBS-11"
    assert yaml["description"] == "test str"


def test_yaml_contacts():
    yaml = YAML(provider={"name": "a", "email": "a@b"}, officer="fred, fred@iter")
    assert yaml["provider", "email"] == "a@b"
    assert yaml["officer", "name"] == "fred"


'''
    @property
    def _status_yaml(self):
        return None
        # {attr: self.ids_metadata.get(attr, "") for attr in ["status", "replaced_by", "replaces", "reason_for_replacement"]}
        # | {"backend": self.backend}

    @property
    def _correction_coils_yaml(self):
        """Return correction coil machine description yaml metadata."""
        return {
            "ids": "coils_non_axisymmetric",
            "pbs": "PBS-11",
            "data_provider": self.provider.split(", ")[0],
            "data_provider_email": self.provider.split(", ")[1],
            "ro": "Fabrice Simon",
            "ro_email": "Fabrice.Simon@iter.org",
            "description": self.ids_metadata["comment"],
            "provenance": self.ids_metadata["source"][0].split()[-1],
        }

    @property
    def _central_solenoid_yaml(self):
        """Return central solenoid machine description yaml metadata."""
        return {
            "ids": "coils_non_axisymmetric",
            "pbs": "PBS-*",
            "data_provider": self.provider.split(", ")[0],
            "data_provider_email": self.provider.split(", ")[1],
            "ro": "*",
            "ro_email": "*",
            "description": self.ids_metadata["comment"],
            "provenance": self.ids_metadata["source"][0].split()[-1],
        }

    @property
    def yaml(self):
        """Return machine description yaml metadata."""
        system = self.ids_metadata["system"]
        return self._base_yaml | getattr(self, f"_{system}_yaml")

'''


if __name__ == "__main__":
    pytest.main([__file__])
