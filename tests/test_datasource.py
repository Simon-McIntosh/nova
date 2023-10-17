from datetime import datetime
import pytest


from nova.imas.datasource import (
    CAD,
    DataSource,
    YAML,
)


@pytest.fixture
def cad():
    return CAD(cross_section={"circle": [0, 0, 0.025]})


def test_default_cad_source(cad):
    assert len(cad.source) == 7


def test_default_cad_source_fields(cad):
    assert [attr.split(":")[0] for attr in cad.source] == [
        "cross-section",
        "reference",
        "objects",
        "filename",
        "date",
        "provider",
        "contact",
    ]


def test_default_date(cad):
    assert cad.date == datetime.now().strftime("%d/%m/%Y")


def test_source_date():
    assert CAD(cross_section={"s": [0, 0, 0.001]}, date="27/7/2017").date == "27/7/2017"


def test_yaml_data():
    yaml = YAML(pbs=11, description="test str")
    assert yaml.data["pbs"] == "PBS-11"
    assert yaml["description"] == "test str"


def test_yaml_contacts():
    yaml = YAML(provider={"name": "a", "email": "a@b"}, officer="fred, fred@iter")
    assert yaml["provider", "email"] == "a@b"
    assert yaml["officer", "name"] == "fred"


@pytest.fixture
def datasource():
    return DataSource(
        pulse=111003,
        run=2,
        name="coils_non_axisymmetric",
        pbs=11,
        provider="Simon McIntosh, simon.mcintosh@iter.org",
        officer="Fabrice Simon, fabrice.simon@iter.org",
        status="active",
        replaces="111003/1",
        reason_for_replacement="resolve conductor centerlines and include coil feeders",
        cad=CAD(
            cross_section={"square": [0, 0, 0.0148]},
            reference="DET-07879",
            objects="Correction Coils + Feeders Centerlines Extraction "
            "for IMAS database",
            filename="CC_EXTRATED_CENTERLINES.xls",
            date="05/10/2023",
            provider="Vincent Bontemps, vincent.bontemps@iter.org",
            contact="Guillaume Davin, Guillaume.Davin@iter.org",
        ),
    )


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
