"""Tests for corpus filename metadata parsing."""

from packaging.version import Version

from nova.assembly.provenance import sourcemeta


def test_sector_module_conforming():
    """A conforming sector-module workbook parses into structured fields."""
    meta = sourcemeta.parse_filename(
        "Sector_Module_#1_CCL_as-built_data_8LMK6A_v8_1.xlsx"
    )
    assert meta.kind == "sector_module"
    assert meta.sector == 1
    assert meta.idm_doc == "8LMK6A"
    assert meta.version == "8.1"
    assert meta.private is False
    assert meta.suffix == ".xlsx"
    assert meta.description == "CCL_as-built_data"


def test_private_leading_underscore():
    """A leading underscore marks a private in-work revision."""
    meta = sourcemeta.parse_filename(
        "_Sector_Module_#6_CCL_as-built_data_8LMK6A_v9_0.xlsx"
    )
    assert meta.private is True
    assert meta.kind == "sector_module"
    assert meta.sector == 6
    assert meta.version == "9.0"
    assert meta.idm_doc == "8LMK6A"


def test_version_ordering():
    """Parsed versions compare numerically, not lexically."""
    low = sourcemeta.parse_filename("Sector_Module_#2_CCL_x_ABCDEF_v4_0.xlsx")
    high = sourcemeta.parse_filename("Sector_Module_#2_CCL_x_ABCDEF_v10_0.xlsx")
    assert Version(low.version) < Version(high.version)
    assert low.parsed_version < high.parsed_version


def test_path_input_uses_basename(tmp_path):
    """Parsing accepts a full path and keys on its basename."""
    path = tmp_path / "IDM" / "Sector_Module_#3_CCL_y_ZZZ99A_v2_5.xlsx"
    meta = sourcemeta.parse_filename(path)
    assert meta.sector == 3
    assert meta.version == "2.5"


def test_opaque_pickle():
    """A non-conforming pickle name is classified opaque, not an error."""
    meta = sourcemeta.parse_filename("ILIS_nominal.pickle")
    assert meta.kind == "opaque"
    assert meta.suffix == ".pickle"
    assert meta.sector is None
    assert meta.version is None
    assert meta.idm_doc is None


def test_opaque_variants():
    """Assorted non-conforming names all classify opaque."""
    for name in [
        "helias4-filaments.xlsx",
        "measurements.nc",
        "notes.txt",
        "bundle.zip",
    ]:
        meta = sourcemeta.parse_filename(name)
        assert meta.kind == "opaque", name


def test_opaque_private_prefix():
    """A private prefix is honoured even for opaque names."""
    meta = sourcemeta.parse_filename("_scratch.txt")
    assert meta.private is True
    assert meta.kind == "opaque"
    assert meta.stem == "scratch"


def test_to_dict_serializable():
    """The metadata maps to plain YAML-serializable scalar types."""
    meta = sourcemeta.parse_filename(
        "Sector_Module_#1_CCL_as-built_data_8LMK6A_v8_1.xlsx"
    )
    payload = meta.to_dict()
    assert payload["kind"] == "sector_module"
    assert payload["version"] == "8.1"
    assert payload["sector"] == 1
    assert all(isinstance(v, (str, int, bool, type(None))) for v in payload.values())
