"""Tests for the preservation manifest model."""

from nova.assembly.provenance import manifest, yamlio


def _make_tree(root):
    """Populate a small corpus tree and return it."""
    (root / "sub").mkdir()
    (root / "Sector_Module_#1_CCL_as-built_data_8LMK6A_v8_1.xlsx").write_bytes(b"wb1")
    (root / "sub" / "notes.txt").write_bytes(b"some notes")
    (root / "ILIS_nominal.pickle").write_bytes(b"\x80\x04pickle")
    return root


def test_build_manifest_header_and_files(tmp_path):
    """A built manifest carries the fixed header and one entry per file."""
    _make_tree(tmp_path)
    doc = manifest.build_manifest(tmp_path, corpus_name="coil-metrology")
    assert doc["generator"] == "nova.assembly.provenance"
    assert doc["corpus"] == "coil-metrology"
    assert "schema" in doc
    paths = [entry["path"] for entry in doc["files"]]
    assert paths == sorted(paths)
    assert "Sector_Module_#1_CCL_as-built_data_8LMK6A_v8_1.xlsx" in paths
    assert "sub/notes.txt" in paths


def test_entry_metadata_and_digest(tmp_path):
    """Each entry embeds digest, size, mtime, and parsed source metadata."""
    _make_tree(tmp_path)
    doc = manifest.build_manifest(tmp_path, corpus_name="c")
    by_path = {entry["path"]: entry for entry in doc["files"]}
    workbook = by_path["Sector_Module_#1_CCL_as-built_data_8LMK6A_v8_1.xlsx"]
    assert workbook["digest"].startswith("sha256:")
    assert workbook["size"] == 3
    assert workbook["mtime"].endswith("+00:00")
    assert workbook["metadata"]["kind"] == "sector_module"
    assert workbook["metadata"]["version"] == "8.1"
    assert by_path["ILIS_nominal.pickle"]["metadata"]["kind"] == "opaque"


def test_no_generation_timestamp(tmp_path):
    """The manifest must contain no generation timestamp (determinism)."""
    _make_tree(tmp_path)
    doc = manifest.build_manifest(tmp_path, corpus_name="c")
    assert "generated" not in doc
    assert "generated_at" not in doc
    assert "timestamp" not in doc


def test_rebuild_byte_identical(tmp_path):
    """Rebuilding an unchanged tree yields byte-identical canonical YAML."""
    _make_tree(tmp_path)
    first = yamlio.canonical_yaml_bytes(
        manifest.build_manifest(tmp_path, corpus_name="c")
    )
    second = yamlio.canonical_yaml_bytes(
        manifest.build_manifest(tmp_path, corpus_name="c")
    )
    assert first == second


def test_write_load_roundtrip(tmp_path):
    """A written manifest loads back equal to the built document."""
    _make_tree(tmp_path)
    doc = manifest.build_manifest(tmp_path, corpus_name="c")
    out = tmp_path / "manifest.yaml"
    manifest.write_manifest(doc, out)
    assert manifest.load_manifest(out) == doc


def test_verify_clean(tmp_path):
    """Verification of an untouched tree reports no differences."""
    _make_tree(tmp_path)
    doc = manifest.build_manifest(tmp_path, corpus_name="c")
    report = manifest.verify(tmp_path, doc)
    assert report == {"missing": [], "changed": [], "new": []}


def test_verify_changed(tmp_path):
    """Tampering with a file content is reported as changed."""
    _make_tree(tmp_path)
    doc = manifest.build_manifest(tmp_path, corpus_name="c")
    (tmp_path / "sub" / "notes.txt").write_bytes(b"tampered content here")
    report = manifest.verify(tmp_path, doc)
    assert report["changed"] == ["sub/notes.txt"]
    assert report["missing"] == []
    assert report["new"] == []


def test_verify_missing(tmp_path):
    """Deleting a file is reported as missing."""
    _make_tree(tmp_path)
    doc = manifest.build_manifest(tmp_path, corpus_name="c")
    (tmp_path / "ILIS_nominal.pickle").unlink()
    report = manifest.verify(tmp_path, doc)
    assert report["missing"] == ["ILIS_nominal.pickle"]
    assert report["changed"] == []


def test_verify_new(tmp_path):
    """Adding a file is reported as new."""
    _make_tree(tmp_path)
    doc = manifest.build_manifest(tmp_path, corpus_name="c")
    (tmp_path / "extra.dat").write_bytes(b"new file")
    report = manifest.verify(tmp_path, doc)
    assert report["new"] == ["extra.dat"]
    assert report["changed"] == []
    assert report["missing"] == []
