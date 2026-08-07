# Diagnostic corrections

This file covers the storage side of the package: the schema a correction is written
against, the artefacts generated from it, and where instances live. The measurement
side — `coupling.py`, `gain.py`, `inversion.py`, `localize.py`, and the
`corrections.py` engine that applies what they establish — is oriented in the package
docstring in `__init__.py`, and each module's own docstring says what it guards
against and why.

`schema/diagnostic_correction.yaml` is the authored source. `correction_model.py`
and `schema/diagnostic_correction.schema.json` are generated from it and committed
so that reading a document needs neither the LinkML toolchain nor a build step —
only `pydantic` and `pyyaml`.

## Regenerating the artefacts

The generators live in the optional `schema` extra, which no runtime path imports:

```bash
uv sync --extra schema
uv run gen-pydantic nova/calibrate/schema/diagnostic_correction.yaml \
  > nova/calibrate/correction_model.py
uv run gen-json-schema nova/calibrate/schema/diagnostic_correction.yaml \
  > nova/calibrate/schema/diagnostic_correction.schema.json
```

Both files are written verbatim from the generator, which is why they carry lines
past the project's line limit and imports the module does not use; `pyproject.toml`
exempts the generated module from those two lint rules and from nothing else. Do
not hand-edit either file — `tests/calibrate/test_generated_artifacts.py` regenerates
both and fails on any difference whenever the toolchain is installed, so an edit
that survives locally will not survive there.

Bump `version` in the LinkML schema and `SCHEMA_VERSION` in
`nova/calibrate/correction_set.py` together; a document declaring a version the
reader does not implement is refused rather than half-understood.

## Instances

`corrections/<machine>/<diagnostic_system>.yaml` — one document per diagnostic
system, named for the data-dictionary IDS that holds it. Each carries its own
`set_version`, bumped when any value, interval or status in it changes. The version
is semantic and never a content hash: a hash orders nothing and tells a consumer
nothing about compatibility.

Read one with:

```python
from nova.calibrate.correction_set import read_correction_set

document = read_correction_set("mast", "magnetics")
```

`read_correction_set` validates on every read — the generated models check the
document's shape, and `validate_correction_set` checks the two faults a well-shaped
document can still carry: intervals that overlap within one channel, kind and
status, and a quantised value that lands off its declared ladder.
