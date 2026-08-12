# Diagnostic corrections

This file covers the storage side of the package: the schema a correction is written
against, the artefacts generated from it, and where instances live. The measurement
side — `coupling.py`, `gain.py`, `gain_check.py`, `inversion.py`, `localize.py`,
`windows.py`, `instrument.py`, and the `corrections.py` engine that applies what they
establish — is oriented in the package docstring in `__init__.py`, and each module's
own docstring says what it guards against and why.

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

## Harvesting a pulse's own vacuum windows

`windows.py` splits a pulse's timeline into the intervals whose field is known
without solving anything — instrument-quiet, vacuum-driven, plasma — and
`instrument.py` takes the offset, integrator drift rate and drift curvature out of
the quiet ones and asks whether the integrator returned to the baseline it left.
The terms emit as `offset` and `drift_rate` correction records through
`instrument_corrections`, so what a channel needs comes back in the same form the
rest of the package reads.

Both take arrays and scalars only. Which current counts as a drive, which plasma
current ends the vacuum, how long the vessel and coil cases take to shed an induced
current, and what a channel is called are all the adapter's to supply. Two of those
choices are worth stating because getting them wrong is silent rather than loud:

- **Use the machine's *labelling* threshold, not its *modelling* one.** Classifying
  an interval is labelling. A bar set at the pickup floor is crossed by conductors
  that idle a little above it, and every crossing chops one quiet interval into two.
  On MAST that is the difference between `EXCITATION_CURRENT` and
  `ENERGISED_CURRENT`, and passing the latter turns an undriven shot's single
  1.5 s quiet window into more than two hundred fragments, none long enough to fit.
- **The settling guard is not free.** It is measured from the last *observed*
  disturbance, so a gap in the record neither starts one nor cancels one. Three
  time constants of MAST's slowest nominal passive mode is 216 ms against records
  1.5 s long, and three of the slowest *fitted* mode is 650 ms, which leaves a
  post-pulse window of 137 samples where the shorter guard leaves 3648. Which one
  an archive can afford is a measurement, not a default.

The closure defect is scored against the prediction's own uncertainty rather than
the channel's sample scatter. A pre-pulse window of a few tens of milliseconds
extrapolated across more than a second carries its fitted rate error over that gap,
which on one archive pulse exceeded the sample noise fifteenfold — scoring against
scatter alone called 65 of 73 channels non-closing on a machine that had done
nothing to them.

## Checking gain in vacuum-driven windows

`gain_check.py` contracts each channel's described Green's row with the recorded
conductor-current array, then fits the measured channel onto that exact vacuum
prediction over the classifier's driven windows. `fit_pulse_gain_checks` takes the
channel list plus callbacks for channel samples and response rows; it knows no store,
machine, current-family names, or channel convention. An optional instrument callback
supplies the offset and integrator walk already measured by `instrument.py`. Without
that callback, the samples supplied by the adapter must already have those additive
terms removed — the gain fit deliberately has no intercept.

A response callback may return either one row or a mapping of pickup-state name to
row. Each state gets its own scalar fit, so amplitude remains the gain while the
current-weighted waveform shape selects the pickup state. If two described states
differ only by an overall scale, the kernel refuses the decision: their residuals
tie after scalar fitting, which means pair state and gain are not identifiable from
that pulse. Accepted results are one gain per channel and pulse and feed directly
into `scale_step.py`; a scale transition is therefore bounded by the two consecutive
pulse numbers that actually measured it rather than by a curated shot block.
