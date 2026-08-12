"""Diagnostic calibration: the corrections a channel needs, and what warranted them.

A diagnostic correction is a function of pulse range rather than a constant.  A
channel holds one calibration state over a run of pulses, steps, holds, and
sometimes steps back; a single number fitted across such a step is an average of
discrete states weighted by pulse count, describing no pulse and moving when the
pulse selection moves.

The schema authored in ``schema/diagnostic_correction.yaml`` carries every correction
kind the calibration ladder produces -- gains, acquisition range settings, pickup
pair states, offsets, drift rates, exclusions, quality states and unit conventions --
each scoped to a validity interval and each carrying the evidence that established
it.  :mod:`nova.calibrate.correction_set` reads a document through the models
generated from that schema and refuses one a consumer could misapply.  Instances live
under ``corrections/<machine>/``, one document per diagnostic system, versioned by
git and by an explicit monotonic set version.

Importing the subpackage is explicit: ``from nova.calibrate import correction_set``.

The measurement side of the ladder sits beside the storage side.
:mod:`nova.calibrate.coupling` regresses what a sensor reads that the description
does not predict onto the drives that could have produced it;
:mod:`nova.calibrate.gain` fits and pools per-channel scales, baselines and drift
rates; :mod:`nova.calibrate.inversion` solves the sensors for the currents and
reports which current combinations they cannot resolve at all;
:mod:`nova.calibrate.localize` asks where a surviving residual came from by scanning
candidate sources over the poloidal plane; :mod:`nova.calibrate.windows` splits a
pulse into the intervals whose field is known without solving anything;
:mod:`nova.calibrate.instrument` measures what a channel reads across those
intervals and whether its integrator returns to the baseline it left; and
:mod:`nova.calibrate.partial` measures a regressor after shared level drives are
removed; :mod:`nova.calibrate.scale_step` and :mod:`nova.calibrate.pair_state`
resolve discrete acquisition states; :mod:`nova.calibrate.noise` measures sensor
floors and drift from waveform arrays; and
:mod:`nova.calibrate.corrections` applies what those fits established, in the order
the schema fixes.

Every module here is numerics over arrays.  A kernel takes samples, sensor positions
and axes, and a response matrix, and never a store path, a pulse number or a channel
naming convention -- those belong to the adapter that feeds it, which is machine code
and lives with the machine.  The one exception is deliberate: correction documents
under ``corrections/<machine>/`` are instance data, and naming a machine is what a
directory of instances is for.
"""

__all__: list[str] = [
    "correction_model",
    "correction_set",
    "corrections",
    "coupling",
    "gain",
    "instrument",
    "inversion",
    "localize",
    "noise",
    "pair_state",
    "partial",
    "scale_step",
    "windows",
]
