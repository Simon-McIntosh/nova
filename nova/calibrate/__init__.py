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
"""

__all__: list[str] = ["correction_model", "correction_set"]
