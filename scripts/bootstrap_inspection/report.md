# Sauter bootstrap-current inspection

This is an owner-authorised non-identical configuration ruling dated 2026-08-23. It is an inspection, not an identity replay of the banked zeros configuration.

## Configuration identity

Exactly `1` configuration leaf changed: `neoclassical.bootstrap_current.model_name` from `zeros` to `sauter`. All other configuration fields and the full source-dictionary shape compare identical.
The banked raw dictionary omits this selector and TORAX validation resolves it to `zeros`. Both inspection dictionaries set the leaf explicitly before validation.

## WindowReceipt evidence

In one process, environment, and prepared fixture, the implicit-default and explicit-zeros contractions were `0.2361950658487989` and `0.2361950658487989`. Their absolute difference `0` passes the `9.9999999999999998e-13` selector-equivalence gate.
Against the untouched historical bank, explicit zeros differs by `1.5267509478888996e-12`, passing the cross-environment `1.0000000000000001e-09` note. The independently measured same-configuration current-tree drift is `1.4484219379440333e-11` and backend drift is `1.6999999999999999e-09`.
The Sauter rerun returned `WindowReceipt` in `4` iterations with contraction `0.2361657751479937`, exit gating norm `0.0012936510592181348`, and exit all-field norm `0.032632660566478525`.

## Bootstrap profile

The zeros fraction remained exactly zero. The Sauter curves span `0` to `0.30852523791868824`. The maximum Nova-facade versus direct-TORAX separation is `0`.

Figure: `docs/figures/flux-function-forward-transport/bootstrap-model-comparison.svg`
Profiles: `scripts/bootstrap_inspection/profiles.tsv`
Machine receipt: `scripts/bootstrap_inspection/receipt.json`
