# DIII-D description and current assertion sweep

## Result

The DIII-D description and current layers now state the measured tree reality.
The production current route has 19 competition-row channels followed by five
conductors driven from ECOILA by the registered `pf_active` circuit, for 24
response columns and zero production current parameters. The former
`OMITTED_POLOIDAL_CONDUCTORS` and `UNKNOWN_POLOIDAL_CONDUCTORS` names have been
removed without compatibility aliases.

The competition dataset still contributes no wall contour, so
`DiiidDatasetMachineDescription.validate()` continues to require
`machine.contour is None`. That dataset-scoped fact is now stated beside the
separate machine-description fact: `diiid_machine_ids.py` authors and validates
a deterministic physical wall ring in the `wall` IDS.

## Corrected assertions

| File | Former text or identity | Replacement | Tree evidence that the former assertion was false or incomplete |
| --- | --- | --- | --- |
| `nova/imas/diiid_current.py` | `OMITTED_POLOIDAL_CONDUCTORS = ("ECOILB", "E567UP", "E567DN", "E89UP", "E89DN")` | The current layer imports and uses `CIRCUIT_DRIVEN_CONDUCTORS` from `diiid_description.py` for declaration order, response construction, and response-order validation. No alias remains. | `PF_ACTIVE_CIRCUIT.validate()` requires its ordered drive names to equal `CIRCUIT_DRIVEN_CONDUCTORS`, includes them in the closed series circuit, and `current_declarations()` assigns every production drive a `CIRCUIT_RELATIONS` relation sourced from ECOILA. |
| `nova/imas/diiid_current.py` | `UNKNOWN_POLOIDAL_CONDUCTORS = ("ECOILB", "E567DN", "E89UP")` | `CIRCUIT_BYPASS_PRIOR_CONDUCTORS` names the three prior-driven slots that exist only when the caller explicitly selects the diagnostic circuit bypass. Production still iterates over all five `CIRCUIT_DRIVEN_CONDUCTORS`. No alias remains. | With the default `use_circuit=True`, the resolution has `unknown_names == ()`, its receipt says `current_authority = "pf_active circuit"`, and the banked circuit receipt records `free_current_count_with_circuit == 0`. The three prior slots exist only in the tested `use_circuit=False` diagnostic route. |
| `nova/imas/diiid_current.py` | `KNOWABLE_RELATIONS` | `CIRCUIT_BYPASS_RELATIONS` | These two independent relations are selected only inside the explicit circuit-bypass branch; production uses `CIRCUIT_RELATIONS` for all five registered drives. |
| `nova/imas/diiid_current.py` | Public parameter `unknown_priors` | `circuit_bypass_priors` in `current_declarations()`, `resolve_diiid_currents()`, and `complete_profile_current_adapter()` | The parameter is ignored by the production circuit route and is required only when `use_circuit=False`; the relation-based name states that scope rather than assigning permanent ignorance to conductors with registered circuit drives. |
| `nova/imas/diiid_current.py` | Module docstring: “Bind DIII-D conductor-current tiers to the complete response matrix.” | “Bind DIII-D conductor currents to the measured circuit and response matrix.” | The complete response is not merely a tier proposal: `PF_ACTIVE_CIRCUIT` is the default current authority and supplies deterministic relations for every added column. |
| `nova/imas/diiid_current.py` | `current_declarations()` docstring: “Return all DIII-D declarations without inference-time label access.” | “Return the measured circuit declarations or an explicit diagnostic bypass.” | Both routes avoid inference-time label access, but only the replacement tells the reader that the measured circuit is production authority and the prior route is an explicit diagnostic selection. |
| `nova/imas/diiid_current.py` | Validation text: “DIII-D unknown priors must name ECOILB, E567DN, and E89UP exactly” | “DIII-D circuit-bypass priors must name ECOILB, E567DN, and E89UP exactly” | The check is reached only for `use_circuit=False`; those conductors are circuit-driven in production. |
| `nova/imas/diiid_current.py` | `circuit_current_map()` docstring: “Drive every unshipped conductor from shipped ECOILA ampere-turns.” | “Drive every circuit-represented conductor from shipped ECOILA ampere-turns.” | The currents lack separate competition-row columns, but their machine relation is present: `PF_ACTIVE_CIRCUIT.currents()` maps ECOILA through all five registered drives. |
| `nova/imas/diiid_current.py` | `resolve_diiid_currents()` docstring: “Resolve 24 ordered currents through the circuit or explicit free slots.” | “Resolve 24 currents with the measured circuit as the default authority.” | The default resolution has no unknown names and uses five `CIRCUIT_RELATIONS`; prior-driven slots are a separately named diagnostic bypass, not the current state of the conductors. |
| `nova/imas/diiid_current.py` | `complete_profile_current_adapter()` docstring: “Append exact omitted-coil columns and bind the ordered tier resolution.” | “Append circuit-driven columns and bind the ordered current resolution.” | Both grid and wall response additions now request and validate `CIRCUIT_DRIVEN_CONDUCTORS`; the adapter receipt records 19 row channels, 24 complete channels, and `pf_active circuit` authority. |
| `nova/imas/diiid_current.py` | Bypass receipt: `current_authority = "free-parameter opt-out"` | `current_authority = "diagnostic prior-driven circuit bypass"` | The branch is not the shipped default and exists to exercise an explicit diagnostic. The production receipt remains `pf_active circuit`. |
| `nova/imas/diiid_description.py` | Module docstring: “routes the shipped ECOILA ampere-turns through the unshipped ohmic conductors” | “routes the shipped ECOILA ampere-turns through the circuit-driven ohmic conductors whose currents have no separate competition-row columns” | `CIRCUIT_DRIVEN_CONDUCTORS` and `PF_ACTIVE_CIRCUIT` supply the relation; what is absent is a separate dataset column, not the wiring authority. |
| `nova/imas/diiid_description.py` | `PfActiveCircuitRecord.currents()` docstring: “Map one shipped ECOILA value to every unshipped conductor.” | “Map one shipped ECOILA value to every circuit-driven conductor.” | The method iterates over the five validated `PF_ACTIVE_CIRCUIT.drives`, and each drive gain is applied to ECOILA. |
| `nova/imas/diiid_description.py` | Wall refusal: “the competition dataset does not ship a wall contour” | “the competition dataset does not ship a wall contour; the dataset-scoped description must leave machine.contour unset, while the DIII-D machine-description IDS carries the physical wall ring” | The first clause remains true and validation behavior is unchanged. Separately, `build_diiid_machine_ids()` authors a `wall` IDS; `DiiidMachineIds.validate()` reads `description_2d[0].limiter.unit[0].outline`, requires a valid simple polygon, and checks its digest and vertex count against repair provenance. |
| `nova/imas/diiid_description.py` | Class documentation described only competition geometry and did not state where the wall lives. | `DiiidDatasetMachineDescription` now says the competition-row layer intentionally leaves `machine.contour` unset and the machine-description IDS separately retains the deterministic physical wall ring. | `dataset_machine_description()` constructs `StaticMachineDescription` with `contour=None`, while `diiid_machine_ids.py::_author_wall()` writes the validity-repaired physical ring into the wall IDS. Both statements are simultaneously true. |
| `nova/imas/diiid_description.py` | `provenance_complete` docstring: “every present and explicitly absent quantity is traced” | “each competition-row quantity and external route is traced” | Treating the wall as simply absent conflated the competition rows with the machine-description IDS. The receipt now records both scopes. |
| `nova/imas/diiid_description.py` | Wall receipt: “wall contour explicitly absent and no external source selected” | “wall contour absent only from the competition rows; the DIII-D machine-description IDS separately retains a deterministic physical wall ring” | The machine IDS set contains `wall`, `pf_active`, and `magnetics`; its wall outline is authored from the repaired source-machine contour and protected by exact digest and vertex-count checks. |
| `tests/imas/test_diiid_current.py` | `_priors`, `test_circuit_opt_out_retains_three_free_current_slots`, and `test_outer_loop_drives_only_three_diiid_unknown_slots` | `_circuit_bypass_priors`, `test_circuit_bypass_exposes_three_prior_driven_diagnostic_slots`, and `test_outer_loop_updates_only_three_circuit_bypass_prior_slots` | These fixtures exercise `use_circuit=False`, not production current authority. The production tests separately require `unknown_names == ()`, `CIRCUIT_DRIVEN_CONDUCTORS` response order, and `pf_active circuit` authority. |
| `tests/imas/test_diiid_current.py` | Adapter order expected `OMITTED_POLOIDAL_CONDUCTORS` | Adapter order expects `CIRCUIT_DRIVEN_CONDUCTORS` | The response columns are authored from `PF_ACTIVE_CIRCUIT` drives in exactly that order. |

## Quantitative current authority

The registered order and gains are:

| Conductor | Integer wiring ratio | Adjudication | Effective gain from ECOILA |
| --- | ---: | --- | ---: |
| ECOILB | 2.0 | `snapped-exact` | 2.0 |
| E567UP | 1.0 | `snapped-exact` | 1.0 |
| E567DN | 1.0 | `snapped-exact` | 1.0 |
| E89UP | 1.0 | `integer-plus-systematic` | 1.0456947569496173 |
| E89DN | 1.0 | `integer-plus-systematic` | 1.0456240764323717 |

The default current adapter therefore carries 19 competition-row conductors +
5 circuit-driven conductors = 24 response currents, with zero unknown names.
The two end-loop conductors retain their explicitly named measured systematic;
the report does not word those effective gains into an exact-integer result.

## Real-wall consumer handoff

The separately fenced forward consumer must not read
`DiiidDatasetMachineDescription.machine.contour`; that field is deliberately
`None` because it represents only the competition rows. It needs the DIII-D
machine-description IDS bundle and the following physical outline:

```python
bundle = build_diiid_machine_ids()
outline = bundle.ids["wall"].description_2d[0].limiter.unit[0].outline
wall_r = np.asarray(outline.r, dtype=float)
wall_z = np.asarray(outline.z, dtype=float)
```

`build_diiid_machine_ids()` validates the bundle before returning it. The wall
ring has already been validity-repaired and canonicalized; validation requires
the published outline to be a valid simple polygon and to match the repair
receipt's SHA-256 digest and vertex count. A forward topology route should keep
that IDS/source/DD provenance in its receipt, use this ring as its explicit
physical-wall selection, and retain the EFIT-grid rectangle only under its own
explicit pseudo-wall fallback selection. The forward driver itself was outside
this correction's write fence and was not modified.

## Sweep and test evidence

- The assigned Python paths contain no occurrence of
  `OMITTED_POLOIDAL_CONDUCTORS`, `UNKNOWN_POLOIDAL_CONDUCTORS`, or
  `unknown_priors` after the correction.
- The current tests require the production resolution to have 24 currents,
  `unknown_names == ()`, five circuit drives, and `pf_active circuit` authority.
- Combined touched-module suite: **26 collected, 26 passed** in 15.46 seconds:
  7 current tests, 7 description tests, 5 dataset-machine-description tests,
  4 machine-IDS tests, and 3 IDS-machine-description tests.
- The run emitted one existing NumPy binary-compatibility `RuntimeWarning` from
  the live machine-IDS export test; it did not fail or skip a test.
- Full log:
  `/home/ITER/mcintos/.config/reckon/crew/runs/r-20260824T044415822924-stale-assertion-correction/touched-module-tests-final.log`.
