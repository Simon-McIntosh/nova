# Fallback rows re-judged against the retained fan's refinement floor

Decision `fallback-fence-norm` is locked to `moment-relative-l2`. The fence for
the five fallback rows (the rows with no independent coupling error term) is the
retained fan's own refinement floor, expressed in the moment relative L2 norm
the rows are judged in. `receipt.json` beside this file is the machine record;
this page is the readable one.

- source revision: `a42dc665cf41a4a698ba95df16fee43cfe3c4a32`
- host: `98dci4-clu-2018.iter.org`, job `1277272`, JAX backend `cpu`, binary64
- driver: `rejudge_fallback_rows.py` (measurement log in the run directory)
- norm: `||closed-form route − retained fan||₂ / ||retained fan||₂` per moment,
  over the boundary (cut) cells of the row

## The fence

The fence is measured in this run, on the weak 110 row, by the same fan arm that
supplies the rows' reference:

```
fence(moment) = (4/3) · || fan@order16 − fan@order8 ||₂ / || fan@order8 ||₂
```

| moment | fence (relative L2) |
| --- | --- |
| current | 2.26749e-16 |
| radial | 4.44149e-15 |
| vertical | 5.06496e-15 |

## The five rows

A row meets the fence when all three moment series sit at or below it. The
native arc resolution is 128 segments — the clip's own sampled arc, so no
re-sampling enters the headline number. The smallest swept arc count meeting the
fence is given for the passing rows; the sweep (8, 16, 32, 64, 128) is in
`receipt.json`.

| case | cells | cuts | current error | radial error | vertical error | verdict | smallest count meeting fence |
| --- | --- | --- | --- | --- | --- | --- | --- |
| weak-rotation-reactor-static | 1000 | 131 | 2.01362e-16 | 1.69513e-15 | 1.70405e-15 | **pass** | 128 |
| moderate-rotation-conventional-static | 1000 | 136 | 2.29975e-16 | 2.19660e-15 | 1.30226e-15 | **fail** | — |
| strong-rotation-compact-static | 110 | 44 | 2.96282e-16 | 1.10522e-15 | 1.61798e-15 | **fail** | — |
| strong-rotation-compact-static | 300 | 73 | 1.73051e-16 | 2.05311e-15 | 1.17388e-15 | **pass** | 128 |
| strong-rotation-compact-static | 1000 | 129 | 2.06993e-16 | 1.88437e-15 | 1.49006e-15 | **pass** | 128 |

Three rows pass, two fail. Each failure is carried by the **current** moment
alone: radial and vertical sit below the fence on every row.

- **strong-rotation-compact-static @ 110** fails on current by 31%
  (2.963e-16 against 2.267e-16), and no swept arc count brings it under — the
  polyline is not the limiting error there.
- **moderate-rotation-conventional-static @ 1000** fails on current by 1.4%
  (2.300e-16 against 2.267e-16). This is a **marginal** verdict: at the
  round-off scale the two numbers are one part in seventy apart, so the
  ordering is not physically meaningful on its own. The evidence fragment says
  so, and the plan's own reading of this row should be treated as open rather
  than settled.

## What this replaces

Two fences are retired here.

1. **One tenth of the fan refinement floor** (`fan-floor-tenth`):
   `{current 2.924e-17, radial 5.109e-16, vertical 5.064e-16}` — a budget, not
   that floor, and it re-imports the tenth the decision removed.
2. **The closed-form round-off envelope** (`closed-form-envelope`):
   `{current 2.038e-16, radial 2.060e-15, vertical 4.789e-16}` — a property of
   the closed-form route's own accumulated round-off, not a bound on the fan's
   refinement, so comparing the route against it measures the route against
   itself.

## A note on the fence's own value

The fan refinement floor recorded by the earlier fan-floor study was
`{current 2.92403e-16, radial 5.10933e-15, vertical 5.06364e-15}`, measured at
revision `d9b594b2`. Measured here at `a42dc665c`, the current moment's floor is
`2.26749e-16` — 22% lower. The floor is a property of the fan arm on the
geometry the rows are built on, and the clip's own arc sampling changed between
those revisions, so the floor moved with it. The fence used for the verdicts
above is the one measured in the same process as the rows, which is the only
pairing in which the row error and its fence describe the same instrument.
Against the *recorded* floor instead, moderate-1000's current error (2.300e-16)
would sit just under it and that row would read pass; the strong-110 failure is
unaffected either way.