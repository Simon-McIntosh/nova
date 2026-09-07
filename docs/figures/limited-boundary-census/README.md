# The limited-boundary defect: what is measured here

A MAST equilibrium label whose plasma is limited stands off the limiter it
should touch. The reconstruction these labels reproduce touches at 1.1 mm; the
labels stand a median 142.91 mm off, and on the worst frame the enclosed plasma
carries 4.2 per cent of the reconstruction area. The defect is live. A repair
was written, merged and gated, and changed nothing.

## The finding

The wall-anchor masking is correct code that never runs on the class it was
written for. Four independent lines converge, and they are not equally
informative: the first two measure the symptom, the last two the mechanism.

| line | what it establishes |
|---|---|
| paired bit-identity of the relabelled labels | the values did not change |
| limiter distance and enclosed area, reproduced independently | the defect survives the repair |
| the exclusion band derives from qualified saddles | a limited frame has none, so the band cannot form |
| masked wall nodes, counted per frame | **zero on every limited frame** |

The last row is the decisive one and comes from a separate census: 154 limited
frames replayed across seven shots, zero masked wall nodes and zero
winning-anchor-masked frames on every shot. Zero means the predicate never
speaks, so the anchor selection is not downstream of a mask firing on the wrong
node — which is what a nonzero count would have shown.

A class asymmetry in the round-off is what separated "the job never ran" from
"the job ran and the predicate never fired": the untargeted diverted class moved
1.18e-13 while the targeted limited class came back at exactly 0.000e+00. A hard
zero on both would instead have meant the comparison matched nothing, so any
zero here is quoted beside its denominator.

## Two numbers that mislead if quoted alone

**142.91 mm is a distance, not an excess.** A diverted boundary is bounded by
the separatrix rather than by contact and correctly stands 103.7 mm off the same
wall, so the limited excess is about 39 mm. A standoff threshold at 50 mm fires
on 79 of 81 frames including all 53 diverted ones: it selects for nothing.
Enclosed area is the measure that separates the populations — limited 0.773
median against 0.042 minimum, diverted 1.036 against 0.693.

**A 0.509 limited median against a banked 0.285 is population, not effect.**
The relabelled corpus is seven shots and the banked one is 871. Matched to the
same seven shots, the banked side reads 0.509 as well. Every band ratio is
identical to four decimals across the repair.

## Where the cold condition sits

`cold_start` marks a run's first frame or one whose predecessor failed the
branch guard, and a figure admits only guarded frames, so the first frame drawn
is cold by construction wherever a run opens with guard failures. All seven
shots do. The first frame drawn is cold on 7 of 7, limited on 7 of 7, and the
single weakest frame in its own sequence by flux span, rank 1 of 33 to 81. So
the cold condition is not a residual beside the frames a viewer or a decoder
starts from; it is the frame they start on, and it is not separable from the
limited class. It also recurs mid-sequence, at drawn ranks 27 through 71, so
trimming a lead frame fixes nothing.

## Files

| file | what it holds |
|---|---|
| `limited-class-census.npz` / `.json` | the corpus as first written: 54,415 guarded frames, 871 shots, with the interpretive reading of that corpus |
| `repaired-class-census.npz` / `.json` | the seven relabelled shots, descriptive only |
| `repair-verdict.json` | the paired and banded comparison of the two, with its findings |
| `boundary-contact-27079.json` / `-repaired.json` | per-class boundary-to-limiter distance, from the same summary the figure receipts write |
| `sequence-start-condition.json` | where the cold condition sits in each drawn sequence |

The banked arrays are the only pre-repair evidence any of this can be judged
against, so the emitter refuses to overwrite an existing array file and exits
nonzero rather than continuing. Relabelled output is written to a new session
root for the same reason.

## Producers

Each takes a session root and writes its own provenance, so one implementation
reads every corpus — a second reader of the same schema would make its own first
divergence look like a repair effect.

```bash
scripts/media/limited_class_census.py    --session <root> --output <json>
scripts/media/emit_census_evidence.py    --source <json> --name <basename>
scripts/media/validate_boundary_repair.py --after <npz>
scripts/media/emit_boundary_contact.py   --session <root> --shot <n>
scripts/media/emit_sequence_start_condition.py --session <root>
```

When a repair relabels 27079, the measurement that decides it is one invocation
of each: the first three drawn frames moving off 0.042, 0.335 and 0.571 on
enclosed area, and the contact figure moving off 142.91 mm — each beside its own
denominator.
