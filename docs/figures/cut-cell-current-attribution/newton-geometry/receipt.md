# Newton geometry step: live clip against the frozen per-trip clip

The three limited 110-cell certificate rows re-solved with the profile support clip as a live Newton-Krylov unknown (chord_live, inner Newton budget 12) and tabled against the frozen-clip rows and the committed whole-cell rows.

## Verdict

Newton on geometry does not beat Picard on geometry here: with the clip a live Newton-Krylov unknown, no row reaches the analytic equilibrium that the frozen per-trip geometry already fails to reach.

## Per-row table

| case | res (comm) | res (live) | axis (comm) | axis (live) | conv (live) | trips (live) | seed amp | term amp | cut churn | toward |
|---|---|---|---|---|---|---|---|---|---|---|
skipped-row
skipped-row
skipped-row

cut churn = count of cells whose cut status changed between successive trip-boundary reads of the live clip support.
