# Topology classification of the banked oracle roots

## Verdict

The bank contains two genuine fixed points, but **both are limited**. The closed-form
analytic state is wall-bound at the outboard limiter and the alternate fixed point is
wall-bound at the inboard limiter. At stored binary64 precision the production null
locator finds **0 finite in-domain X-points in all four reads** (two roots at coarse
and fine resolution), while every production boundary read is exactly the
corresponding wall read: the boundary-to-wall point distance and boundary-to-wall
flux difference are both zero.

The statement that the closed-form banked root is diverted is therefore not
supported by these arrays or by the analytic solution. There is **no genuinely
diverted fixture in the bank**. A separate fixture must be constructed and
independently banked with a finite in-domain saddle whose flux binds the
separatrix. Changing a stored topology label would not create that evidence.

![Fine-grid root topology, including the out-of-domain coordinate-axis saddle](topology-classification.png)

## Saddle analysis of the analytic state

For the closed form, `dpsi/dZ = -2*k_z*Z` and
`dpsi/dR = -2*R*G'(R^2-R0^2)`. In the physical cylindrical domain `R > 0`,
`G'` vanishes only at `R = R0`, so the sole physical stationary point is the
magnetic-axis maximum at `(1.7, 0) m`; there is no physical X-point.

Extending the formula to the excluded coordinate axis produces a mathematical saddle at
`(0, 0) m`. Its per-radian Hessian eigenvalues are
`[0.77744086558616043, -0.43965429246500937] Wb/m^2`, with
determinant `-0.34180521369266781`, hence opposite curvature signs. That point
is not a plasma X-point: it lies outside every wall and stored solve lattice. Its
nearest wall-vertex distance is `0.97862926143505558 m` and its nearest
fine-grid node is `0.97952220309003357 m` away. The fine grid starts at
`R = 0.97945194077533482 m`; the wall starts at
`R = 0.97835170363738877 m`.

## Stored-precision production reads

| Resolution | Root | Class | Finite X-points | Binding wall contact [m] | Boundary flux [Wb] | Fixed-point evidence |
|---|---|---:|---:|---|---:|---:|
| coarse | closed form analytic | limited | 0 | `[2.15, 1.148592147035757e-16]` | -8.6736173798840355e-19 | exact closed form |
| coarse | alternate fixed point | limited | 0 | `[0.9783517036373888, -0.0001210354195923481]` | -0.24590656947069361 | 2.4941706048983241e-15 |
| fine | closed form analytic | limited | 0 | `[2.15, 1.148592147035757e-16]` | -8.6736173798840355e-19 | exact closed form |
| fine | alternate fixed point | limited | 0 | `[0.9783517036373888, -0.0007319415012626677]` | -0.24697114012695601 | 2.1720128948495349e-15 |

The locator used its float64 local quadratic fit on unchanged binary64 bank arrays.
Input SHA-256 identities were checked against each source receipt before
classification. Neither receipt label was used as a classification input, and no
nonlinear solve was run. The raw field gauge was retained separately for each state;
its axis, wall, and boundary values were all read from that same field.

The independent root separation is retained: the coarse axes differ by
`74.8388723 mm` and the fine axes by
`74.1888564 mm`. The alternate roots remain criterion-qualified at
relative residuals `2.4941706048983241e-15` and `2.1720128948495349e-15`. Distinct
fixed points do not imply distinct topology classes.

## Acceptance implication

The bank can exercise recovery of two distinct roots and same-class portfolio
behavior. It cannot close an acceptance test requiring one limited and one diverted
branch. Until a finite, in-domain saddle-bound state is independently constructed
and banked, the diverted branch must report unavailable or non-converged on this
fixture; treating either limited state as diverted would be a false pass.

Machine-readable receipt: `topology-classification.json`.
