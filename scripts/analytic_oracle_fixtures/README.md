# Closed-form analytic oracle fixtures

`measure.py` constructs the moderate rotating analytic equilibrium without a
map reader or data archive. The exact field, source profiles, current density,
axis, and zero-flux separatrix all come from the same closed form. A smooth
limiter has one exact tangency to that separatrix, allowing Nova's production
topology reader to recover the boundary constant without mixing flux gauges.

Each requested hex resolution builds exact authored-polygon zeroth and first
flux blocks and persists them under an `analytic-oracle-hex-machine` semantic
identity. The store family is `analytic_oracle_hex_machine`; it is distinct
from the reference-reproduction carrier. The prescribed exterior is the
closed-form total field minus the exact analytic-density plasma image on the
same fixed supports. Consequently, the exact field is the boundary condition
by construction and the reported one-step forcing isolates the production
density-moment image.

The state in that comparison is not obtained from Nova's coupling. `x_a` is
evaluated independently as `2*pi*case.flux(R,Z)` at every grid, wall, and
direct-sample target, and the receipt compares `g(x_a)` with those same direct
closed-form values. The banked tautology audit closes the residual against
`G*(m_production(x_a)-m_exact)` and perturbs the single exterior-current
amplitude. A finite response matching the prescribed-exterior prediction
demonstrates that the round-off baseline is not a hard-coded fixed point. The
exact exterior deliberately removes the kernel and far-field composition
terms measured by `benchmarks/efit_analytic_roundtrip_floor.py`; the remaining
quantity is the production density-moment representation error.

The two CPU debug-lane commands are encoded in `run-fixture.sh`. Their complete
stdout/stderr records are `coarse-run-bank.log` and `fine-run-bank.log`.
`measure.py --merge` combines the two independently banked receipts into
`results.json` and renders the comparison figure.
