# Exchange-collapse diagnosis

The exchanged transport source reaches the fixture's vacuum/coreless fixed point on geometry exchange 5. This is a successful, finite equilibrium solve, not a transport failure and not a nonlinear-solver failure. The official `TopologySolveReceipt` reads `solver_succeeded=true`, final class `limited`, history `limited -> limited`, and zero transitions; the equilibrium nevertheless has zero core cells and zero plasma current. The exact Green-function evaluation independently has zero axis-connected core cells.

No Nova product file was modified. The probe reuses the existing 25 x 25 free-boundary fixture, its 49 x 49 exact extraction lattice, the public transport-to-source mapping, and the same 10 ms TORAX interval, Anderson solve budget, and 0.5 geometry damping as the demonstration.

## Drive scale against the fixture

The machine-readable comparison is in `drives.tsv`. It evaluates both callable source families at the same 17 normalised-flux points. At the collapsing exchange, the vector-magnitude ratios are:

| family | fixture magnitude | exchanged magnitude | exchanged / fixture | peak ratio |
|---|---:|---:|---:|---:|
| p-prime | 2.030603e6 Pa/Wb | 6.983162e5 Pa/Wb | 0.343896 | 0.288782 |
| ff-prime | 1.692169 T^2 m^2/Wb | 3.104453 T^2 m^2/Wb | 1.834599 | 2.298475 |

The pointwise ratios are finite wherever the fixture drive is nonzero. The edge ratio is marked `undefined` because both fixture gradients vanish there; the exchanged edge values are retained verbatim rather than divided by zero. These are order-unity changes, not an unbounded or orders-of-magnitude scale error. More decisively, exchange 4 remains confined with 576 exact core cells at p-prime and ff-prime magnitude ratios 0.341840 and 1.866129, while exchange 5 becomes coreless after only small ratio changes to 0.343896 and 1.834599.

## Stand-alone solve and topology receipt

The probe follows the demonstration's damped geometry trajectory and solves each sampled exchanged source independently from the fixture equilibrium. The measured terminal sequence is:

| exchange | exact core cells | solve-lattice core cells | fixed-point residual | plasma current | final topology | transitions | solver succeeded |
|---:|---:|---:|---:|---:|---|---:|---|
| 1 | 543 | 133 | 9.84e-16 | 609590.7 A | limited | 0 | true |
| 2 | 559 | 139 | 3.94e-16 | 611560.4 A | limited | 0 | true |
| 3 | 569 | 145 | 5.91e-16 | 612428.5 A | limited | 0 | true |
| 4 | 576 | 145 | 3.95e-16 | 611612.6 A | limited | 0 | true |
| 5 | 0 | 0 | 7.61e-17 | 0.0 A | limited | 0 | true |

Thus the solver does not fail and does not report a topology-class jump. It converges more deeply than the 1e-6 qualification threshold, but the selected limited fixed point has no confined domain. This is the known second fixed point of the absolute-source fixture: the non-confined vacuum branch can retain a named wall-limited topology even though no axis-connected plasma cells remain.

## Machine-consistency intervention

The fixture pressure scale and the initial TORAX pressure scale are already identical: both peak at 230257.98507535705 Pa. The pressure rescaling factor is therefore exactly 1.0. Following the declared intervention, both temperature channels and density were each multiplied by sqrt(1.0) = 1.0, and one transport exchange was rerun on the geometry immediately preceding the collapse.

That pressure-matched exchange reproduces the same sampled drives, the same 0.343896 and 1.834599 magnitude ratios, the same 7.61e-17 equilibrium residual, `solver_succeeded=true`, limited-to-limited history with zero transitions, zero plasma current, and zero connected core cells on both the solve and exact-evaluation lattices. Pressure rescaling therefore does not retain an axis-connected core; it falsifies demonstration-regime pressure scale as the cause.

## Ownership

The transport engine completes every interval with `NO_ERROR`, the seam returns finite order-unity flux functions, and changing no source amplitude between the measured pressure intervention and the ordinary case leaves the result unchanged. Adding a source-amplitude admissibility bound would tune away a valid order-unity source and would not distinguish exchange 4 from exchange 5. The corrective owner is therefore the equilibrium branch/seed and domain-qualification surface: it must either preserve the confined branch across the small source change or return a receipt that explicitly identifies a converged non-confined fixed point before extraction. The demonstration owner and the transport return-channel owner need no change from this finding.

Classification: **equilibrium branch or topology loss**.
