# Deterministic twin-truth package

The package contains six marched windows for one known truth and three member-preserving counterfactuals. `trajectory.tsv` carries prescribed flux-function/drive scales and convergence/conservation receipts; `observations.tsv` carries Thomson, net-current, COCOS, unit, support and error receipts per row; `coupling_states.jsonl` is the exact versioned window-boundary handoff.

Ambix should group by `member_id`, order by `window_index`, reconstruct each boundary with `CouplingState.from_dict`, and admit only rows whose window receipt converged. The `truth` observation rows are deterministic measurements; the other member rows are counterfactual predictions.

The joint recovery gate is deliberately **not run here**. Estimator selection, inference, coverage and scoring remain Ambix-owned; Nova provides only truth, forward members, observations and receipts.
