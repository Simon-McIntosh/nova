# Which direction generator escapes the stalled state?

The multi-branch predictor is the only measured generator with candidates that pass the unchanged merit, own-mask residual and origin-model trust inequalities on both rows. Those candidates occur only at diagnostic fractions outside the deployed six-factor ladder. The unchanged production selector refuses the selected direction on both rows. No measured generator closes the cold seed or reaches the analytic field; none is ready to adopt as a demonstrated convergence repair.

Multi-branch prediction is the supported next candidate-generator design, coupled to rebuilding its model: it generates substantially better qualified proposals whose actual decrease passes, and the same trust predicate accepts them under their rebuilt branch model. That is a measured reason to continue this design, not authority to promote it. The best proposals do not retain their construction branch at the endpoint, and one-step residuals remain far above the locked bound.

## Results with the unchanged origin model

| Cells | Generator | Best requested fraction | Merit | Sup residual | Rule-passing | Deployed selector takes it |
|---:|---|---:|---:|---:|---|---|
| 135 | branch_frozen | 0.001 | 0.0399490551567 | 0.0451923120658 | False | False |
| 135 | clip_geometry_continuation | 0.001 | 0.0400532380038 | 0.0453979135637 | False | False |
| 135 | multi_branch | 0.001 | 0.0391396120225 | 0.0450578059084 | True | False |
| 342 | branch_frozen | 0.001 | 0.0546260008946 | 0.0518528637919 | False | False |
| 342 | clip_geometry_continuation | 0.001 | 0.0546646270113 | 0.0517927471971 | False | False |
| 342 | multi_branch | 0.01 | 0.0511915374004 | 0.047897670812 | True | False |

Fractions 0.01 and 0.001 are requested diagnostics. Production uses 1, 0.5, 0.25, 0.125, 0.0625 and 0.03125. Every selected generator was also passed to the unmodified production selector with its real six-factor list. The rule-passing diagnostic candidates are not recorded as accepted production trips.

## Stronger candidates exposed by rebuilding the model

135 cells: the best qualified actual-merit predictor is constructed at fraction 0.312477111816 along the original Newton segment, then tested at fraction 1 of its displacement from the stalled state. It reaches merit 0.0130857362076 and residual 0.013409234984; the origin model predicts merit 0.24956358432. Actual sufficient decrease and strict residual decrease pass, but origin-model trust refuses. Its rebuilt branch model predicts 1.6034467397e-15, with predicted incumbent merit 0.107465949249; rebuilt-model acceptance is True. The linear solve is qualified, achieved residual ratio 6.92878055717e-15; endpoint branch consistency is False.

135 cells: 133 classifications in the coarse survey, 133 after doubling; 1436 sampled fractions including transition bisection to 1e-6. All 133 predictors are retained: 133 finite, 126 qualified, 15 with full-endpoint classification matching their construction branch. The selected diagnostic candidate is qualified=True and branch-consistent=False.

The selected 135-cell diagnostic step changes 0 confined/open centroid classifications and 2 nonzero-current supports. Its residual changes from 0.0451931554866 to 0.0450578059084, a reduction of 0.299491 percent. This remains above 1e-12, so converged=false.

342 cells: the best qualified actual-merit predictor is constructed at fraction 0.17126750946 along the original Newton segment, then tested at fraction 1 of its displacement from the stalled state. It reaches merit 0.0260888773763 and residual 0.030815366543; the origin model predicts merit 0.273371979729. Actual sufficient decrease and strict residual decrease pass, but origin-model trust refuses. Its rebuilt branch model predicts 1.68499456721e-15, with predicted incumbent merit 0.133721483303; rebuilt-model acceptance is True. The linear solve is qualified, achieved residual ratio 4.11273078138e-15; endpoint branch consistency is False.

342 cells: 316 classifications in the coarse survey, 316 after doubling; 3732 sampled fractions including transition bisection to 1e-6. All 316 predictors are retained: 316 finite, 306 qualified, 23 with full-endpoint classification matching their construction branch. The selected diagnostic candidate is qualified=True and branch-consistent=False.

The selected 342-cell diagnostic step changes 2 confined/open centroid classifications and 0 nonzero-current supports. Its residual changes from 0.0518511705161 to 0.047897670812, a reduction of 7.62471 percent. This remains above 1e-12, so converged=false.

## Controls, scope and limitations

Both original Newton ladders reproduce as nondecreasing at every requested fraction. The full analytic directions reproduce merits 9.62674026395e-5 and 7.580669184e-5, pass actual decrease and fail model trust. Their one-map residuals are about 9.5e-5, not the locked 1e-12 fixed-point bound.

The support-frozen two-input map equals the production map on its diagonal exactly on both stalled rows. Its Newton direction equals the map defect to rounding precision because the static closure loses its flux-dependent support motion when the polygons are fixed. The classification counter detects a seeded one-bit flip; the support detector sees known nonzero currents. These controls distinguish the uniform refusal ladders from empty or insensitive instruments.

The geometry-continuation arm uses the full original Newton endpoint as its non-oracle support target, then relaxes flux. It does not claim to exhaust all possible support homotopies. The multi-branch census is stable under the stated sampling refinement, not a proof against arbitrarily narrow missed branch excursions. All failed qualification, nonmatching branches, and rejected candidates remain in the raw record.

No source under nova/ changed, no production budget or tolerance changed, no hidden globalization carry was replayed, and no complete cold solve ran. The 342-cell tangent discrepancy remains a separately scoped investigation; it is not fixed or dismissed by these measurements.

## Execution and artifacts

One all_debug allocation, 1275512, used eight CPUs and 96 GiB with a 37-minute scheduler ceiling. Each row ran in a fresh process using the root interpreter directly, JAX_PLATFORMS=cpu, and TMPDIR=/tmp in submit and payload; no uv ran on the compute node. The measurement revision is 2502e08b50f955f45e826ed933522efec6b98bd8. Each raw receipt preserves its input path and SHA-256; compact receipts preserve the raw receipt hash. CPU walls are not GPU timings.

Row process walls: 331.682 seconds (135 cells), 1256.894 seconds (342 cells). The coarse wrapper exits zero. The fine Python process exits zero, independently observed in its zombie process status and preserved in numerical-process-exit.json. Its redundant 1050-second timeout wrapper was suspended to use the already-authorized 37-minute allocation, then resumed after the Python process finished. The queued alarm makes that wrapper return 124 and the launcher return 1; scheduler state is FAILED, exit 1:0, after 27m09s. These nonzero wrapper and scheduler receipts are retained, not represented as a green aggregate gate. The completed numerical receipts and inspected figures establish the measurement. The as-run launcher is preserved; the reproducible launcher now relies on the single scheduler ceiling. Scoped lint, independent receipt arithmetic, changed-cell index reconstruction, hashes, links and append-only metadata checks validate this investigation. No unrelated suite was run.

[Method](method.md), [coarse receipt](cells-135.json), [fine receipt](cells-342.json). The complete fraction tables follow.

## 135 cells

Incumbent merit 0.0399490213834; residual 0.0451931554866.

| Generator | Fraction | Actual merit | Predicted merit | Sup residual | Confined flips | Support changes | Shadow flips | Sufficient merit | Residual decreases | Trust | Accepted | Branch-model accepted | Fallback same-direction rule pass |
|---|---:|---:|---:|---:|---:|---:|---:|---|---|---|---|---|---|
| analytic | 1 | 9.62674026395e-05 | 0.271049525734 | 9.53950481603e-05 | 49 | 47 | 0 | True | True | False | False | False | False |
| analytic | 0.5 | 0.0829338134996 | 0.127744321818 | 0.0822004976243 | 16 | 15 | 0 | False | False | False | False | False | False |
| analytic | 0.25 | 0.0675414910675 | 0.0744716590789 | 0.069404269267 | 2 | 4 | 0 | False | False | False | False | False | False |
| analytic | 0.125 | 0.0529740019793 | 0.0539513332615 | 0.0578825119226 | 2 | 4 | 0 | False | False | False | False | False | False |
| analytic | 0.0625 | 0.045551466247 | 0.0456973687524 | 0.0509915671521 | 0 | 4 | 0 | False | False | False | False | False | False |
| analytic | 0.01 | 0.0405348888033 | 0.040536975571 | 0.0460652137678 | 0 | 0 | 0 | False | False | False | False | False | False |
| analytic | 0.001 | 0.0399993294337 | 0.039999344921 | 0.0452802767812 | 0 | 0 | 0 | False | False | False | False | False | False |
| newton | 1 | 0.60174362648 | 1.33030250718e-14 | 0.558420654403 | 82 | 76 | 0 | False | False | False | False | False | False |
| newton | 0.5 | 0.711833959084 | 0.0421397830362 | 0.735444444087 | 49 | 52 | 11 | False | False | False | False | False | False |
| newton | 0.25 | 0.119955864329 | 0.0444099443283 | 0.131950460227 | 31 | 33 | 0 | False | False | False | False | False | False |
| newton | 0.125 | 0.0517282756248 | 0.042025249145 | 0.0613809420784 | 14 | 13 | 0 | False | False | False | False | False | False |
| newton | 0.0625 | 0.0434960050866 | 0.0409363493088 | 0.0481144471795 | 2 | 5 | 0 | False | False | False | False | False | False |
| newton | 0.01 | 0.0405219232783 | 0.0401001833129 | 0.0455703270762 | 0 | 4 | 0 | False | False | False | False | False | False |
| newton | 0.001 | 0.039964535416 | 0.0399640228673 | 0.0452125716774 | 0 | 0 | 0 | False | False | False | False | False | False |
| map_defect | 1 | 0.0685667974488 | 0.0663279439565 | 0.0631547044272 | 6 | 6 | 0 | False | False | False | False | False | False |
| map_defect | 0.5 | 0.04962369208 | 0.0484225343595 | 0.0503679889736 | 4 | 4 | 0 | False | False | False | False | False | False |
| map_defect | 0.25 | 0.0425569377743 | 0.0422503392582 | 0.0463482157692 | 0 | 2 | 0 | False | False | False | False | False | False |
| map_defect | 0.125 | 0.0406038574545 | 0.0405353118803 | 0.0451473499122 | 0 | 0 | 0 | False | True | False | False | False | False |
| map_defect | 0.0625 | 0.0401127227246 | 0.0400955939627 | 0.0451560456421 | 0 | 0 | 0 | False | True | False | False | False | False |
| map_defect | 0.01 | 0.0399531213746 | 0.0399526919266 | 0.0451850967404 | 0 | 0 | 0 | False | True | False | False | False | False |
| map_defect | 0.001 | 0.0399490551567 | 0.0399490508791 | 0.0451923120658 | 0 | 0 | 0 | False | True | False | False | False | False |
| branch_frozen | 1 | 0.0685667974488 | 0.0663279439565 | 0.0631547044272 | 6 | 6 | 0 | False | False | False | False | False | False |
| branch_frozen | 0.5 | 0.04962369208 | 0.0484225343595 | 0.0503679889736 | 4 | 4 | 0 | False | False | False | False | False | False |
| branch_frozen | 0.25 | 0.0425569377743 | 0.0422503392582 | 0.0463482157692 | 0 | 2 | 0 | False | False | False | False | False | False |
| branch_frozen | 0.125 | 0.0406038574545 | 0.0405353118803 | 0.0451473499122 | 0 | 0 | 0 | False | True | False | False | False | False |
| branch_frozen | 0.0625 | 0.0401127227246 | 0.0400955939627 | 0.0451560456421 | 0 | 0 | 0 | False | True | False | False | False | False |
| branch_frozen | 0.01 | 0.0399531213746 | 0.0399526919266 | 0.0451850967404 | 0 | 0 | 0 | False | True | False | False | False | False |
| branch_frozen | 0.001 | 0.0399490551567 | 0.0399490508791 | 0.0451923120658 | 0 | 0 | 0 | False | True | False | False | False | False |
| clip_geometry_continuation | 1 | 0.779726607268 | 0.410784090533 | 0.791952902642 | 55 | 50 | 5 | False | False | False | False | False | False |
| clip_geometry_continuation | 0.5 | 0.127182065999 | 0.214890603825 | 0.129589892889 | 45 | 49 | 0 | False | False | False | False | False | False |
| clip_geometry_continuation | 0.25 | 0.0731661726904 | 0.102525259058 | 0.0806621880435 | 21 | 23 | 0 | False | False | False | False | False | False |
| clip_geometry_continuation | 0.125 | 0.0640650021812 | 0.065312041901 | 0.072841794558 | 4 | 9 | 0 | False | False | False | False | False | False |
| clip_geometry_continuation | 0.0625 | 0.0513146356561 | 0.0506639560299 | 0.0590247728316 | 2 | 4 | 0 | False | False | False | False | False | False |
| clip_geometry_continuation | 0.01 | 0.0415739149459 | 0.0411367645783 | 0.0473980791324 | 0 | 2 | 0 | False | False | False | False | False | False |
| clip_geometry_continuation | 0.001 | 0.0400532380038 | 0.0400530553494 | 0.0453979135637 | 0 | 0 | 0 | False | False | False | False | False | False |
| multi_branch | 1 | 6.79212735767 | 0.208049889157 | 6.16791171725 | 41 | 49 | 0 | False | False | False | False | False | False |
| multi_branch | 0.5 | 3.31555009038 | 0.195821740899 | 3.00129564093 | 39 | 45 | 0 | False | False | False | False | False | False |
| multi_branch | 0.25 | 1.58426620895 | 0.174956886754 | 1.43749302951 | 39 | 43 | 0 | False | False | False | False | False | False |
| multi_branch | 0.125 | 0.729923303779 | 0.144152941561 | 0.652669201608 | 31 | 39 | 0 | False | False | False | False | False | False |
| multi_branch | 0.0625 | 0.321768873946 | 0.107945064962 | 0.286600224512 | 27 | 28 | 0 | False | False | False | False | False | False |
| multi_branch | 0.01 | 0.0545446676979 | 0.0479095982411 | 0.0677210771609 | 6 | 8 | 0 | False | False | False | False | False | False |
| multi_branch | 0.001 | 0.0391396120225 | 0.039125239917 | 0.0450578059084 | 0 | 2 | 0 | True | True | True | True | True | True |

Branch census: {"accepted_predictor_count": 11, "additional_after_doubling": 0, "best_actual_qualified": {"achieved_reduction": 6.928780557166333e-15, "actual": {"denominator_wb": 80.51597577200894, "merit": 0.01308573620755603, "numerator_wb": 1.0536108194464815, "relative_sup": 0.013409234984016891}, "branch_consistent": false, "branch_model_accepted": true, "branch_model_trust": true, "branch_predicted": {"denominator_wb": 79.56918656576252, "merit": 1.6034467396975445e-15, "numerator_wb": 1.2758495277925757e-13, "relative_sup": 2.073471818565451e-15}, "branch_predicted_incumbent": 0.1074659492487706, "fraction": 1.0, "origin_fraction": 0.31247711181640625, "predicted": {"denominator_wb": 103.76244383437249, "merit": 0.24956358431999887, "numerator_wb": 25.895327401108567, "relative_sup": 0.23266095794019337}, "predictor": 39, "qualification": 1, "rules": {"accepted_if_considered": false, "actual_decrease": 0.026863285175851265, "actual_over_predicted_decrease": -0.1281556243016255, "fallback_note": "Continuation uses these same inequalities; production continuation uses the map defect, measured separately.", "fallback_same_direction_refused": true, "model_trust": false, "predicted_decrease": -0.2096145629365916, "refusing_rules": ["model trust"], "strict_residual_decrease": true, "sufficient_merit": true}}, "best_rebuilt_model_accepted": {"achieved_reduction": 6.928780557166333e-15, "actual": {"denominator_wb": 80.51597577200894, "merit": 0.01308573620755603, "numerator_wb": 1.0536108194464815, "relative_sup": 0.013409234984016891}, "branch_consistent": false, "branch_model_accepted": true, "branch_model_trust": true, "branch_predicted": {"denominator_wb": 79.56918656576252, "merit": 1.6034467396975445e-15, "numerator_wb": 1.2758495277925757e-13, "relative_sup": 2.073471818565451e-15}, "branch_predicted_incumbent": 0.1074659492487706, "fraction": 1.0, "origin_fraction": 0.31247711181640625, "predicted": {"denominator_wb": 103.76244383437249, "merit": 0.24956358431999887, "numerator_wb": 25.895327401108567, "relative_sup": 0.23266095794019337}, "predictor": 39, "qualification": 1, "rules": {"accepted_if_considered": false, "actual_decrease": 0.026863285175851265, "actual_over_predicted_decrease": -0.1281556243016255, "fallback_note": "Continuation uses these same inequalities; production continuation uses the map defect, measured separately.", "fallback_same_direction_refused": true, "model_trust": false, "predicted_decrease": -0.2096145629365916, "refusing_rules": ["model trust"], "strict_residual_decrease": true, "sufficient_merit": true}}, "coarse_branches": 133, "consistent_count": 15, "empirical_not_exhaustive_proof": true, "fine_branches": 133, "finite_count": 133, "predictor_count": 133, "qualified_count": 126, "qualified_incumbent_model_accepted_count": 16, "qualified_rebuilt_model_accepted_count": 26, "sample_count": 1436, "selected_achieved_reduction": 1.0315247745414499e-12, "selected_consistent": false, "selected_qualification": 1, "transition_tolerance": 1e-06}

## 342 cells

Incumbent merit 0.0546254885059; residual 0.0518511705161.

| Generator | Fraction | Actual merit | Predicted merit | Sup residual | Confined flips | Support changes | Shadow flips | Sufficient merit | Residual decreases | Trust | Accepted | Branch-model accepted | Fallback same-direction rule pass |
|---|---:|---:|---:|---:|---:|---:|---:|---|---|---|---|---|---|
| analytic | 1 | 7.58066918431e-05 | 0.291954563011 | 9.59626538977e-05 | 129 | 130 | 0 | True | True | False | False | False | False |
| analytic | 0.5 | 0.0905009356447 | 0.146189409853 | 0.0822639557979 | 44 | 42 | 0 | False | False | False | False | False | False |
| analytic | 0.25 | 0.0828093758503 | 0.0937760406298 | 0.0767436788296 | 17 | 17 | 0 | False | False | False | False | False | False |
| analytic | 0.125 | 0.0701645470727 | 0.0725278639163 | 0.0662279704684 | 10 | 5 | 0 | False | False | False | False | False | False |
| analytic | 0.0625 | 0.0625462915171 | 0.0630831880279 | 0.0594836957753 | 4 | 1 | 0 | False | False | False | False | False | False |
| analytic | 0.01 | 0.0558640743145 | 0.0558771010642 | 0.0531247628995 | 0 | 0 | 0 | False | False | False | False | False | False |
| analytic | 0.001 | 0.0547479076982 | 0.0547480340091 | 0.0519795640322 | 0 | 0 | 0 | False | False | False | False | False | False |
| newton | 1 | 1.30185720015 | 1.62721781102e-14 | 1.21529497825 | 169 | 180 | 13 | False | False | False | False | False | False |
| newton | 0.5 | 0.603350667908 | 0.0372912169441 | 0.551908074239 | 185 | 180 | 0 | False | False | False | False | False | False |
| newton | 0.25 | 0.658808077815 | 0.0812348685874 | 0.683485408986 | 112 | 115 | 5 | False | False | False | False | False | False |
| newton | 0.125 | 0.120857407144 | 0.0677927205829 | 0.129814406606 | 81 | 82 | 0 | False | False | False | False | False | False |
| newton | 0.0625 | 0.0575485735299 | 0.0603930302098 | 0.0612376556909 | 35 | 37 | 0 | False | False | False | False | False | False |
| newton | 0.01 | 0.0550399712941 | 0.0554509947636 | 0.0522584144563 | 8 | 2 | 0 | False | False | False | False | False | False |
| newton | 0.001 | 0.0547009892462 | 0.0547065229243 | 0.0519273483914 | 0 | 0 | 0 | False | False | False | False | False | False |
| map_defect | 1 | 0.0658686058668 | 0.0621796261681 | 0.0634882131002 | 9 | 8 | 0 | False | False | False | False | False | False |
| map_defect | 0.5 | 0.0573522193484 | 0.0565249159687 | 0.0537247974157 | 7 | 5 | 0 | False | False | False | False | False | False |
| map_defect | 0.25 | 0.0553654507073 | 0.0551788978621 | 0.0524276025451 | 5 | 5 | 0 | False | False | False | False | False | False |
| map_defect | 0.125 | 0.0548466373685 | 0.054800633339 | 0.0521039708013 | 2 | 2 | 0 | False | False | False | False | False | False |
| map_defect | 0.0625 | 0.0546971151422 | 0.0546859286582 | 0.0519670393034 | 2 | 1 | 0 | False | False | False | False | False | False |
| map_defect | 0.01 | 0.0546315220939 | 0.0546312795954 | 0.051868302097 | 1 | 0 | 0 | False | False | False | False | False | False |
| map_defect | 0.001 | 0.0546260008946 | 0.0546259984847 | 0.0518528637919 | 1 | 0 | 0 | False | False | False | False | False | False |
| branch_frozen | 1 | 0.0658686058668 | 0.0621796261681 | 0.0634882131002 | 9 | 8 | 0 | False | False | False | False | False | False |
| branch_frozen | 0.5 | 0.0573522193484 | 0.0565249159687 | 0.0537247974157 | 7 | 5 | 0 | False | False | False | False | False | False |
| branch_frozen | 0.25 | 0.0553654507073 | 0.0551788978621 | 0.0524276025451 | 5 | 5 | 0 | False | False | False | False | False | False |
| branch_frozen | 0.125 | 0.0548466373685 | 0.054800633339 | 0.0521039708013 | 2 | 2 | 0 | False | False | False | False | False | False |
| branch_frozen | 0.0625 | 0.0546971151422 | 0.0546859286582 | 0.0519670393034 | 2 | 1 | 0 | False | False | False | False | False | False |
| branch_frozen | 0.01 | 0.0546315220939 | 0.0546312795954 | 0.051868302097 | 1 | 0 | 0 | False | False | False | False | False | False |
| branch_frozen | 0.001 | 0.0546260008946 | 0.0546259984847 | 0.0518528637919 | 1 | 0 | 0 | False | False | False | False | False | False |
| clip_geometry_continuation | 1 | 0.46955151216 | 0.397345216011 | 0.661920736338 | 172 | 185 | 10 | False | False | False | False | False | False |
| clip_geometry_continuation | 0.5 | 0.372245391197 | 0.279971269717 | 0.541393616172 | 111 | 120 | 5 | False | False | False | False | False | False |
| clip_geometry_continuation | 0.25 | 0.198294057654 | 0.135197840523 | 0.237871924103 | 50 | 49 | 4 | False | False | False | False | False | False |
| clip_geometry_continuation | 0.125 | 0.0744385074498 | 0.0762090333841 | 0.0869175833343 | 34 | 31 | 0 | False | False | False | False | False | False |
| clip_geometry_continuation | 0.0625 | 0.0587462993115 | 0.061475892623 | 0.0641857906573 | 18 | 11 | 0 | False | False | False | False | False | False |
| clip_geometry_continuation | 0.01 | 0.0549433992817 | 0.0551323334157 | 0.0510661513109 | 5 | 1 | 0 | False | True | False | False | False | False |
| clip_geometry_continuation | 0.001 | 0.0546646270113 | 0.0546669883576 | 0.0517927471971 | 0 | 0 | 0 | False | True | False | False | False | False |
| multi_branch | 1 | 1.77120142927 | 1.52361826656 | 2.09896512385 | 48 | 51 | 0 | False | False | False | False | False | False |
| multi_branch | 0.5 | 0.952381577308 | 0.891716716334 | 1.13990556068 | 33 | 31 | 0 | False | False | False | False | False | False |
| multi_branch | 0.25 | 0.4975617042 | 0.484954449031 | 0.600830266975 | 20 | 18 | 0 | False | False | False | False | False | False |
| multi_branch | 0.125 | 0.257648580615 | 0.256595014842 | 0.313129448285 | 9 | 12 | 0 | False | False | False | False | False | False |
| multi_branch | 0.0625 | 0.135711880228 | 0.136557466223 | 0.16559082381 | 7 | 8 | 0 | False | False | False | False | False | False |
| multi_branch | 0.01 | 0.0511915374004 | 0.0512071343734 | 0.047897670812 | 2 | 0 | 0 | True | True | True | True | True | True |
| multi_branch | 0.001 | 0.054174407513 | 0.0541746391012 | 0.0514563597304 | 0 | 0 | 0 | True | True | True | True | True | True |

Branch census: {"accepted_predictor_count": 19, "additional_after_doubling": 0, "best_actual_qualified": {"achieved_reduction": 4.112730781378902e-15, "actual": {"denominator_wb": 92.35903535879845, "merit": 0.026088877376311992, "numerator_wb": 2.409543548070156, "relative_sup": 0.030815366542958467}, "branch_consistent": false, "branch_model_accepted": true, "branch_model_trust": true, "branch_predicted": {"denominator_wb": 90.13486935293467, "merit": 1.6849945672075353e-15, "numerator_wb": 1.518767651756559e-13, "relative_sup": 2.481690463632997e-15}, "branch_predicted_incumbent": 0.13372148330288702, "fraction": 1.0, "origin_fraction": 0.17126750946044922, "predicted": {"denominator_wb": 121.90291414475554, "merit": 0.273371979729171, "numerator_wb": 33.324840974506984, "relative_sup": 0.2628866581008315}, "predictor": 119, "qualification": 1, "rules": {"accepted_if_considered": false, "actual_decrease": 0.028536611129582355, "actual_over_predicted_decrease": -0.1304551719664146, "fallback_note": "Continuation uses these same inequalities; production continuation uses the map defect, measured separately.", "fallback_same_direction_refused": true, "model_trust": false, "predicted_decrease": -0.21874649122327663, "refusing_rules": ["model trust"], "strict_residual_decrease": true, "sufficient_merit": true}}, "best_rebuilt_model_accepted": {"achieved_reduction": 4.112730781378902e-15, "actual": {"denominator_wb": 92.35903535879845, "merit": 0.026088877376311992, "numerator_wb": 2.409543548070156, "relative_sup": 0.030815366542958467}, "branch_consistent": false, "branch_model_accepted": true, "branch_model_trust": true, "branch_predicted": {"denominator_wb": 90.13486935293467, "merit": 1.6849945672075353e-15, "numerator_wb": 1.518767651756559e-13, "relative_sup": 2.481690463632997e-15}, "branch_predicted_incumbent": 0.13372148330288702, "fraction": 1.0, "origin_fraction": 0.17126750946044922, "predicted": {"denominator_wb": 121.90291414475554, "merit": 0.273371979729171, "numerator_wb": 33.324840974506984, "relative_sup": 0.2628866581008315}, "predictor": 119, "qualification": 1, "rules": {"accepted_if_considered": false, "actual_decrease": 0.028536611129582355, "actual_over_predicted_decrease": -0.1304551719664146, "fallback_note": "Continuation uses these same inequalities; production continuation uses the map defect, measured separately.", "fallback_same_direction_refused": true, "model_trust": false, "predicted_decrease": -0.21874649122327663, "refusing_rules": ["model trust"], "strict_residual_decrease": true, "sufficient_merit": true}}, "coarse_branches": 316, "consistent_count": 23, "empirical_not_exhaustive_proof": true, "fine_branches": 316, "finite_count": 316, "predictor_count": 316, "qualified_count": 306, "qualified_incumbent_model_accepted_count": 29, "qualified_rebuilt_model_accepted_count": 58, "sample_count": 3732, "selected_achieved_reduction": 2.5306347251174306e-14, "selected_consistent": false, "selected_qualification": 1, "transition_tolerance": 1e-06}


The fallback column applies the unchanged continuation inequalities to the listed candidate. Production continuation proposes the map defect, whose complete ladder is shown separately; no hidden recovery radius is replayed.
