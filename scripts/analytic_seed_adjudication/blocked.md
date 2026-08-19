NEEDS-HELP: The production coarse fixture cannot construct its authoritative pre-clip sample targets.

tried: Ran the production fixture constructor once through the restart-ready adjudication harness. After the 566-cell centre coupling completed, `PlasmaGrid._build_direct_samples` raised `ValueError: a hex plasma grid needs at least one complete generator cell`; no seed defect or solve was evaluated.

options: (1) extend scope to `nova/biot/plasmagrid.py` and its tests, wire the section metadata correctly, and replace the one-canonical-offset assumption with the already-measured fixed per-cell tiling construction; (2) supply a lineage commit that already carries that constructor repair, then rerun this node unchanged; (3) authorize a measurement-only constructor monkeypatch, which would produce numbers but would not measure the production callable contracted by the node.

leaning: Option 1, because the constructor is production authority and the existing implementation conflicts with the banked size-graded-tiling measurement: offset spread 0.304 m versus a 2.37e-13 m round-off bound.

cost-if-wrong: The coarse coupling build already cost 3 minutes 31 seconds. Choosing a non-authoritative workaround would require discarding its results and rebuilding both the 566-cell and 1069-cell fixtures, historically about 20 minutes before the six direct solves.
