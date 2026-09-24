"""Build vmap-exit.json from the exit-loop job receipts beside slice-one's."""

import json
from pathlib import Path

D = Path(__file__).parent
NAMES = ("_qualified_krylov_step", "_single_site_krylov")


def site_share(callers_path):
    """Live-map sites whose solver call chains pass through the qualified step."""
    rows = json.loads(Path(callers_path).read_text())
    sites = [
        row
        for row in rows
        if any(any(n in f for f in chain for n in NAMES) for chain, _ in row["callers"])
    ]
    return dict(
        sites=len(sites),
        instructions=sum(row["instructions"] for row in sites),
        rows=[dict(instructions=r["instructions"], kinds=r["kinds"]) for r in sites],
    )


def load(name):
    return json.loads((D / name).read_text())


def cost(name):
    receipt = load(name)
    arm = receipt["arm"]
    return dict(
        backend=receipt["backend"],
        arm=arm,
        median_seconds=receipt["walls"][arm]["median_seconds"],
        base_median_seconds=receipt["walls"]["base"]["median_seconds"],
        ratio_over_base=receipt["ratio_over_base"],
        member_applications_first_four=receipt["member_applications_first_four"],
    )


program = load("program-exit-300.json")
receipt = dict(
    revision=program["revision"],
    unbatched_optimised_instructions_300=program["optimized_instructions"],
    compile_seconds_300=program["compile_seconds"],
    census_job=program["job_id"],
    slice_one_instructions_300=load("program-stream-300.json")[
        "optimized_instructions"
    ],
    qualified_step_live_map_sites_300=site_share(D / "callers-exit-300.json"),
    slice_one_vmapped_width16_cpu=dict(stream_seconds=1.240, base_seconds=0.086),
    vmapped_width16=dict(
        exit_h200=cost("vmap-exit-cost-h200.json"),
        exit_cpu=cost("vmap-exit-cost-cpu.json"),
        restored_scan_h200=cost("vmap-exit-negative-control-h200.json"),
        restored_scan_cpu=cost("vmap-exit-negative-control-cpu.json"),
        consumer_gated_h200=cost("vmap-exit-cost-gated-h200.json"),
        consumer_gated_cpu=cost("vmap-exit-cost-gated-cpu.json"),
    ),
)
(D / "vmap-exit.json").write_text(json.dumps(receipt, indent=2) + "\n")
print(json.dumps(receipt, indent=2))
