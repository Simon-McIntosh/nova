"""Build batch-rule.json from the batching-rule gate receipts beside this file."""

import json
import re
from pathlib import Path

D = Path(__file__).parent


def load(name):
    return json.loads((D / name).read_text())


def cost(name):
    receipt = load(name)
    arm = receipt["arm"]
    return dict(
        job=re.search(
            r"job=(\d+)", (D / name.replace(".json", ".log")).read_text()
        ).group(1),
        backend=receipt["backend"],
        arm=arm,
        median_seconds=receipt["walls"][arm]["median_seconds"],
        base_median_seconds=receipt["walls"]["base"]["median_seconds"],
        ratio_over_base=receipt["ratio_over_base"],
    )


def terminal(log):
    return float(
        re.search(
            r"TERMINAL arm=\w+ residual=([0-9.e+-]+)", (D / log).read_text()
        ).group(1)
    )


program = load("program-rule-300.json")
exit_head_terminal = 0.06325730373091486
rule_terminal = terminal("diverted-trace-rule-plain.log")
receipt = dict(
    revision=program["revision"],
    vmapped_width16_h200=dict(
        rule_first=cost("batch-rule-cost-h200-first.json"),
        rule_second=cost("batch-rule-cost-h200.json"),
        carry_selecting_control_first=cost(
            "batch-rule-negative-control-h200-first.json"
        ),
        carry_selecting_control_second=cost("batch-rule-negative-control-h200.json"),
    ),
    vmapped_width16_cpu=dict(
        rule=cost("batch-rule-cost-cpu.json"),
        carry_selecting_control=cost("batch-rule-negative-control-cpu.json"),
    ),
    unbatched_optimised_instructions_300=program["optimized_instructions"],
    census_job=program["job_id"],
    exit_loop_instructions_300=load("program-exit-300.json")["optimized_instructions"],
    diverted_terminal=dict(
        rule=rule_terminal,
        exit_loop_head=exit_head_terminal,
        difference=abs(rule_terminal - exit_head_terminal),
        bound=3.07e-12,
    ),
    toy_identity=re.search(
        r"RULE_TOY_IDENTITY (\d+/\d+)", (D / "rule-toy-identity.log").read_text()
    ).group(1),
    tests={
        name: re.findall(r"=+ (.*) in [0-9.]+s", (D / f"rule-{name}.log").read_text())[
            -1
        ]
        for name in ("single_site_krylov", "fixed_point", "reduced_newton")
    },
)
(D / "batch-rule.json").write_text(json.dumps(receipt, indent=2) + "\n")
print(json.dumps(receipt, indent=2))
