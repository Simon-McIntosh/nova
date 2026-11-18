"""Count outlined loop bodies and their instantiations in lowered StableHLO.

Definitions and call sites are distinct: a fully unrolled scan can call one
outlined definition twice, which still gives XLA two copies to inline. A
rolled scan instead calls its definition once from a while body.
"""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

from scripts.prototypes.point_program_attribution.bodies import (
    CALL,
    count_calls,
    count_function_bodies,
    functions,
    values,
)


def inspect(text):
    spans = dict(functions(text))
    assert "main" in spans and len(spans) > 1, "no outlined program observed"
    callers = {}
    for name, lines in spans.items():
        for callee, count in Counter(CALL.findall("\n".join(lines))).items():
            callers.setdefault(callee, {})[name] = count
    scopes = {name: values(lines) for name, lines in spans.items()}
    closed = [
        {
            "body": name,
            "callers": callers.get(name, {}),
            "call_sites": sum(callers.get(name, {}).values()),
            "operations": len(scopes[name][0]),
            "signature": lines[0].strip(),
        }
        for name, lines in spans.items()
        if name.startswith("closed_call")
    ]
    assert closed, "counter must see the known-present scan bodies"
    return {
        "stablehlo_bytes": len(text.encode()),
        "sha256": hashlib.sha256(text.encode()).hexdigest(),
        "operations": sum(len(body) for body, _ in scopes.values()),
        "calls": count_calls(text),
        "function_bodies": count_function_bodies(scopes),
        "while_sites": {
            name: sum("stablehlo.while(" in line for line in lines)
            for name, lines in spans.items()
            if any("stablehlo.while(" in line for line in lines)
        },
        "closed_bodies": closed,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("module", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    receipt = inspect(args.module.read_text())
    receipt["module"] = str(args.module)
    args.output.write_text(json.dumps(receipt, indent=2) + "\n")
    print(
        json.dumps({key: value for key, value in receipt.items() if key != "calls"}),
        flush=True,
    )


if __name__ == "__main__":
    main()
