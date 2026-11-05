"""Run the existing bank measurement and retain its read counters."""

import json
from pathlib import Path
import sys

from scripts.prototypes.bank_solve_cycle import measure


def main():
    output = Path(sys.argv[sys.argv.index("--output") + 1])
    original_wait = measure.jax.block_until_ready
    original_emit = measure.reference.emit
    latest = []
    reads = []

    def wait(value):
        result = original_wait(value)
        if isinstance(result, measure.support_read.SolveResult):
            latest[:] = [result]
        return result

    def emit(label, **values):
        if label == "SOLVE":
            counts = measure.np.asarray(latest[0].reads).tolist()
            reads.append(counts)
            values["reads"] = counts
        original_emit(label, **values)

    measure.jax.block_until_ready = wait
    measure.reference.emit = emit
    try:
        measure.main()
    finally:
        if output.exists():
            receipt = json.loads(output.read_text())
            for row, counts in zip(receipt["rows"], reads, strict=True):
                row["reads"] = counts
            output.write_text(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
