"""Reuse the solve trace while recording allocator peaks beside each batch."""

import json
from pathlib import Path
import sys

from scripts.prototypes.bank_wall_events import capture


def main():
    destination = Path(sys.argv[sys.argv.index("--run") + 1])
    emit = capture.emit
    memory = {}

    def record(label, **values):
        emit(label, **values)
        if label == "SOLVE":
            stats = capture.jax.devices()[0].memory_stats()
            assert stats["peak_bytes_in_use"] > 0
            memory[str(values["lanes"])] = stats
            (destination / "memory.json").write_text(json.dumps(memory, indent=2))
            emit("MEMORY", lanes=values["lanes"], **stats)

    capture.emit = record
    import nova.equilibrium.current as current

    print("MODULE", current.__file__, flush=True)
    capture.main()


if __name__ == "__main__":
    main()
