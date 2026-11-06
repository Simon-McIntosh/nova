"""Book one diverted bank state and record its receipt fields.

Records the six moments of the diverted fixture, their byte digests, and the
cells whose boundary uses an algebraic-conic cut interval, so two revisions can
be compared cell by cell. Run with the revision's tree first on PYTHONPATH.
"""

import argparse
import hashlib
import json

# ruff: noqa: E402
from nova.jax.config import configure_dtypes

configure_dtypes()

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    options = parser.parse_args()

    import jax

    import nova
    from tests.test_support_read import _case

    print("nova.__file__", nova.__file__, flush=True)
    print("jax x64", jax.config.jax_enable_x64, flush=True)
    print("platforms", jax.default_backend(), flush=True)

    operands, cache, roots, pitch, step, _, _ = _case("diverted")
    support, booked = step(operands, cache, roots + 0.03 * pitch, roots)
    arrays = {name: np.asarray(getattr(booked, name)) for name in booked._fields}
    selected = np.asarray(support.reading.fragment_selected)
    conic_cells = np.any(selected, axis=-1) if selected.ndim > 1 else selected != 0
    digests = {
        name: hashlib.sha256(value.tobytes()).hexdigest()
        for name, value in arrays.items()
    }
    np.savez(
        options.out,
        conic_cells=conic_cells,
        **{f"field_{name}": value for name, value in arrays.items()},
    )
    report = {
        "fields": list(booked._fields),
        "digests": digests,
        "shapes": {name: list(value.shape) for name, value in arrays.items()},
    }
    report["conic_cell_count"] = int(np.count_nonzero(conic_cells))
    report["conic_cells"] = np.flatnonzero(conic_cells).tolist()
    print("BOOK_REPORT", json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
