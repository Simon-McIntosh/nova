"""Show that the figure's artifact checks fire on a figure that breaks them.

The plot writes nothing if a text artist leaves the canvas, two text artists
overlap, or a text artist covers plotted data. A check that cannot fail proves
nothing, so this control draws the same figure twice: once as delivered, and
once with a label placed deliberately on the residual ladder. It requires the
first to pass and the second to raise, and it writes no figure either way.
"""

import argparse
import sys
import tempfile
from pathlib import Path

from scripts.prototypes.bank_solve_lanes import plot


def run(receipt, output, break_it):
    """Run the plot, optionally labelling a point that sits on the data."""
    original = plot.save_checked

    def wrapped(fig, path):
        if break_it:
            axis = fig.axes[0]
            axis.text(
                9.0,
                1e-8,
                "deliberate overlap",
                fontsize=14,
                color="#c0442b",
                zorder=9,
            )
        return original(fig, path)

    plot.save_checked = wrapped
    sys.argv = ["plot.py", str(receipt), "--output", str(output)]
    try:
        plot.main()
    finally:
        plot.save_checked = original


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("receipt", type=Path)
    args = parser.parse_args()
    with tempfile.TemporaryDirectory() as scratch:
        delivered = Path(scratch) / "delivered.svg"
        run(args.receipt, delivered, break_it=False)
        print(f"CONTROL delivered figure written: {delivered.exists()}", flush=True)
        assert delivered.exists(), "the delivered figure was not written"
        broken = Path(scratch) / "broken.svg"
        try:
            run(args.receipt, broken, break_it=True)
        except ValueError as error:
            print(f"CONTROL broken figure refused: {error}", flush=True)
        else:
            raise SystemExit("the overlap check did not fire on a broken figure")
        assert not broken.exists(), "a broken figure was written"
    print("CONTROL ok", flush=True)


if __name__ == "__main__":
    main()
