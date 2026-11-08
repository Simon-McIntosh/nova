"""Require close spectrum samples to retain their visible multiplicity."""

import argparse
import importlib.util
from pathlib import Path
from unittest.mock import patch

import numpy as np


def check(source, receipt, output, png):
    spec = importlib.util.spec_from_file_location("spectrum_plot", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    save = module.checked_save

    def inspect(fig, path):
        fig.canvas.draw()
        ax = fig.axes[1]
        line = ax.lines[0]
        points = np.column_stack(line.get_data())
        centres = ax.transData.transform(points)
        separation = np.linalg.norm(centres[-1] - centres[-2])
        diameter = line.get_markersize() * fig.dpi / 72
        represented = any(t.get_text() == "2 states" for t in ax.texts)
        print(
            f"TERMINAL_PAIR separation_px={separation:.6f} "
            f"marker_diameter_px={diameter:.6f} count_label={represented}",
            flush=True,
        )
        assert separation >= diameter or represented, (
            "terminal pair has no visible multiplicity"
        )
        save(fig, path)

    module.checked_save = inspect
    argv = [str(source), str(receipt), str(output), "--png", str(png)]
    with patch("sys.argv", argv):
        module.main()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--png", type=Path, required=True)
    args = parser.parse_args()
    check(args.source, args.receipt, args.output, args.png)


if __name__ == "__main__":
    main()
