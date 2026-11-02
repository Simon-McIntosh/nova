"""Redraw one certificate row's panel from its part, without re-running the solve."""

# ruff: noqa: E402

from nova.jax.config import configure_dtypes

configure_dtypes()

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import certificate_panel


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--part", type=Path, required=True)
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument(
        "--raw",
        action="store_true",
        help="draw the driver's labels as composed and report why they fail",
    )
    args = parser.parse_args()
    part = json.loads(args.part.read_text())
    if not args.raw:
        checked = certificate_panel.draw(
            part, args.panel, certificate_panel.row_title(part)
        )
        print(f"panel={args.panel} labels_checked={checked}")
        return
    from benchmarks import solovev_certificate as certificate

    raw = args.panel.with_name(args.panel.stem + "-unguarded.png")
    figure = certificate._plot(
        *certificate_panel.panel_arrays(part),
        raw,
        certificate_panel.row_title(part),
    )
    try:
        certificate_panel.check_labels(figure)
    except ValueError as error:
        print(f"UNCLEARED {error}")
        raise SystemExit(1) from error
    finally:
        plt.close(figure)
        for leftover in (raw, raw.with_suffix(".svg")):
            leftover.unlink(missing_ok=True)
    print("CLEARED")
    raise SystemExit(0)


if __name__ == "__main__":
    main()
