"""Draw selected bank stages on a linear wall-time axis."""

from __future__ import annotations

import argparse
from pathlib import Path

from scripts.prototypes.bank_step_speed.plot_stage import main as plot_stages

STAGES = (
    "read",
    "residual_and_booking",
    "local_booking_tangent",
    "geometry_tangent_one_direction",
    "whole_step",
)
LABELS = {
    "read": "read",
    "residual_and_booking": "residual and booking",
    "local_booking_tangent": "booking tangent",
    "geometry_tangent_one_direction": "moving geometry",
    "whole_step": "whole step",
}


def main(base: str, head: str, output: str, png: str) -> None:
    plot_stages(
        Path(base),
        Path(head),
        Path(output),
        Path(png),
        "base",
        "bucketed",
        axis_scale="linear",
        selected_stages=STAGES,
        stage_labels=LABELS,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True)
    parser.add_argument("--head", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--png", required=True)
    main(**vars(parser.parse_args()))
