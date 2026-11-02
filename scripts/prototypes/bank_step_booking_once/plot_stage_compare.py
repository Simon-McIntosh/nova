"""Draw the booking-tangent and whole-step walls for the two source arms."""

import argparse
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

STAGES = {"local_booking_tangent": "booking tangent", "whole_step": "whole step"}


def read(path, name):
    values = {}
    for line in Path(path).read_text().splitlines():
        match = re.match(rf"STAGE_DEVICE name={name} wall_ms=([0-9.]+)", line)
        if match:
            values[name] = float(match.group(1))
    return values


def main(base, head, out):
    base_values = read(base, "base")
    head_values = read(head, "head")
    plt.style.use("data-ink")
    figure, axes = plt.subplots(1, 2, figsize=(14, 7.2), layout="constrained")
    for axis, (key, label) in zip(axes, STAGES.items(), strict=True):
        arms = ("base", "collapsed")
        values = (base_values[key], head_values[key])
        bars = axis.bar(arms, values, color=("#929292", "#2e8b57"), width=0.55)
        axis.bar_label(bars, fmt="%.2f ms", padding=3)
        axis.set_title(label)
        axis.set_ylabel("warm wall [ms]")
        axis.set_ylim(0, max(values) * 1.25)
        for spine in ("top", "right"):
            axis.spines[spine].set_visible(False)
    figure.savefig(out)
    print(json.dumps({"base": base_values, "collapsed": head_values}, sort_keys=True))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True)
    parser.add_argument("--head", required=True)
    parser.add_argument("--out", required=True)
    main(**vars(parser.parse_args))