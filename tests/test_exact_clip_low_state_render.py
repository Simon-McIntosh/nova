"""Assert the terminal-panel titles state every level and every panel flag.

The render receipt the discriminator writes beside the figures is the artifact path
here. It records, per figure, every title line drawn on each panel and, per panel, the
count of null glyphs drawn for each null set, so the two defects this module guards
against are checkable from the receipt alone:

* the stated level array is printed in full -- the recovered token count matches the
  declared count, and the last recovered token is the last level, so no value is cut
  off by the panel width;
* every panel prints its terminal residual and converged flag, equal to the values the
  receipt records for that panel's arm.

The negative control is declared in the module docstring and applied by running this
file as a script: it truncates the final level value exactly as the panel width did,
and requires the check to reject the mutated receipt.
"""

from __future__ import annotations

import copy
import json
import re
from pathlib import Path

import numpy as np

from benchmarks import exact_clip_low_state_discriminator as discriminator

FIGURE_NAMES = ("weak-static-110-terminals.png", "weak-static-300-terminals.png")

RECEIPT_PATH = discriminator.DEFAULT_OUTPUT_ROOT / discriminator.RENDER_RECEIPT_NAME

_LEVEL_LABELS = ("shared levels", "fixed levels")
_COUNT_PATTERN = re.compile(r"^(?P<label>.+?) \((?P<count>\d+)\): (?P<first>.*)$")


def _parse_level_block(lines: list[str], label: str) -> tuple[int, np.ndarray]:
    """Recover the declared count and every stated level from the title lines."""
    match = _COUNT_PATTERN.match(lines[0])
    if match is None or not match.group("label").startswith(label):
        raise AssertionError(
            f"panel title does not open the {label!r} block: {lines[0]!r}"
        )
    tokens = [token for token in match.group("first").split(",")]
    for line in lines[1:]:
        tokens.extend(line.split(","))
    values = np.array(
        [float(token) for token in tokens if token.strip()], dtype=np.float64
    )
    return int(match.group("count")), values


def _level_block(lines: list[str], label: str) -> list[str]:
    """Return the maximal run of lines after the first that carries only levels."""
    block = [lines[0]]
    for line in lines[1:]:
        if _LEVEL_LABELS[0] in line or _LEVEL_LABELS[1] in line:
            break
        block.append(line)
    return block


def _panel_problems(figure: dict) -> list[str]:
    """Return every way one figure's panels fail the two stated conditions."""
    problems: list[str] = []
    title_lines = list(figure["title_lines"])
    for line in title_lines:
        if "residual=" in line or _LEVEL_LABELS[0] in line or _LEVEL_LABELS[1] in line:
            continue
        problems.append(f"unrecognised title line: {line!r}")
    if not title_lines:
        problems.append("figure records no title lines at all")
        return problems

    for panel in figure["panels"]:
        state = panel["state"]
        for kind, label, expected in (
            (
                "overlay_title_lines",
                _LEVEL_LABELS[0],
                np.asarray(figure["shared_flux_levels_wb"], dtype=np.float64),
            ),
            (
                "difference_title_lines",
                _LEVEL_LABELS[1],
                np.asarray(
                    figure["difference_levels_fraction_of_span"], dtype=np.float64
                ),
            ),
        ):
            lines = list(panel[kind])
            flags = [line for line in lines if "residual=" in line]
            if not lines:
                problems.append(f"{state}: {label} panel records no title lines")
                continue
            if not flags:
                problems.append(
                    f"{state}: {label} panel prints no residual and converged"
                )
                continue
            residual = float(panel["terminal_residual"])
            converged = bool(panel["converged"])
            for flag in flags:
                if not flag.startswith(
                    f"residual={residual:.3e}; converged={converged}"
                ):
                    problems.append(
                        f"{state}: {label} panel flag {flag!r} does not match "
                        f"residual={residual:.3e}; converged={converged}"
                    )
            block = _level_block(lines[lines.index(flags[0]) + 1 :], label)
            if not block:
                problems.append(f"{state}: {label} panel states no level array")
                continue
            count, values = _parse_level_block(block, label)
            if count != len(expected):
                problems.append(
                    f"{state}: {label} declares {count} levels, but the "
                    f"receipt holds {len(expected)}"
                )
            if len(values) != count:
                problems.append(
                    f"{state}: {label} states {len(values)} terms for a declared count "
                    f"of {count} -- a term was cut off the panel"
                )
            wanted = [f"{float(level):.6g}" for level in expected]
            stated = [
                token.strip()
                for line in block
                for token in line.split(",")
                if token.strip()
            ]
            if stated[: len(values)] != wanted[: len(values)]:
                problems.append(
                    f"{state}: {label} stated {stated} against expected {wanted}"
                )
            if values.size and expected.size and values[-1] != expected[-1]:
                problems.append(
                    f"{state}: {label} last level {values[-1]!r} is not "
                    f"the last level "
                    f"{expected[-1]!r}"
                )
        for null_set in ("analytic", "solved"):
            for panel_kind, drawn in panel["nulls_drawn_by_panel"].items():
                if not drawn.get(null_set, {}).get("x_points_drawn"):
                    problems.append(
                        f"{state}: {panel_kind} panel drew no {null_set} null glyph"
                    )
    return problems


def _receipt() -> dict:
    if not RECEIPT_PATH.is_file():
        raise AssertionError(
            f"{RECEIPT_PATH} is absent; render the panels through the driver's "
            "render-only entry point before this check"
        )
    return json.loads(RECEIPT_PATH.read_text(encoding="utf-8"))


def test_receipt_records_the_render_only_route():
    receipt = _receipt()
    assert receipt["render_only"] is True, (
        "the receipt did not come from a solve-free render"
    )
    assert [Path(figure["figure"]).name for figure in receipt["figures"]] == list(
        FIGURE_NAMES
    )


def test_every_panel_states_the_full_level_array_and_its_flags():
    receipt = _receipt()
    problems = [
        problem for figure in receipt["figures"] for problem in _panel_problems(figure)
    ]
    assert not problems, "\n".join(problems)


def test_truncated_level_text_is_rejected():
    """The check accepts the intact receipt and rejects a truncated level value.

    The mutation reproduces the panel-width cut-off this module exists to guard: the
    final level value loses its last two characters, so the recovered token count falls
    below the declared count and the last stated level is no longer the last level.
    """
    receipt = _receipt()
    clean = [
        problem for figure in receipt["figures"] for problem in _panel_problems(figure)
    ]
    assert clean == [], "\n".join(clean)
    mutated = _mutate_last_level_value(receipt)
    problems = [
        problem for figure in mutated["figures"] for problem in _panel_problems(figure)
    ]
    assert problems, "the truncated receipt passed the check"


def _mutate_last_level_value(problems_source: dict) -> dict:
    """Drop the final level's last two characters, as the panel width did."""
    mutated = copy.deepcopy(problems_source)
    figure = mutated["figures"][0]
    panel = figure["panels"][0]
    lines = panel["difference_title_lines"]
    lines[-1] = lines[-1][:-2]
    return mutated


def main() -> int:
    print(
        "DECLARED NEGATIVE CONTROL: truncate the final level value in the "
        "first figure's difference-panel title by two characters, reproducing "
        "the panel-width cut-off."
    )
    receipt = _receipt()
    clean = [
        problem for figure in receipt["figures"] for problem in _panel_problems(figure)
    ]
    if clean:
        print("RECEIPT ALREADY FAILS THE CHECK; the control cannot discriminate:")
        print("\n".join(clean))
        return 1
    mutated = _mutate_last_level_value(receipt)
    problems = [
        problem for figure in mutated["figures"] for problem in _panel_problems(figure)
    ]
    if not problems:
        print("NEGATIVE CONTROL DID NOT FAIL THE CHECK")
        return 1
    print("\n".join(problems))
    print("NEGATIVE_CONTROL_FAILED_AS_EXPECTED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
