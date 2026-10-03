#!/usr/bin/env python3
"""Verify each boundary-wall-proximity caption against its own receipt.

Run from the repository root or from anywhere; the fragment defaults to
``docs/evidence/fragments/playable-forward-solve/pfs-boundary-wall-proximity-captions.html``
and every receipt is resolved beside this script. Every number a caption states is
written as ``<code>.json.dot.path = value</code>``; the check resolves that path in the
panel's own ``boundary-wall-proximity.json`` and fails if any caption number disagrees.
It also fails when a caption carries no bindings or a title does not name its gate.

Usage:
    python check_boundary_wall_proximity_captions.py [--fragment PATH]

Exit status is 0 only when every check passes; the last line is ``EXIT=<code>``.
"""

from __future__ import annotations

import argparse
import html
import json
import math
import re
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[3]
DEFAULT_FRAGMENT = REPO_ROOT / (
    "docs/evidence/fragments/playable-forward-solve/"
    "pfs-boundary-wall-proximity-captions.html"
)

# Each panel's expected gate token and the phrase its title must carry. The title
# names which gate produced the census; the token is the machine-checkable identity.
PANELS: dict[str, dict[str, str]] = {
    "area-evidence": {
        "gate_token": "without-explicit-branch-gate",
        "gate_phrase": "without the explicit branch gate",
    },
    "area-evidence-explicit-gate": {
        "gate_token": "explicit-branch-gate",
        "gate_phrase": "behind the explicit branch gate",
    },
}

FIGURE_RE = re.compile(r"<figure\b([^>]*)>(.*?)</figure>", re.DOTALL | re.IGNORECASE)
ATTR_RE = re.compile(r'data-(panel|gate)\s*=\s*"([^"]*)"')
H3_RE = re.compile(r"<h[1-6]\b[^>]*>(.*?)</h[1-6]>", re.DOTALL | re.IGNORECASE)
CAPTION_RE = re.compile(
    r"<figcaption\b[^>]*>(.*?)</figcaption>", re.DOTALL | re.IGNORECASE
)
TAG_RE = re.compile(r"<[^>]+>")
# A binding is a dotted JSON path, `=`, and a JSON scalar. The leading dot is the
# receipt root; the path must contain at least one dot so prose cannot match it.
BINDING_RE = re.compile(
    r"(\.[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*)"
    r"\s*=\s*"
    r'("[^"]*"|-?(?:[0-9]+\.?[0-9]*|\.[0-9]+)(?:[eE][-+]?[0-9]+)?|true|false)'
)


def text_of(chunk: str) -> str:
    return html.unescape(TAG_RE.sub(" ", chunk)).strip()


def resolve(receipt: dict, dotted: str):
    node = receipt
    for key in dotted.lstrip(".").split("."):
        if not isinstance(node, dict) or key not in node:
            raise KeyError(dotted)
        node = node[key]
    return node


def values_agree(stated: str, actual) -> bool:
    if stated.startswith('"') and stated.endswith('"'):
        return stated[1:-1] == str(actual)
    low = stated.lower()
    if low in ("true", "false"):
        return isinstance(actual, bool) and (low == "true") == actual
    try:
        number = float(stated)
    except ValueError:
        return False
    if isinstance(actual, bool) or actual is None:
        return False
    try:
        target = float(actual)
    except (TypeError, ValueError):
        return False
    if math.isnan(number) or math.isnan(target):
        return math.isnan(number) and math.isnan(target)
    if number == target:
        return True
    return math.isclose(number, target, rel_tol=1e-9, abs_tol=1e-12)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fragment", default=str(DEFAULT_FRAGMENT))
    args = parser.parse_args(argv)
    fragment = Path(args.fragment)

    try:
        revision = (
            subprocess.run(
                ["git", "-C", str(REPO_ROOT), "rev-parse", "--short", "HEAD"],
                capture_output=True,
                text=True,
                check=False,
            ).stdout.strip()
            or "unknown"
        )
    except OSError:
        revision = "unknown"
    print(f"# revision={revision} tree={REPO_ROOT} fragment={fragment}")

    if not fragment.is_file():
        print(f"FAIL fragment-present missing {fragment}")
        print("EXIT=1")
        return 1

    body = fragment.read_text(encoding="utf-8")
    results: list[tuple[str, bool, str]] = []

    def note(ident: str, ok: bool, detail: str = "") -> None:
        results.append((ident, ok, detail))

    seen: dict[str, dict] = {}
    for attrs, inner in FIGURE_RE.findall(body):
        attr = dict(ATTR_RE.findall(attrs))
        panel = attr.get("panel")
        if panel not in PANELS:
            note(
                f"figure::{panel or 'unknown'}::known-panel",
                False,
                panel or "no data-panel",
            )
            continue
        seen[panel] = {"attr": attr, "inner": inner}

    for panel, spec in PANELS.items():
        if panel not in seen:
            note(f"{panel}::figure-present", False, "no figure block")
            continue
        note(f"{panel}::figure-present", True)
        block = seen[panel]
        receipt = json.loads(
            (HERE / panel / "boundary-wall-proximity.json").read_text(encoding="utf-8")
        )

        gate = block["attr"].get("gate", "")
        note(
            f"{panel}::data-gate",
            gate == spec["gate_token"],
            f"expected {spec['gate_token']!r} got {gate!r}",
        )

        title_match = H3_RE.search(block["inner"])
        title = text_of(title_match.group(1)) if title_match else ""
        note(
            f"{panel}::title-names-gate",
            spec["gate_phrase"].lower() in title.lower(),
            f"title {title!r} does not name {spec['gate_phrase']!r}",
        )

        cap_match = CAPTION_RE.search(block["inner"])
        caption = text_of(cap_match.group(1)) if cap_match else ""
        bindings = BINDING_RE.findall(caption)
        note(
            f"{panel}::caption-has-bindings",
            len(bindings) > 0,
            "no `path = value` binding",
        )

        for dotted, stated in bindings:
            ident = f"{panel}::binding::{dotted}"
            try:
                actual = resolve(receipt, dotted)
            except KeyError:
                note(ident, False, "path not present in receipt")
                continue
            ok = values_agree(stated, actual)
            note(ident, ok, f"caption={stated!r} receipt={actual!r}")

    failed = 0
    for ident, ok, detail in results:
        if ok:
            print(f"PASS {ident}")
        else:
            failed += 1
            print(f"FAIL {ident} {detail}")
    print(f"# {len(results) - failed}/{len(results)} passed")
    print(f"EXIT={1 if failed else 0}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())