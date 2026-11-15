"""Check retained scenario receipts and the published artifact inventory."""

import argparse
import copy
import hashlib
from html.parser import HTMLParser
import json
import math
from pathlib import Path
import re
import shlex
import subprocess
import sys
import xml.etree.ElementTree as ET

from PIL import Image


class Inventory(HTMLParser):
    def __init__(self):
        super().__init__()
        self.ids, self.images = [], []

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if "id" in attrs:
            self.ids.append(attrs["id"])
        if tag == "img":
            self.images.append(attrs["src"])


def check_receipt(receipt):
    assert receipt["complete"] and receipt["width"] == 1
    assert [r["index"] for r in receipt["slices"]] == list(range(129, 149))
    for i, row in enumerate(receipt["slices"]):
        assert row["warm_start_from"] == (None if i == 0 else row["index"] - 1)
        value = row["fresh_residual"][0]
        if row["converged"][0]:
            assert math.isfinite(value) and value <= 1e-10, (
                f"slice {row['index']} claims convergence at residual {value}"
            )
            assert row["fresh_valid"][0]
        else:
            assert row["failure_stage"][0] != "converged"
    count = sum(all(r["converged"]) for r in receipt["slices"])
    assert count == receipt["converged_slices"]
    return count


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--figures", type=Path, required=True)
    parser.add_argument("--fragment", type=Path, required=True)
    parser.add_argument("--preview", type=Path, required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--mutate-failure", action="store_true")
    args = parser.parse_args()
    print("MODULE", __file__, "CWD", Path.cwd(), flush=True)
    measured = json.loads((args.results / "width-1-process-1/receipt.json").read_text())
    if args.mutate_failure:
        mutated = copy.deepcopy(measured)
        mutated["slices"][2]["converged"][0] = True
        check_receipt(mutated)
        raise AssertionError("false convergence escaped the receipt guard")
    for process in range(3):
        root = args.results / f"width-1-process-{process}"
        receipt = json.loads((root / "receipt.json").read_text())
        assert check_receipt(receipt) == 16
        assert (root / "gate.log").read_text().splitlines()[-1] == "EXIT=0"
        logs = (root / "compilations.log").read_text()
        assert len(re.findall(r"Compiling jit\(solve\)", logs)) == 1
    incomplete = json.loads(
        (args.results / "width-16-process-0/receipt.json").read_text()
    )
    assert not incomplete["complete"] and not incomplete["slices"]
    refused = (args.results / "width-16-process-0/gate.log").read_text()
    assert "RESOURCE_EXHAUSTED" in refused and refused.splitlines()[-1] == "EXIT=1"
    controls = json.loads(
        (args.results / "width-1-process-0/negative-controls.json").read_text()
    )
    assert controls["closure_compilations"] == [1, 1]
    assert controls["updates"] == [0] and controls["converged"] == [False]
    assert controls["residual"][0] > 1e-10
    assert controls["with_failed_fraction"] < controls["normal_fraction"]
    inventory = Inventory()
    inventory.feed(args.fragment.read_text())
    assert len(inventory.ids) == len(set(inventory.ids))
    assert all(i.startswith("dina-flattop-scenario-") for i in inventory.ids)
    assert len(inventory.images) == 22
    prefix = "/nova/figures/sub-millisecond-forward-path/dina-flattop-scenario/"
    for url in inventory.images:
        assert url.startswith(prefix)
        assert (args.figures / url.removeprefix(prefix)).is_file()
    digests = {}
    for index in range(129, 149):
        name = f"slice-{index:04d}"
        svg = ET.parse(args.figures / f"{name}.svg").getroot()
        assert svg.attrib["width"] == "31.1cm" and svg.attrib["height"] == "16cm"
        path = args.preview / f"{name}.png"
        with Image.open(path) as preview:
            assert preview.size == (1400, 720)
            preview.verify()
        digests[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    with Image.open(args.figures / "flattop.gif") as animation:
        assert animation.n_frames == 20 and animation.size == (1400, 720)
        duration = 0
        for index in range(animation.n_frames):
            animation.seek(index)
            duration += animation.info["duration"]
        assert duration == 10000
    render = json.loads((args.preview / "render-receipt.json").read_text())
    assert render["contact"]["pulse_frame_indices"] == [0, 2, 3, 5, 7, 19]
    assert len(render["inspections"]) == 20
    assert all(
        r["clipped_text"] == r["overlapping_text"] == 0 for r in render["inspections"]
    )
    command = [
        sys.executable,
        "-m",
        "scripts.prototypes.dina_flattop_scenario.verify",
        *sys.argv[1:],
        "--mutate-failure",
    ]
    log = args.preview / "false-convergence.log"
    with log.open("w") as stream:
        stream.write(
            f"# revision: {args.revision} tree: {Path.cwd()} "
            f"command: {shlex.join(command)}\n"
        )
        stream.flush()
        result = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT)
        stream.write(f"EXIT={result.returncode}\n")
    assert result.returncode != 0
    assert "slice 131 claims convergence at residual" in log.read_text()
    report = dict(
        revision=args.revision,
        complete=True,
        numerical_goal_met=False,
        width_one_slices=20,
        converged=16,
        width_sixteen_slices=0,
        svg_count=20,
        gif_frames=20,
        contact_indices=[129, 131, 132, 134, 136, 148],
        negative_control_exit=result.returncode,
        preview_sha256=digests,
    )
    (args.preview / "verification.json").write_text(json.dumps(report, indent=2) + "\n")
    print(
        "ARTIFACTS_VERIFIED slices=20 converged=16 svg=20 gif_frames=20 "
        "false_convergence_refused=True numerical_goal_met=False",
        flush=True,
    )


if __name__ == "__main__":
    main()
