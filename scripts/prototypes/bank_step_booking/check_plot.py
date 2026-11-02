"""Check that the rendered stage legend names its measured configuration."""

import argparse
from pathlib import Path
import xml.etree.ElementTree as ET


def check(path: Path) -> None:
    root = ET.parse(path).getroot()
    labels = [
        "".join(element.itertext()).strip()
        for element in root.iter()
        if element.tag.endswith("}text") or element.tag == "text"
    ]
    print("SVG_LABELS", labels, flush=True)
    assert "batched baseline" in labels, "SVG text reader missed the known baseline"
    assert "density basis, four cut lanes" in labels, "green legend misnames the result"
    assert "shared boundary" not in labels, "stale green legend remains"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("svg", type=Path)
    check(parser.parse_args().svg)
