"""Remove geometry identity checks in a scratch arm and run their refusal tests."""

from pathlib import Path
import sys

from scripts.iron_graded_refinement.gate import arm


def main():
    tree, revision, output = Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3])
    module = tree / "nova/biot/magnetisation.py"
    text = module.read_text()
    start = text.index("        if (\n", text.index("    def validate("))
    stop = text.index("            raise ValueError", start)
    text = text[:start] + "        if False:\n" + text[stop:]
    marker = '    if metadata["geometry_digest"] != _geometry_digest(cells, basis):'
    assert marker in text
    module.write_text(text.replace(marker, "    if False:"))
    receipt = arm(
        "geometry-mutant",
        tree,
        revision,
        [
            "-m",
            "pytest",
            "-p",
            "no:cacheprovider",
            "-q",
            "tests/test_magnetisation.py",
            "-k",
            "persisted_interaction_refuses_different_mesh",
        ],
        output,
    )
    assert receipt["exit"] != 0, "geometry mutation escaped the refusal tests"


if __name__ == "__main__":
    main()
