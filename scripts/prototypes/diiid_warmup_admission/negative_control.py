"""Restore unconditional warm-up propagation in an archived copy and test it."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import tempfile

from measure import extract
from regression import NODE_TEST, gate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    repository = Path(__file__).resolve().parents[3]
    revision = subprocess.check_output(
        ["git", "-C", str(repository), "rev-parse", "HEAD"], text=True
    ).strip()
    with tempfile.TemporaryDirectory(
        prefix="warmup-unconditional-", dir=os.environ["TMPDIR"]
    ) as temporary:
        tree = Path(temporary) / "tree"
        extract(tree, revision, repository)
        module = tree / "nova/equilibrium/fixed_point.py"
        text = module.read_text()
        guarded = "candidate = jnp.where(finite_proposal, proposal, state)"
        assert text.count(guarded) == 1
        text = text.replace(guarded, "candidate = proposal")
        text = text.replace(
            "current_shadow = jnp.where(\n"
            "                finite_proposal, proposed_shadow, previous_shadow\n"
            "            )",
            "current_shadow = proposed_shadow",
        )
        text = text.replace(
            "next_image = jnp.where(finite_proposal, proposed_image, mapped)",
            "next_image = proposed_image",
        )
        module.write_text(text)
        result = gate(tree, revision, NODE_TEST, args.output, quiet=True)
        assert result["exit_status"] != 0
        assert result["failure_ids"]
        print("CONTROL_REFUSED=" + ",".join(result["failure_ids"]), flush=True)


if __name__ == "__main__":
    main()
